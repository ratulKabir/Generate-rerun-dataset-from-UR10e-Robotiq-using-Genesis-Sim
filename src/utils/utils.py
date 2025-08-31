import csv
import numpy as np
import genesis as gs
from datetime import datetime
from typing import List, Optional, Tuple

from utils.rerun_utils import RerunLogger


UR10E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

def get_driver_dof_indices(robot):
    idx = []
    for j in robot.joints:
        if j.name in ["left_driver_joint", "right_driver_joint"]:
            idx.append(int(j.dofs_idx_local[0]))
    if not idx:
        raise RuntimeError("Driver joints not found")
    return idx

def set_gripper(robot, open_frac: float):
    open_frac = float(np.clip(open_frac, 0.0, 1.0))
    idxs = get_driver_dof_indices(robot)
    q = [open_frac] * len(idxs)   # symmetric opening
    robot.control_dofs_position(np.array(q, dtype=np.float32), dofs_idx_local=idxs)

# -------- IK path builder over Cartesian waypoints --------
def cartesian_waypoint_path(robot, ee, waypoints_xyz: List[np.ndarray],
                            quat_wxyz: Optional[np.ndarray],
                            steps_per_segment: int = 60) -> np.ndarray:
    """
    Sequential IK to follow Cartesian points with (optional) fixed orientation.
    Returns a (N, ndof) numpy array of joint configs.
    """
    path = []
    q_curr = robot.get_dofs_position()[0]  # current full dofs (actuated)
    for i in range(len(waypoints_xyz) - 1):
        a = waypoints_xyz[i]
        b = waypoints_xyz[i + 1]
        for s in range(steps_per_segment):
            u = (s + 1) / steps_per_segment
            p = (1 - u) * a + u * b
            if quat_wxyz is not None:
                qpos = robot.inverse_kinematics(link=ee, pos=p[None, :], quat=quat_wxyz[None, :])
            else:
                qpos = robot.inverse_kinematics(link=ee, pos=p[None, :])
            if qpos is None:
                # keep last known pose if IK fails momentarily
                qnext = q_curr
            else:
                qnext = qpos[0].cpu().numpy()
                q_curr = qnext
            path.append(qnext)
    return np.asarray(path, dtype=np.float32)


def make_pick_place_waypoints(cube_pos: np.ndarray,
                              place_xyz: np.ndarray,
                              hover_in: float,
                              hover_out: float,
                              margin_z: float) -> Tuple[List[np.ndarray], List[int]]:
    """
    Build Cartesian waypoints (EE positions) for:
      1) pregrasp (hover above cube, open)
      2) descend to just above cube top (open)
      3) dwell at same pose to close gripper (open->close happens here)
      4) postgrasp (lift up, closed)
      5) preplace (move above place target, closed)
      6) descend to just above place target (closed)
      7) postplace (lift up, open after release)
      8) (home handled separately)

    Returns:
      waypoints_xyz: list of np.array([x,y,z]) in sequence
      event_markers: [close_at_step, open_at_step] as segment indices (for timing)
    """
    # infer top surface z from cube center + half size known in prepare_env (0.07 → half 0.035)
    cube_top_z  = cube_pos[2]            # you already place cube center at z = half size
    place_top_z = place_xyz[2]           # you set this to 0.035 in get_path()

    pregrasp   = np.array([cube_pos[0],  cube_pos[1],  cube_top_z  + hover_in],  np.float32)
    grasp      = np.array([cube_pos[0],  cube_pos[1],  cube_top_z  + margin_z], np.float32)

    postgrasp  = np.array([cube_pos[0],  cube_pos[1],  cube_top_z  + hover_out], np.float32)

    preplace   = np.array([place_xyz[0], place_xyz[1], place_top_z + hover_in], np.float32)
    place      = np.array([place_xyz[0], place_xyz[1], place_top_z + margin_z], np.float32)

    postplace  = np.array([place_xyz[0], place_xyz[1], place_top_z + hover_out], np.float32)

    waypoints_xyz: List[np.ndarray] = [
        pregrasp,   # seg 0: pregrasp -> grasp
        grasp,      # seg 1: grasp -> postgrasp
        postgrasp,  # seg 2: postgrasp -> preplace
        preplace,   # seg 3: preplace -> place
        place,      # seg 4: place -> postplace
        postplace,
    ]

    # We’ll handle dwell by inserting repeated points into the joint path (below),
    # but we also want clean indices to trigger close/open while stationary:
    # Segment indices (0-based) aligned with 'waypoints_xyz' above.
    close_segment = 0   # after reaching 'grasp' and before lifting
    open_segment  = 3   # after reaching 'place' (before postplace)

    return waypoints_xyz, [close_segment, open_segment]


def build_pick_place_joint_path(robot, ee,
                                waypoints_xyz: List[np.ndarray],
                                quat_wxyz: Optional[np.ndarray],
                                steps_per_segment: int,
                                dwell_steps: int):
    """
    IK path between waypoints. Adds a dwell AFTER reaching 'grasp' so we can close while stationary.
    Returns:
      path:   [T, ndof]
      events: {'close_step': int, 'open_step': int}
    """
    per_seg_paths: List[np.ndarray] = []
    seg_start_step_idx: List[int] = []
    tcursor = 0

    for si in range(len(waypoints_xyz) - 1):
        seg_start_step_idx.append(tcursor)
        a, b = waypoints_xyz[si], waypoints_xyz[si + 1]
        seg_path = cartesian_waypoint_path(
            robot, ee, [a, b],
            quat_wxyz=quat_wxyz,
            steps_per_segment=steps_per_segment
        )
        per_seg_paths.append(seg_path)
        tcursor += len(seg_path)

    ndof = robot.get_dofs_position().shape[-1]
    path = np.concatenate(per_seg_paths, axis=0) if per_seg_paths else np.empty((0, ndof), dtype=np.float32)

    # Insert dwell right after finishing segment 0 (pregrasp->grasp)
    grasp_seg_idx = 0
    dwell_added = 0
    insert_at = None
    if per_seg_paths and dwell_steps > 0:
        insert_at = seg_start_step_idx[grasp_seg_idx] + per_seg_paths[grasp_seg_idx].shape[0]
        if 0 < insert_at <= path.shape[0]:
            dwell_pose = path[insert_at - 1]
            dwell_block = np.repeat(dwell_pose[None, :], dwell_steps, axis=0)
            path = np.concatenate([path[:insert_at], dwell_block, path[insert_at:]], axis=0)
            dwell_added = dwell_steps
        else:
            insert_at = None

    # Helper: end idx of segment k in the (possibly extended) path
    def end_of_segment(seg_idx: int) -> int:
        end = 0
        for k in range(seg_idx + 1):
            end += per_seg_paths[k].shape[0]
            if k == grasp_seg_idx:
                end += dwell_added
        return max(0, end - 1)

    # Fire CLOSE **mid-dwell** so it's clearly after arrival & stationary
    if insert_at is not None:
        close_step = insert_at + max(0, dwell_steps // 2)
    else:
        close_step = end_of_segment(0)  # fallback

    # Fire OPEN after finishing segment 3 (preplace->place)
    open_step = end_of_segment(3)

    events = {"close_step": int(close_step), "open_step": int(open_step)}
    return path, events


def load_csv_traj(path: str) -> np.ndarray:
    rows: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            if not r:
                continue
            vals = [float(x) for x in r[:6]]
            rows.append(vals)
    if not rows:
        raise ValueError(f"Empty trajectory CSV: {path}")
    return np.asarray(rows, dtype=np.float32)

def find_motors_idx(robot, joint_names: Tuple[str, ...]) -> List[int]:
    idx = []
    for name in joint_names:
        j = robot.get_joint(name)
        if j is None:
            raise RuntimeError(f"Joint not found: {name}")
        idx.append(int(j.dofs_idx_local[0]))
    return idx

def prepare_env(cfg):
    session = cfg.session or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = RerunLogger(session, cfg.outdir)
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5), 
            camera_lookat=(0.0, 0.0, 0.5), 
            camera_fov=30, 
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=cfg.dt),
        show_viewer=(not cfg.headless),
    )

    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.MJCF(file=cfg.robot_xml))

    cube_z = cfg.cube_size[2] / 2.0
    cube = scene.add_entity(gs.morphs.Box(
        size=tuple(cfg.cube_size),
        pos=(cfg.cube_pos[0], cfg.cube_pos[1], cube_z),
        fixed=False, collision=True, visualization=True, contype=0xFFFF, conaffinity=0xFFFF
    ))

    motors_idx = find_motors_idx(robot, UR10E_JOINTS)
    ee = robot.get_link('wrist_3_link')

    cam = None
    if cfg.add_camera:
        cam = scene.add_camera(res=tuple(cfg.cam_res), pos=(0,0,0), lookat=(0,0,0), fov=50, GUI=False)
        # offset
        offset_T = np.eye(4, dtype=np.float32)
        offset_T[:3, 3] = np.array(cfg.cam_offset_xyz, dtype=np.float32)
        th = np.deg2rad(cfg.cam_tilt_deg)
        R_x = np.array([[1,0,0],[0,np.cos(th),-np.sin(th)],[0,np.sin(th),np.cos(th)]], dtype=np.float32)
        offset_T[:3, :3] = R_x
        cam.attach(ee, offset_T)

    scene.build(n_envs=1)
    robot.set_dofs_position(np.array(cfg.home_qpose, dtype=np.float32), motors_idx)
    return scene, robot, ee, cube, motors_idx, logger, cam

def get_path(cfg, robot, ee, cube, motors_dof_idx):
    set_gripper(robot, open_frac=0.0)
    ee_quat_wxyz = ee.get_quat().cpu().numpy()[0].astype(np.float32)

    cube_pos = cube.get_pos().cpu().numpy()[0].astype(np.float32)
    place_xyz = np.array(cfg.place_xyz, dtype=np.float32)

    waypoints_xyz, _ = make_pick_place_waypoints(
        cube_pos=cube_pos, place_xyz=place_xyz,
        hover_in=cfg.hover_in, hover_out=cfg.hover_out,
        margin_z=cfg.margin_z
    )

    path, events = build_pick_place_joint_path(
        robot, ee, waypoints_xyz=waypoints_xyz, quat_wxyz=ee_quat_wxyz,
        steps_per_segment=cfg.steps_per_segment, dwell_steps=cfg.dwell_steps,
    )

    # 6) Append a return-to-home at the end
    home_qpos = cfg.home_qpose

    tail = []
    if path.shape[0] > 0:
        q_curr = path[-1]  # full actuated dof vector (e.g., 14)
        # Build a full-length target that keeps non-arm joints as-is
        home_full = q_curr.copy()
        home_full[motors_dof_idx] = home_qpos  # set only the 6 arm joints

        for s in range(120):  # ~1–2s blend
            u = (s + 1) / 120.0
            tail.append((1 - u) * q_curr + u * home_full)

    if tail:
        path = np.concatenate([path, np.asarray(tail, dtype=np.float32)], axis=0)

    return path, events


def manipulate_robot(robot, path, step, motors_dof_idx, events):
    try:
        q_cmd = path[step]
        robot.control_dofs_position(
            q_cmd[motors_dof_idx],
            dofs_idx_local=motors_dof_idx
        )
        last_cmd = q_cmd
    except Exception:
        robot.control_dofs_position(
            last_cmd[motors_dof_idx],
            dofs_idx_local=motors_dof_idx
        )

    # --- gripper events BEFORE stepping ---
    close_step = events["close_step"]
    open_step  = events["open_step"]

    gripper_status = None
    if step == open_step:
        gripper_status = 0.0
        set_gripper(robot, open_frac=gripper_status)
    elif step == close_step:
        gripper_status = 1.0
        set_gripper(robot, open_frac=gripper_status)

    return q_cmd, gripper_status

def cam_follow_arm_and_log(cam_every, cam, logger, idx):
    if cam is not None:
        cam.move_to_attach()
        if cam_every and (idx % cam_every == 0):  # ★ use i for stable cadence
            rgb, _, _, _ = cam.render()
            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            logger.log_image(rgb)