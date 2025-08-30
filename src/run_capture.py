#!/usr/bin/env python3
"""
Run a Genesis simulation of UR10e + Robotiq, execute a simple trajectory,
log synchronized streams to Rerun, and save a single .rrd file.

Usage examples:
  python run_capture.py \
    --robot xml/universal_robots_ur10e/ur10e_robotiq.xml \
    --outdir outputs/dataset \
    --add-camera \
    --cam-fps 30

  python run_capture.py --traj configs/sample_traj.csv

Outputs:
  dataset/session_<YYYYMMDD_HHMMSS>.rrd

Notes:
- Uses simulation time as the only clock.
- Logs: joint q/dq, target commands, EE pose, gripper width, optional RGB.
- Camera is optional; if unavailable in your Genesis build, disable --add-camera.
"""
from __future__ import annotations
import os
import csv
import math
import argparse
import torch
import numpy as np
import rerun as rr
import genesis as gs
from datetime import datetime
from typing import List, Optional, Tuple


# --- Pick & place tuning params ---
HOVER_IN   = 0.8   # pre-grasp / pre-place hover height above surface [m]
HOVER_OUT  = 0.8   # post-grasp / post-place retreat height [m]
MARGIN_Z   = 0.28  # how close to the top surface we descend before closing/opening [m]
DWELL_STEPS = 50    # small hold after closing before lifting (lets contacts settle)
STEPS_PER_SEGMENT = 150  # IK interpolation per segment (approach/lift/place)
HOME_QPOSE = np.array([math.pi/2, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0], dtype=np.float32)

# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genesis UR10e + Robotiq → Rerun capture")
    p.add_argument("--robot", default="./assets/xml/universal_robots_ur10e/ur10e_robotiq.xml",
                   help="Path to MJCF/XML robot (UR10e + Robotiq)")
    p.add_argument("--traj", default=None,
                   help="Optional CSV of joint targets (columns: 6 joint angles in rad)")
    p.add_argument("--outdir", default="outputs/dataset", help="Output directory for .rrd")
    p.add_argument("--session", default=None, help="Override session name (without .rrd)")
    p.add_argument("--dt", type=float, default=0.01, help="Physics timestep [s]")
    p.add_argument("--add-camera", default=True, action="store_true", help="Attach an RGB camera and log frames")
    p.add_argument("--cam-fps", type=float, default=30.0, help="Target camera FPS (approx)")
    p.add_argument("--cam-res", type=int, nargs=2, default=[640, 480], metavar=("W","H"),
                   help="Camera resolution WxH")
    p.add_argument("--duration", type=float, default=None,
                   help="Optional max duration [s] (overrides trajectory length)")
    return p.parse_args()

# -------- Gripper helpers (best-effort for Robotiq 2F) --------

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
                              hover_in: float = HOVER_IN,
                              hover_out: float = HOVER_OUT,
                              margin_z: float = MARGIN_Z,
                              dwell_steps: int = DWELL_STEPS) -> Tuple[List[np.ndarray], List[int]]:
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
                                steps_per_segment: int = STEPS_PER_SEGMENT,
                                dwell_steps: int = DWELL_STEPS):
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

# ------------------------- Helpers/Types ------------------------

UR10E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


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


def get_gripper_width(robot) -> float:
    """
    Estimate Robotiq 2F-85 opening width [m] from driver joint positions.
    The XML defines 'left_driver_joint' and 'right_driver_joint' as actuated.
    Their range is 0..0.8 (per joint). Full opening is about 0.085 m.
    """
    vals = []
    for j in robot.joints:
        if j.name in ["left_driver_joint", "right_driver_joint"]:
            try:
                q = float(j.get_qpos()) if hasattr(j, "get_qpos") else float(j.qpos())
                vals.append(q)
            except Exception:
                pass

    if not vals:
        return 0.0

    # Use the average of the two driver joints
    avg_q = np.mean(vals)

    # Scale joint value (0..0.8) → physical width (~0..0.085 m for Robotiq 2F-85)
    width = (avg_q / 0.8) * 0.085
    return float(np.clip(width, 0.0, 0.085))



# ------------------------- Rerun Logging ------------------------

class RerunLogger:
    def __init__(self, session: str, outdir: str):
        self.session = session
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        rr.init(session, spawn=False)

    def set_time(self, t: float):
        rr.set_time_seconds("sim_time", float(t))

    def log_joint_state(self, q: torch.Tensor, dq: torch.Tensor):
        rr.log("robot/state/q", rr.Tensor(q.cpu().numpy()))
        rr.log("robot/state/dq", rr.Tensor(dq.cpu().numpy()))

    def log_cmd(self, q_cmd: np.ndarray):
        rr.log("commands/q_target", rr.Tensor(np.asarray(q_cmd)))

    def log_gripper(self, width_m: float):
        rr.log(
            "gripper/state/width",
            rr.Tensor(np.array([float(width_m)], dtype=np.float32))
        )

    def log_ee(self, pos_xyz: np.ndarray, quat_xyzw: np.ndarray):
        rr.log("world/ee", rr.Transform3D(
            translation=pos_xyz,
            rotation=rr.Quaternion(xyzw=quat_xyzw),
        ))

    def log_image(self, rgb: np.ndarray):
        rr.log("cam/rgb", rr.Image(rgb))

    def save(self) -> str:
        path = os.path.join(self.outdir, f"{self.session}.rrd")
        rr.save(path)
        return path


def prepare_env(args):
        session = args.session or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = RerunLogger(session, args.outdir)

        gs.init(backend=gs.gpu)

        scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=args.dt),
            show_viewer=True,
        )

        plane = scene.add_entity(gs.morphs.Plane())
        robot = scene.add_entity(gs.morphs.MJCF(file=args.robot))

        # Cube: center-positioned; set z to half the height so it sits on the plane
        cube_size = (0.1, 0.04, 0.05)
        _cube = scene.add_entity(
            gs.morphs.Box(
                size=cube_size,                         # (x, y, z) extents in meters
                pos=(-0.65, -0.5, cube_size[2] / 2.0),  # center position; z = half height
                fixed=False,                            # dynamic body (not fixed to world)
                collision=True,                         # participates in collisions
                visualization=True,                     # render it
                contype=0xFFFF,
                conaffinity=0xFFFF,
                # quat/euler optional; if used, quat is (w, x, y, z)
            )
        )

        motors_dof_idx = find_motors_idx(robot, UR10E_JOINTS)

        ee = robot.get_link('wrist_3_link')
        if ee is None:
            raise RuntimeError("End-effector link 'wrist_3_link' not found; adjust name to your MJCF")

        # Free camera that will follow the wrist if enabled
        cam = None
        if args.add_camera:
            try:
                cam = scene.add_camera(
                    res=tuple(args.cam_res),
                    pos=(0.0, 0.0, 0.0),
                    lookat=(0.0, 0.0, 0.0),
                    fov=50,
                    GUI=False,
                )

                # Build an offset transform: translation + rotation relative to link
                offset_T = np.eye(4, dtype=np.float32)
                offset_T[:3, 3] = np.array([0.0, 0.0, 0.2], dtype=np.float32) 

                # rotation: tilt down around X-axis
                theta = np.deg2rad(60)  
                R_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta),  np.cos(theta)]
                ], dtype=np.float32)

                offset_T[:3, :3] = R_x

                cam.attach(ee, offset_T)
            except Exception as e:
                print(f"[WARN] Camera creation failed: {e}")
                cam = None

        scene.build(n_envs=1) # TODO: n_envs > 1 is not tested
        
        home_qpos = HOME_QPOSE
        robot.set_dofs_position(home_qpos, motors_dof_idx)

        return scene, robot, ee, _cube, home_qpos, motors_dof_idx, logger, cam

def get_path(robot, ee, _cube, motors_dof_idx):
    # 1) Ensure gripper open
    set_gripper(robot, open_frac=0.0)

    # 2) Fix EE orientation (keep current wrist orientation, which should be tool-z vertical)
    ee_quat_wxyz = ee.get_quat().cpu().numpy()[0].astype(np.float32)

    # 3) Cube & place
    cube_pos = _cube.get_pos().cpu().numpy()[0].astype(np.float32)
    place_xyz = np.array([-0.30, 0.40, 0.035], dtype=np.float32)  # top surface target

    # 4) Waypoints for full pick-and-place
    waypoints_xyz, seg_events = make_pick_place_waypoints(
        cube_pos=cube_pos,
        place_xyz=place_xyz,
        hover_in=HOVER_IN,
        hover_out=HOVER_OUT,
        margin_z=MARGIN_Z,
        dwell_steps=DWELL_STEPS,
    )

    # 5) Path for those segments (+ dwell) and event steps
    path, events = build_pick_place_joint_path(
        robot, ee,
        waypoints_xyz=waypoints_xyz,
        quat_wxyz=ee_quat_wxyz,
        steps_per_segment=STEPS_PER_SEGMENT,
        dwell_steps=DWELL_STEPS,
    )

    # 6) Append a return-to-home at the end
    home_qpos = HOME_QPOSE

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



def add_to_logger(logger, robot, ee, motors_dof_idx, t, q_cmd):
    q = robot.get_dofs_position(dofs_idx_local=motors_dof_idx)[0]
    dq = robot.get_dofs_velocity(dofs_idx_local=motors_dof_idx)[0]

    ee_pos = ee.get_pos().cpu().numpy()[0]
    ee_quat_wxyz = ee.get_quat().cpu().numpy()[0]
    ee_quat = np.array([ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]], dtype=np.float32)

    width_m = get_gripper_width(robot)

    logger.set_time(t)
    logger.log_joint_state(q, dq)
    logger.log_cmd(q_cmd)
    logger.log_gripper(width_m)
    logger.log_ee(ee_pos, ee_quat)

# --------------------------- Main -------------------------------

def main():
    args = parse_args()

    scene, robot, ee, _cube, home_qpos, motors_dof_idx, logger, cam = prepare_env(args=args)

    path, events = get_path(robot, ee, _cube, motors_dof_idx)
    
    dt = float(scene.sim_options.dt)
    t = 0.0
    step = 0
    cam_every = max(1, int(round(1.0 / (dt * max(1e-9, args.cam_fps))))) if (args.add_camera and cam is not None) else 0

    last_cmd = home_qpos.copy()

    total_steps = len(path)
    if args.duration is not None:
        total_steps = min(total_steps, int(round(args.duration / dt)))

    close_step = events["close_step"]
    open_step  = events["open_step"]
    
    for i in range(total_steps):
        try:
            q_cmd = path[i]
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
        # TODO: make the logic proper
        if step == open_step:
            set_gripper(robot, open_frac=0.0)
        elif step == close_step:
            set_gripper(robot, open_frac=1.0)

        scene.step()

        # ★ log at current t (post-step), then advance time
        add_to_logger(logger, robot, ee, motors_dof_idx, t, q_cmd)
        t += dt  # ★ moved after logging

        # Camera follow & capture
        if cam is not None:
            cam.move_to_attach()
            if cam_every and (i % cam_every == 0):  # ★ use i for stable cadence
                rgb, _, _, _ = cam.render()
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                if rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
                logger.log_image(rgb)

        step += 1  # ★ move to the end (optional but tidy)

    out_path = logger.save()
    print(f"Saved Rerun recording: {out_path}")


if __name__ == "__main__":
    main()
