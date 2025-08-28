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
import sys
import csv
import math
import argparse
import torch
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import rerun as rr
import genesis as gs
from scipy.spatial.transform import Rotation as R


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
    """Best-effort estimate of Robotiq opening width [m]."""
    candidates = [
        "finger_joint1", "finger_joint2",
        "left_inner_finger_joint", "right_inner_finger_joint",
        "left_outer_knuckle_joint", "right_outer_knuckle_joint",
        "left_inner_knuckle_joint", "right_inner_knuckle_joint",
    ]
    vals = []
    for name in candidates:
        try:
            j = robot.get_joint(name)
            if j is not None:
                q = float(j.qpos()) if hasattr(j, "qpos") else float(j.get_qpos())
                vals.append(q)
        except Exception:
            pass
    if not vals:
        try:
            for j in robot.get_joints():
                jname = getattr(j, "name", "")
                if "finger" in jname:
                    q = float(j.qpos()) if hasattr(j, "qpos") else float(j.get_qpos())
                    vals.append(q)
        except Exception:
            return 0.0
    return float(np.clip(np.sum(vals), 0.0, 0.14))


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


# --------------------------- Main -------------------------------

def main():
    args = parse_args()

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
    cube_size = (0.07, 0.07, 0.07)
    cube = scene.add_entity(
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
                pos=(0.0, 0.0, 1.0),
                lookat=(0.0, 0.0, 0.0),
                # fov=40,
                GUI=False,
            )

            # Build an offset transform: translation + rotation relative to link
            offset_T = np.eye(4, dtype=np.float32)
            offset_T[:3, 3] = np.array([-0.5, 0.0, 0.5], dtype=np.float32) 

            # rotation: tilt down around X-axis by 45°
            theta = np.deg2rad(-25)  # negative = look downward
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
    
    home_qpos = np.array([0, -math.pi/2, math.pi/2, -math.pi/2, -math.pi/2, 0], dtype=np.float32)
    robot.set_dofs_position(home_qpos, motors_dof_idx)
    
    if args.traj is not None:
        path = load_csv_traj(args.traj)
    else:
        q_goal = robot.inverse_kinematics(link=ee, pos=np.array([[-0.65, -0.5, 0.5]], dtype=np.float32))
        path = robot.plan_path(qpos_goal=q_goal)
        if path is None or len(path) == 0:
            raise RuntimeError("Path planning failed; provide --traj CSV instead.")
        path = path.cpu().numpy()
    
    dt = float(scene.sim_options.dt)
    t = 0.0
    step = 0
    cam_every = max(1, int(round(1.0 / (dt * max(1e-9, args.cam_fps))))) if (args.add_camera and cam is not None) else 0

    last_cmd = home_qpos.copy()

    total_steps = len(path)
    if args.duration is not None:
        total_steps = min(total_steps, int(round(args.duration / dt)))

    for i in range(total_steps):
        try:
            q_cmd = path[i]
            robot.control_dofs_position(q_cmd)
            last_cmd = q_cmd
        except Exception:
            q_cmd = last_cmd
            robot.control_dofs_position(q_cmd)

        scene.step()
        t += dt
        step += 1

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

        # Update camera pose to follow wrist
        if cam is not None:
            cam.move_to_attach()

            if step % cam_every == 0:
                rgb, _, _, _ = cam.render()  # returns RGB (HxWx3 or HxWx4 float/uint8)
                rgb = np.asarray(rgb)
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                if rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
                logger.log_image(rgb)

    out_path = logger.save()
    print(f"Saved Rerun recording: {out_path}")


if __name__ == "__main__":
    main()
