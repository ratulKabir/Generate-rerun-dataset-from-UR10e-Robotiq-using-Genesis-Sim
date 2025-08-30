#!/usr/bin/env python3
"""
Generate a Rerun dataset from a Genesis simulation of **UR10e + Robotiq 2F**.

This script:
  • Spawns a UR10e with a Robotiq gripper in Genesis
  • Runs a smooth pick-and-place trajectory (IK over Cartesian waypoints)
  • Logs synchronized streams to Rerun using **simulation time** as the single clock
  • Exports a single `.rrd` file for clean, replayable visualization

Configuration:
  • YAML-first at `src/configs/default.yaml` (robot XML, dt, camera, waypoints, etc.)
  • CLI overrides take precedence over YAML (see examples below)

Logged streams (namespaces):
  • robot/state/q           — 6-DoF joint positions (Tensor[f32,6])
  • robot/state/dq          — 6-DoF joint velocities (Tensor[f32,6])
  • commands/q_target       — commanded 6-DoF joint targets
  • world/ee                — EE pose (Transform3D)
  • gripper/state/width     — Robotiq opening width [m]
  • cam/rgb (optional)      — RGB frames at cfg.cam_fps

Output:
  • <outdir>/session_<YYYYMMDD_HHMMSS>.rrd

Usage examples (from repo root):
  # Use YAML defaults
  python src/run_capture.py --config src/configs/default.yaml

  # Override some params without editing YAML
  python src/run_capture.py --config src/configs/default.yaml --duration 90 --no-add-camera

  # Swap robot XML or timestep
  python src/run_capture.py --robot ./assets/xml/universal_robots_ur10e/ur10e_robotiq.xml --dt 0.005

Notes:
  • The camera is optional; disable via YAML (add_camera: false) or `--no-add-camera`.
  • Replay with:  rerun viewer <outdir>/session_<timestamp>.rrd
"""

from __future__ import annotations
import argparse

from configs.configs import load_config
from utils.rerun_utils import add_to_logger
from utils.utils import (prepare_env, 
                         manipulate_robot, 
                         get_path, 
                         cam_follow_arm_and_log)

# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Genesis UR10e + Robotiq → Rerun capture")
    p.add_argument("--config", default="./src/configs/default.yaml", help="YAML with params")
    # optional overrides (null = keep YAML):
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--add-camera", dest="add_camera", action="store_true")
    p.set_defaults(add_camera=None)  # so None means “no override”
    p.add_argument("--no-add-camera", dest="add_camera", action="store_false")
    p.add_argument("--outdir", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--robot", dest="robot_xml", default=None)
    p.add_argument("--dt", type=float, default=None)
    return p.parse_args()

# --------------------------- Main -------------------------------

def main():
    args = parse_args()
    cli_overrides = {
        "duration": args.duration,
        "add_camera": args.add_camera,
        "outdir": args.outdir,
        "session": args.session,
        "robot_xml": args.robot_xml,
        "dt": args.dt,
    }
    cfg = load_config(args.config, cli_overrides)

    scene, robot, ee, cube, motors_dof_idx, logger, cam = prepare_env(cfg)
    path, events = get_path(cfg, robot, ee, cube, motors_dof_idx)
    
    dt = float(scene.sim_options.dt)
    t = 0.0
    cam_every = max(1, int(round(1.0 / (dt * max(1e-9, cfg.cam_fps))))) if (cfg.add_camera and cam is not None) else 0

    total_steps = len(path)
    if args.duration is not None:
        total_steps = min(total_steps, int(round(args.duration / dt)))
    
    for i in range(total_steps):
        q_cmd = manipulate_robot(robot, path, i, motors_dof_idx, events)

        scene.step()

        # ★ log at current t (post-step), then advance time
        add_to_logger(logger, robot, ee, motors_dof_idx, t, q_cmd)
        t += dt  # 

        # Camera follow & capture
        cam_follow_arm_and_log(cam_every, cam, logger, i)

    out_path = logger.save()
    print(f"Saved Rerun recording: {out_path}")

if __name__ == "__main__":
    main()
