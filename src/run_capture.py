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
    # ---- 1) Config: CLI > YAML defaults -------------------------------------
    args = parse_args()
    cli_overrides = {
        "duration": args.duration,       # if None, keep YAML 'duration'
        "add_camera": args.add_camera,   # None means no override
        "outdir": args.outdir,
        "session": args.session,
        "robot_xml": args.robot_xml,
        "dt": args.dt,
    }
    cfg = load_config(args.config, cli_overrides)

    # ---- 2) Build environment & instrumentation -----------------------------
    # prepare_env() creates the Genesis scene, loads the robot, adds a cube,
    # resolves motor indices, creates the Rerun logger, and (optionally) mounts a wrist camera.
    scene, robot, ee, cube, motors_dof_idx, logger, cam = prepare_env(cfg)

    # Build the full joint-space path (IK over Cartesian waypoints) and compute
    # event indices (e.g., when to close/open the gripper).
    path, events = get_path(cfg, robot, ee, cube, motors_dof_idx)

    # Simulation clock (single source of truth). dt is fixed-step physics time.
    dt = float(scene.sim_options.dt)
    t = 0.0

    # Camera capture cadence in “sim steps per frame”.
    # Example: dt=0.01 (100 Hz), cam_fps=30 → cam_every ≈ 100/30 ≈ 3–4 steps per frame.
    cam_every = (
        max(1, int(round(1.0 / (dt * max(1e-9, cfg.cam_fps)))))
        if (cfg.add_camera and cam is not None) else 0
    )

    # If a duration is provided (via YAML or CLI), truncate the path to match it.
    total_steps = len(path)
    if cfg.duration is not None:
        total_steps = min(total_steps, int(round(cfg.duration / dt)))

    # ---- 3) Main simulation loop --------------------------------------------
    # Each iteration:
    #   - send joint targets for this step (and trigger gripper events inside manipulate_robot)
    #   - step physics once
    #   - log state & commands at the current simulation time
    #   - advance sim time
    #   - (optionally) follow wrist with camera and log RGB at target cadence
    for i in range(total_steps):
        # Sends control for the arm joints at step i and returns the actual command used
        # (your helper handles edge cases and gripper events using 'events').
        q_cmd = manipulate_robot(robot, path, i, motors_dof_idx, events)

        # Advance the physics by one fixed dt.
        scene.step()

        # Log all streams at the *current* simulation time t (post-step),
        # then increment t for the next iteration.
        add_to_logger(logger, robot, ee, motors_dof_idx, t, q_cmd)
        t += dt

        # If a camera is present, keep it attached to the wrist and log RGB
        # every 'cam_every' sim steps for ~cfg.cam_fps.
        cam_follow_arm_and_log(cam_every, cam, logger, i)

    # ---- 4) Persist Rerun recording -----------------------------------------
    out_path = logger.save()
    print(f"Saved Rerun recording: {out_path}")

if __name__ == "__main__":
    main()
