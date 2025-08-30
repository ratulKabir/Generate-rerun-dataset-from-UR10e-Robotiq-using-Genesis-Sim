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
import argparse
import numpy as np

from configs.configs import load_config
from utils.rerun_utils import add_to_logger
from utils.utils import prepare_env, manipulate_robot, get_path

# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Genesis UR10e + Robotiq → Rerun capture")
    p.add_argument("--config", default="./src/configs/default.yaml", help="YAML with params")
    # optional overrides (null = keep YAML):
    p.add_argument("--duration", type=float, default=None)
    p.add_argument("--add-camera", dest="add_camera", action="store_true")
    p.add_argument("--no-add-camera", dest="add_camera", action="store_false")
    p.set_defaults(add_camera=None)  # so None means “no override”
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

    out_path = logger.save()
    print(f"Saved Rerun recording: {out_path}")


if __name__ == "__main__":
    main()
