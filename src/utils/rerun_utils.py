import os
import torch
import rerun as rr
import numpy as np

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
