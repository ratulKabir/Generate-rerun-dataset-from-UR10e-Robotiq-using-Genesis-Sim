#!/usr/bin/env python3
import numpy as np
import genesis as gs

ROBOT_PATH = "./assets/xml/universal_robots_ur10e/ur10e_robotiq.xml"

########################## init ##########################
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.MJCF(file=ROBOT_PATH))

########################## build ##########################
scene.build()

# Arm joints
UR10E_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
motors_dof_idx = [robot.get_joint(name).dofs_idx_local[0] for name in UR10E_JOINTS]
home_qpos = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0], dtype=np.float32)
robot.set_dofs_position(home_qpos, motors_dof_idx)

# Gripper driver joints (Robotiq 2F-85)
def get_driver_dof_indices(robot):
    idx = []
    for j in robot.joints:
        if j.name in ["left_driver_joint", "right_driver_joint"]:
            idx.append(int(j.dofs_idx_local[0]))
    return idx

gripper_idx = get_driver_dof_indices(robot)

def set_gripper(open_frac: float):
    """Set gripper opening: 0.0 = closed, 1.0 = fully open."""
    q = [open_frac] * len(gripper_idx)
    robot.control_dofs_position(np.array(q, dtype=np.float32), dofs_idx_local=gripper_idx)

########################## run loop ##########################
steps_per_phase = 300
for i in range(3000):
    scene.step()
    robot.set_dofs_position(home_qpos, motors_dof_idx)

    # alternate gripper open/close every steps_per_phase
    if (i // steps_per_phase) % 2 == 0:
        set_gripper(1.0)  # open
    else:
        set_gripper(0.0)  # close
