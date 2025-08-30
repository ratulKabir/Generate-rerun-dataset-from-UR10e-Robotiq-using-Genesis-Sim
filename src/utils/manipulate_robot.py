import numpy as np
import genesis as gs


# ROBOT_PATH = "xml/universal_robots_ur10e/ur10e.xml"
ROBOT_PATH = "xml/universal_robots_ur10e/ur10e_robotiq.xml"

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
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(
        file=ROBOT_PATH,
    ),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size = (0.14, 0.14, 0.14),
        pos  = (-0.65, -0.5, 0.02),
    )
)
########################## build ##########################
scene.build()


joints_name = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
)

motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

# ############ Optional: set control gains ############
# kp=np.array([4500, 4500, 3500, 3500, 2000, 2000])
# kv=np.array([450, 450, 350, 350, 200, 200])
# force_min=np.array([-87, -87, -87, -87, -12, -12])
# force_max=np.array([87, 87, 87, 87, 12, 12])

# # set positional gains
# franka.set_dofs_kp(
#     kp=kp,
#     dofs_idx_local=motors_dof_idx,
# )
# # set velocity gains
# franka.set_dofs_kv(
#     kv=kv,
#     dofs_idx_local=motors_dof_idx,
# )
# # set force range for safety
# franka.set_dofs_force_range(
#     lower=force_min,
#     upper=force_max,
#     dofs_idx_local=motors_dof_idx,
# )
# Hard reset
# franka.set_dofs_position(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), motors_dof_idx)
home_qpos = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
franka.set_dofs_position(home_qpos, motors_dof_idx)

# get the end-effector link
end_effector = franka.get_link('wrist_3_link')

# move to pre-grasp pose
qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = np.array([-0.65, -0.5, 0.5]),
    # quat = np.array([0, 1, 0, 0]),
)
# gripper open pos
path = franka.plan_path(
    qpos_goal = qpos,
    # num_waypoints = 500, # 2s duration
)
# execute the planned path
for i in range(max(len(path), 1000)):
    try:
        franka.control_dofs_position(path[i])
        last_pt = path[i]
    except:
        franka.control_dofs_position(last_pt)
    # franka.set_dofs_position(home_qpos, motors_dof_idx)
    scene.step()




