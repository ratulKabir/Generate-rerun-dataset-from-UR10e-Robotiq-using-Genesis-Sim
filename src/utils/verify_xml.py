import numpy as np
import genesis as gs


ROBOT_PATH = "./assets/xml/universal_robots_ur10e/ur10e_robotiq_debug.xml"

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

# motors_dof_idx = [franka.get_joint(name).dofs_idx_local[0] for name in joints_name]

# home_qpos = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
# franka.set_dofs_position(home_qpos, motors_dof_idx)

# execute the planned path
for i in range(5000):
    scene.step()
    # franka.set_dofs_position(home_qpos, motors_dof_idx)




