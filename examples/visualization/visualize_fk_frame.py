import numpy as np
import genesis as gs

from genesis.engine.entities.rigid_entity import RigidEntity
from koch11.core.kinematics.kinematics import forward_kinematics
from koch11.dynamixel.koch11 import dh_params
from koch11.core.kinematics.math_utils import quat_rpy_radians, rotation_matrix_to_rpy_radians, transform_to_pos_quat


gs.init(backend=gs.gpu)
scene = gs.Scene(
  viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

robot: RigidEntity = scene.add_entity(
    gs.morphs.URDF(
        file  = './assets/urdf/koch11.urdf',
        pos   = (0.0, 0.0, 0.0),
        euler = (0, 0, 0),
        fixed=True
    ),
)
scene.build()

dofs_index = [0, 1, 2, 3, 4]
delta = 0.01
q = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
while 1:
  T = forward_kinematics(dh_params, q)
    
  link_pos = robot.get_links_pos().cpu().numpy()
  link_quat = robot.get_links_quat().cpu().numpy()
  pquat = transform_to_pos_quat(T)
  if q[0] < 2 * np.pi / 3:
    q[0] += delta
  elif q[1] < np.pi / 2:
    q[1] += delta
  elif q[2] > -np.pi / 2:
    q[2] -= delta
  elif q[3] > -np.pi / 2:
    q[3] -= delta
  elif q[4] < np.pi / 2:
    q[4] += delta
  
  rpy = rotation_matrix_to_rpy_radians(T, 0.0)
  pos = T[0:3, 3]
  
  robot.set_dofs_position(q, dofs_index)
  
  scene.clear_debug_objects()
  scene.draw_debug_frame(T, 0.1)
    
  scene.step()