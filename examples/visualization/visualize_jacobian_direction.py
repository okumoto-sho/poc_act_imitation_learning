import numpy as np
import genesis as gs

from typing import List
from genesis.engine.entities.rigid_entity import RigidEntity
from koch11.core.kinematics.kinematics import forward_kinematics_all_links, calculate_basic_jacobian_xyz_omega
from koch11.dynamixel.koch11 import dh_params
from koch11.core.kinematics.math_utils import rotation_around_axis, transform_to_quat, quat_to_rotation_matrix, quat_around_axis

def main():
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

  direction = 1
  dofs_index = [0, 1, 2, 3, 4]
  delta = 0.01
  q = np.array([0.0, np.deg2rad(45.0), np.deg2rad(45.0), 0.0, 0.0])
  dq = np.array([delta, 0.0, 0.0, 0.0, 0.0])
  dpos = np.array([0.01, 0.0, 0.0])
  while 1:      
    q += dq
    robot.set_dofs_position(q, dofs_index)
    
    scene.clear_debug_objects()
    fk_all_links = forward_kinematics_all_links(dh_params, q)
    jacobian = calculate_basic_jacobian_xyz_omega(dh_params, q)
    dpos = jacobian @ dq
    print(f"{jacobian} {dpos}")
    scene.draw_debug_arrow(fk_all_links[-1][0:3, 3], 100.0 * dpos)
    
    if q[0] > np.pi / 2 or q[0] < -np.pi / 2:
      dq *= -1
          
    scene.step()