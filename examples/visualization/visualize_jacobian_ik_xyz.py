import numpy as np
import genesis as gs
import time

from typing import List
from genesis.engine.entities.rigid_entity import RigidEntity
from koch11.core.kinematics.kinematics import forward_kinematics, forward_kinematics_all_links, inverse_kienmatics_xyz
from koch11.dynamixel.koch11 import dh_params
from koch11.core.kinematics.math_utils import rotation_around_axis, transform_to_quat, quat_to_rotation_matrix, quat_around_axis

def prepare_simulation(show_viewer: bool=True):
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
      show_viewer = show_viewer,
  )
  robot: RigidEntity = scene.add_entity(
      gs.morphs.URDF(
          file  = './urdf/koch11.urdf',
          pos   = (0.0, 0.0, 0.0),
          euler = (0, 0, 0),
          fixed=True
      ),
  )
  scene.build()
  
  return scene, robot

scene, robot = prepare_simulation(show_viewer=True)
direction = 1
dofs_index = [0, 1, 2, 3, 4]
delta = 0.01
q = np.array([0.0, np.deg2rad(10.0), np.deg2rad(45.0), np.deg2rad(90.0), 0.0])
pos = forward_kinematics(dh_params, q)[0:3 , 3]
dpos = np.array([0.0, 0.0, 0.001])
epsilon = 1e-6

print(pos)
while 1:
  if ((pos + dpos)[2] > 0.18 and pos[2] < 0.18) or ((pos + dpos)[2] < 0.0 and pos[2] > 0.0):
    dpos *= -1
    
  pos += dpos
  
  start = time.time()
  q = inverse_kienmatics_xyz(dh_params, pos, q, convergence_tolerance=0.0001, update_step=1.0)
  end = time.time()
  print(f"{(end - start) * 1000} {robot.get_links_pos()[4]}")
  
  robot.set_dofs_position(q, dofs_index)
  scene.step()