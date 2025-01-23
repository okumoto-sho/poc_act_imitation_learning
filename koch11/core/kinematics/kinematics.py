import numpy as np
import dataclasses

from typing import List
from koch11.core.kinematics.math_utils import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z, translation_matrix
from numpy.typing import ArrayLike

@dataclasses.dataclass
class DhParam:
  a: float
  alpha: float
  d: float
  theta: float
  
def forward_kinematics(dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)):
  fk_all_links = forward_kinematics_all_links(dh_params, q_radians, ee_transform)
  return fk_all_links[-1]

def forward_kinematics_all_links(dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)):
  if len(dh_params) != len(q_radians):
    raise ValueError("Given different length dh_params and q_radians")

  T = np.identity(4)
  fk_all_links =[]
  for dh_param, q_radians in zip(dh_params, q_radians):
    Tx = translation_matrix(np.array([dh_param.a, 0, 0]))
    Rx = rotation_matrix_x(dh_param.alpha)
    Tz = translation_matrix(np.array([0.0, 0.0, dh_param.d]))
    Rz = rotation_matrix_z(dh_param.theta + q_radians)
    T = T @ Tx @ Rx @ Tz @ Rz
    fk_all_links.append(T)
  
  T = T @ ee_transform
  fk_all_links.append(T)
  return fk_all_links

def calculate_z_directions_all_links(dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)):
  fk_all_links = forward_kinematics_all_links(dh_params, q_radians, ee_transform)
  return [T[0:3, 2] for T in fk_all_links]

def calculate_basic_jacobian_xyz(dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)):
  transform_all_links = forward_kinematics_all_links(dh_params, q_radians, ee_transform)
  ee_pos = transform_all_links[-1][0:3, 3]
  jacobian_xyz = np.zeros((3, len(q_radians)))
  for i, transform in enumerate(transform_all_links[:-1]):
    link_pos = transform[0:3, 3]
    rel_ee_link = ee_pos - link_pos
    z = transform[0:3, 2]
    jacobian_xyz[0:3 ,i] = np.cross(z, rel_ee_link)
  return jacobian_xyz

def inverse_kienmatics_xyz(dh_params: List[DhParam], goal_pos: np.ndarray, init_q_radians: np.ndarray, ee_transform=np.identity(4), update_step=1.0, max_iter=1000, convergence_tolerance=0.00001, epsilon=1e-5) -> List[np.ndarray]:
  current_q_radians = init_q_radians
  for _ in range(max_iter):
    J = calculate_basic_jacobian_xyz(dh_params, current_q_radians, ee_transform)
    cur_pos = forward_kinematics(dh_params, current_q_radians, ee_transform)[0:3, 3]
    diff_pos = goal_pos - cur_pos

    if np.linalg.norm(diff_pos) < convergence_tolerance:
      return current_q_radians
        
    dq = J.T @ np.linalg.inv(J @ J.T + epsilon * np.identity(3)) @ diff_pos
    current_q_radians = current_q_radians + update_step * dq
  
  return current_q_radians

