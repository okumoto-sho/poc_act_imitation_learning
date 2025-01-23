import numpy as np

from typing import List, Any
from koch11.core.robot_client import RobotClient, OperatingMode, QRange
from koch11.core.kinematics.kinematics import DhParam, forward_kinematics, forward_kinematics_all_links, calculate_basic_jacobian_xyz
from koch11.core.kinematics.math_utils import transform_to_xyz_rpy
from koch11.dynamixel.dynamixel_client import DynamixelXLSeriesClient, ControlTable


class DynamixelRobotClient(RobotClient):
  def __init__(self, motor_ids: List[int], dh_params:List[DhParam], q_range: np.ndarray, dq_range: np.ndarray, q_offsets=np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi]), port_name="/dev/ttyACM0", baud_rate=2000000, retry_num=30):
    self.motor_ids = motor_ids
    self.dh_params = dh_params
    self.client = DynamixelXLSeriesClient(port_name, baud_rate)
    self.retry_num = retry_num
    self.control_cycle = 0.002
    self.q_offsets = q_offsets
    self.q_range = q_range
    self.dq_range = dq_range
  
  def _check_data_length(self, data: np.ndarray | List[Any]):
    if len(data) != len(self.motor_ids):
      raise ValueError(f"Different length data is given. Expected length is {len(self.motor_ids)}")
    
  def _pwm_to_q_radians(self, data: np.ndarray | List[int]):
    return np.array(data) * 0.087891 * (2 * np.pi / 360.0) + self.q_offsets
  
  def _q_radians_to_pwm(self, data: np.ndarray):
    q_radians = (data - self.q_offsets) * (360.0 / (2 * np.pi))
    return list((q_radians * (1 / 0.087891)).astype(np.int32))
                
  def _pwm_to_dq_radians(self, data: np.ndarray | List[int]):
    rev_per_min = np.array(data) * 0.229
    return (2 * np.pi * rev_per_min / 60.0)
  
  def _dq_radians_to_pwm(self, data: np.ndarray):
    rev_per_min = (60.0 * data / (2 * np.pi))
    rev_per_min_pwm = (rev_per_min / 0.229).astype(np.int32)
    return rev_per_min_pwm
                
  def make_control_enable(self):
    self.client.sync_write(self.motor_ids, ControlTable.TorqueEnable, [1] * len(self.motor_ids), self.retry_num)
    
  def make_control_disable(self):
    self.client.sync_write(self.motor_ids, ControlTable.TorqueEnable, [0] * len(self.motor_ids), self.retry_num)
      
  def is_control_enabled(self):
    ret = self.client.sync_read(self.motor_ids, ControlTable.TorqueEnable, self.retry_num)
    return all([x == 1 for x in ret])
      
  def forward_kinematics(self, q: np.ndarray, ee_transform: np.ndarray = np.identity(4)):
    self._check_data_length(q)
    return forward_kinematics(self.dh_params, q, ee_transform)
  
  def forward_kinematics_all_links(self, q: np.ndarray, ee_transform: np.ndarray = np.identity(4), convergence_tol=1e-4):
    self._check_data_length(q)
    return forward_kinematics_all_links(self.dh_params, q, ee_transform, convergence_tol)

  def get_jacobian(self, q):
    return calculate_basic_jacobian_xyz(self.dh_params, q)

  def servo_q(self, q: np.ndarray):
    self._check_data_length(q)
    q = self._q_radians_to_pwm(q)
    self.client.sync_write(self.motor_ids, ControlTable.GoalPosition, q, self.retry_num)

  def servo_dq(self, dq: np.ndarray):
    self._check_data_length(dq)
    dq = self._dq_radians_to_pwm(dq)
    self.client.sync_write(self.motor_ids, ControlTable.GoalVelocity, dq, self.retry_num)
  
  def set_q_p_gains(self, p_gains: np.ndarray):
    self._check_data_length(p_gains)
    self.client.sync_write(self.motor_ids, ControlTable.PositionPGain, p_gains.astype(np.int32), self.retry_num)
    
  def set_q_i_gains(self, i_gains: np.ndarray):
    self._check_data_length(i_gains)
    self.client.sync_write(self.motor_ids, ControlTable.PositionIGain, i_gains.astype(np.int32), self.retry_num)
    
  def set_q_d_gains(self, d_gains: np.ndarray):
    self._check_data_length(d_gains)
    self.client.sync_write(self.motor_ids, ControlTable.PositionDGain, d_gains.astype(np.int32), self.retry_num)
  
  def set_dq_p_gains(self, p_gains: np.ndarray):
    self._check_data_length(p_gains)
    self.client.sync_write(self.motor_ids, ControlTable.VelocityPGain, p_gains.astype(np.int32), self.retry_num)
    
  def set_dq_i_gains(self, i_gains: np.ndarray):
    self._check_data_length(i_gains)
    self.client.sync_write(self.motor_ids, ControlTable.VelocityIGain, i_gains.astype(np.int32), self.retry_num)

  def get_q_p_gains(self):
    return self.client.sync_read(self.motor_ids, ControlTable.PositionPGain, self.retry_num)
  
  def get_q_i_gains(self):
    return self.client.sync_read(self.motor_ids, ControlTable.PositionIGain, self.retry_num)
  
  def get_q_d_gains(self):
    return self.client.sync_read(self.motor_ids, ControlTable.PositionDGain, self.retry_num)

  def get_dq_p_gains(self):
    return self.client.sync_read(self.motor_ids, ControlTable.VelocityPGain, self.retry_num)
  
  def get_dq_i_gains(self):
    return self.client.sync_read(self.motor_ids, ControlTable.VelocityIGain, self.retry_num)

  def get_present_q(self):
    ret = self.client.sync_read(self.motor_ids, ControlTable.PresentPosition, self.retry_num)
    return self._pwm_to_q_radians(ret)
  
  def get_present_p(self):
    ret = self.get_present_q()
    fk = self.forward_kinematics(ret)
    return transform_to_xyz_rpy(fk)
  
  def get_present_dq(self):
    ret = self.client.sync_read(self.motor_ids, ControlTable.PresentVelocity, self.retry_num)
    return self._pwm_to_dq_radians(ret)
  
  def get_present_dp(self):
    ret = self.get_present_dq()
    J = self.get_jacobian(self.get_present_q())
    return J @ ret
  
  def get_q_range(self):
    return self.q_range
  
  def get_dq_range(self):
    return self.dq_range
  
  def set_operation_mode(self, operating_mode: OperatingMode):
    self.client.sync_write(self.motor_ids, ControlTable.OperatingMode, [operating_mode] * len(self.motor_ids), self.retry_num)
    
  def get_operation_mode(self):
    ret = self.client.sync_read(self.motor_ids, ControlTable.OperatingMode, self.retry_num)
    return ret[0]
  
  def get_present_tempratures(self):
    ret = self.client.sync_read(self.motor_ids, ControlTable.PresentTemperature, self.retry_num)
    return ret