import numpy as np

from koch11.dynamixel.dynamixel_client import DynamixelXLSeriesClient, ControlTable
from typing import List, Any
from koch11.core.robot_client import OperatingMode


class DynamixelRobotClient:
    def __init__(
        self,
        motor_ids: List[int],
        port_name="/dev/ttyACM1",
        baud_rate=2000000,
        retry_num=30,
    ):
        self.motor_ids = motor_ids
        self.client = DynamixelXLSeriesClient(port_name, baud_rate)
        self.retry_num = retry_num

    def _check_data_length(self, data: np.ndarray | List[Any]):
        if len(data) != len(self.motor_ids):
            raise ValueError(
                f"Different length data is given. Expected length is {len(self.motor_ids)}"
            )

    def _to_degrees(self, data: np.ndarray | List[int]):
        return np.array(data) * (360.0 / 4096.0)

    def _to_rev_per_min(self, data: np.ndarray | List[int]):
        return np.array(data) * (234.496 / 1024)

    def _to_position_pwm_values(self, data: np.ndarray):
        return list((data * (4096.0 / 360.0)).astype(np.int32))

    def _to_velocity_pwm_values(self, data: np.ndarray):
        return list((data * (1024 / 234.496)).astype(np.int32))

    def get_present_load(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.PresentLoad, self.retry_num
        )

    def get_present_pwms(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.PresentPwm, self.retry_num
        )

    def get_present_velocities(self) -> np.ndarray:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.ProfileVelocity, self.retry_num
        )
        return self._to_rev_per_min(ret)

    def get_present_positions(self) -> np.ndarray:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.PresentPosition, self.retry_num
        )
        return self._to_degrees(ret)

    def get_operating_modes(self) -> List[OperatingMode]:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.OperatingMode, self.retry_num
        )
        return ret

    def set_operating_modes(self, operating_modes: List[OperatingMode]):
        self._check_data_length(operating_modes)
        self.client.sync_write(
            self.motor_ids, ControlTable.OperatingMode, operating_modes, self.retry_num
        )

    def set_operating_mode(self, operating_mode: OperatingMode):
        self.set_operating_modes([operating_mode] * len(self.motor_ids))

    def set_goal_positions(self, q: np.ndarray):
        self._check_data_length(q)
        q = self._to_position_pwm_values(q)
        self.client.sync_write(
            self.motor_ids, ControlTable.GoalPosition, q, self.retry_num
        )

    def set_goal_velocioties(self, qd: np.ndarray):
        self._check_data_length(qd)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.GoalVelocity,
            self._to_velocity_pwm_values(qd),
            self.retry_num,
        )

    def set_goal_pwms(self, pwms: np.ndarray):
        self._check_data_length(pwms)
        self.client.sync_write(
            self.motor_ids, ControlTable.GoalPwm, pwms, self.retry_num
        )

    def get_goal_positions(self) -> np.ndarray:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.GoalPosition, self.retry_num
        )
        return self._to_degrees(ret)

    def get_goal_velocities(self) -> np.ndarray:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.GoalVelocity, self.retry_num
        )
        return self._to_rev_per_min(ret)

    def get_goal_pwms(self) -> np.ndarray:
        return np.array(
            self.client.sync_read(self.motor_ids, ControlTable.GoalPwm, self.retry_num)
        )

    def set_position_p_gains(self, p_gains: np.ndarray):
        self._check_data_length(p_gains)
        self.client.sync_write(
            self.motor_ids, ControlTable.PositionPGain, p_gains, self.retry_num
        )

    def set_position_i_gains(self, i_gains: np.ndarray):
        self._check_data_length(i_gains)
        self.client.sync_write(
            self.motor_ids, ControlTable.PositionIGain, i_gains, self.retry_num
        )

    def set_position_d_gains(self, d_gains: np.ndarray):
        self._check_data_length(d_gains)
        self.client.sync_write(
            self.motor_ids, ControlTable.PositionDGain, d_gains, self.retry_num
        )

    def get_position_p_gains(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionPGain, self.retry_num
        )

    def get_position_i_gains(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionIGain, self.retry_num
        )

    def get_position_d_gains(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionDGain, self.retry_num
        )

    def set_velocity_p_gains(self, p_gains: np.ndarray):
        self._check_data_length(p_gains)
        self.client.sync_write(
            self.motor_ids, ControlTable.VelocityPGain, p_gains, self.retry_num
        )

    def set_velocity_i_gains(self, i_gains: np.ndarray):
        self._check_data_length(i_gains)
        self.client.sync_write(
            self.motor_ids, ControlTable.VelocityPGain, i_gains, self.retry_num
        )

    def get_velocity_p_gains(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.VelocityPGain, self.retry_num
        )

    def get_velocity_i_gains(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.VelocityIGain, self.retry_num
        )

    def set_pwm_limits(self, pwm_limits: np.ndarray):
        self._check_data_length(pwm_limits)
        self.client.sync_write(
            self.motor_ids, ControlTable.PwmLimit, pwm_limits, self.retry_num
        )

    def get_pwm_limits(self) -> np.ndarray:
        return self.client.sync_read(
            self.motor_ids, ControlTable.PwmLimit, self.retry_num
        )

    def set_velocity_limits(self, velocity_limits: np.ndarray):
        self._check_data_length(velocity_limits)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.VelocityLimit,
            self._to_velocity_pwm_values(velocity_limits),
            self.retry_num,
        )

    def get_velocity_limits(self, velocity_limits: np.ndarray) -> np.ndarray:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.VelocityLimit, self.retry_num
        )
        return self._to_rev_per_min(ret)

    def get_min_position_limits(self) -> np.ndarray:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.MinPositionLimit, self.retry_num
        )
        return self._to_degrees(ret)

    def set_min_position_limits(self, min_position_limits: np.ndarray):
        self._check_data_length(min_position_limits)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.MinPositionLimit,
            self._to_position_pwm_values(min_position_limits),
            self.retry_num,
        )

    def get_max_position_limits(self) -> np.ndarray:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.MaxPositionLimit, self.retry_num
        )
        return self._to_degrees(ret)

    def set_max_position_limits(self, max_position_limits: np.ndarray):
        self._check_data_length(max_position_limits)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.MaxPositionLimit,
            self._to_position_pwm_values(max_position_limits),
            self.retry_num,
        )

    def set_torque_enable(self):
        data = [1] * len(self.motor_ids)
        self.client.sync_write(
            self.motor_ids, ControlTable.TorqueEnable, data, self.retry_num
        )

    def set_torque_disable(self):
        data = [0] * len(self.motor_ids)
        self.client.sync_write(
            self.motor_ids, ControlTable.TorqueEnable, data, self.retry_num
        )

    def close(self):
        self.set_torque_disable()
        self.client.close()
