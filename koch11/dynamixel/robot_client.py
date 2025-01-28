import numpy as np

from typing import List
from koch11.core.robot_client import RobotClient, OperatingMode
from koch11.core.kinematics.kinematics import DhParam
from koch11.dynamixel.dynamixel_client import DynamixelXLSeriesClient, ControlTable


class DynamixelRobotClient(RobotClient):
    def __init__(
        self,
        motor_ids: List[int],
        dh_params: List[DhParam],
        q_range: dict,
        dq_range: dict,
        q_offsets=np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi / 2]),
        q_rot_direction=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        port_name="/dev/ttyACM0",
        baud_rate=2000000,
        retry_num=30,
    ):
        super().__init__(dh_params, q_range, dq_range, 0.002)
        self.motor_ids = motor_ids
        self.client = DynamixelXLSeriesClient(port_name, baud_rate)
        self.retry_num = retry_num
        self.q_offsets = q_offsets
        self.q_rot_direction = q_rot_direction

    def pwm_to_q_radians(self, data: np.ndarray | List[int]):
        q = np.array(data) * 0.087891 * (2 * np.pi / 360.0) + self.q_offsets
        return q * self.q_rot_direction

    def q_radians_to_pwm(self, data: np.ndarray):
        q_radians = (
            self.q_rot_direction * (data - self.q_offsets) * (360.0 / (2 * np.pi))
        )
        return list((q_radians * (1 / 0.087891)).astype(np.int32))

    def pwm_to_dq_radians(self, data: np.ndarray | List[int]):
        rev_per_min = np.array(data) * 0.229
        return self.q_rot_direction * 2 * np.pi * rev_per_min / 60.0

    def dq_radians_to_pwm(self, data: np.ndarray):
        rev_per_min = self.q_rot_direction * 60.0 * data / (2 * np.pi)
        rev_per_min_pwm = (rev_per_min / 0.229).astype(np.int32)
        return rev_per_min_pwm

    def _is_control_enabled_impl(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.TorqueEnable, self.retry_num
        )
        return all([x == 1 for x in ret])

    def _make_control_enable_impl(self):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.TorqueEnable,
            [1] * len(self.motor_ids),
            self.retry_num,
        )

    def _make_control_disable_impl(self):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.TorqueEnable,
            [0] * len(self.motor_ids),
            self.retry_num,
        )

    def _servo_q_impl(self, q: np.ndarray):
        q = self.q_radians_to_pwm(q)
        self.client.sync_write(
            self.motor_ids, ControlTable.GoalPosition, q, self.retry_num
        )

    def _servo_dq_impl(self, dq: np.ndarray):
        dq = self.dq_radians_to_pwm(dq)
        self.client.sync_write(
            self.motor_ids, ControlTable.GoalVelocity, dq, self.retry_num
        )

    def _servo_voltage_impl(self, v: np.ndarray):
        pass

    def _get_present_q_impl(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.PresentPosition, self.retry_num
        )
        return self.pwm_to_q_radians(ret)

    def _get_present_dq_impl(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.PresentVelocity, self.retry_num
        )
        return self.pwm_to_dq_radians(ret)

    def _get_present_voltage_impl(self):
        pass

    def _set_q_p_gains_impl(self, gains: np.ndarray):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.PositionPGain,
            gains.astype(np.int32),
            self.retry_num,
        )

    def _set_q_i_gains_impl(self, gains: np.ndarray):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.PositionIGain,
            gains.astype(np.int32),
            self.retry_num,
        )

    def _set_q_d_gains_impl(self, gains: np.ndarray):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.PositionDGain,
            gains.astype(np.int32),
            self.retry_num,
        )

    def _get_q_p_gains_impl(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionPGain, self.retry_num
        )

    def _get_q_i_gains_impl(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionIGain, self.retry_num
        )

    def _get_q_d_gains_impl(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionDGain, self.retry_num
        )

    def _set_dq_p_gains_impl(self, gains: np.ndarray):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.VelocityPGain,
            gains.astype(np.int32),
            self.retry_num,
        )

    def _set_dq_i_gains_impl(self, gains: np.ndarray):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.VelocityIGain,
            gains.astype(np.int32),
            self.retry_num,
        )

    def _set_operation_mode_impl(self, operating_mode: OperatingMode):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.OperatingMode,
            [operating_mode] * len(self.motor_ids),
            self.retry_num,
        )

    def _get_dq_p_gains_impl(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.VelocityPGain, self.retry_num
        )

    def _get_dq_i_gains_impl(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.VelocityIGain, self.retry_num
        )

    def _get_operation_mode_impl(self) -> OperatingMode:
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.OperatingMode, self.retry_num
        )
        return ret[0]

    def _get_present_tempratures_impl(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.PresentTemperature, self.retry_num
        )
        return ret
