import numpy as np
import time

from typing import List, Any
from koch11.core.robot_client import RobotClient, OperatingMode
from koch11.core.kinematics.kinematics import (
    DhParam,
    forward_kinematics,
    forward_kinematics_all_links,
    calculate_basic_jacobian_xyz_omega,
    inverse_kienmatics,
    plan_ik_q_trajectory,
)
from koch11.core.kinematics.math_utils import transform_to_xyz_rpy
from koch11.dynamixel.dynamixel_client import DynamixelXLSeriesClient, ControlTable


class DynamixelRobotClient(RobotClient):
    def __init__(
        self,
        motor_ids: List[int],
        dh_params: List[DhParam],
        q_range: dict,
        dq_range: dict,
        q_offsets=np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi]),
        port_name="/dev/ttyACM0",
        baud_rate=2000000,
        retry_num=30,
    ):
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
            raise ValueError(
                f"Different length data is given. Expected length is {len(self.motor_ids)}"
            )

    def _pwm_to_q_radians(self, data: np.ndarray | List[int]):
        return np.array(data) * 0.087891 * (2 * np.pi / 360.0) + self.q_offsets

    def _q_radians_to_pwm(self, data: np.ndarray):
        q_radians = (data - self.q_offsets) * (360.0 / (2 * np.pi))
        return list((q_radians * (1 / 0.087891)).astype(np.int32))

    def _pwm_to_dq_radians(self, data: np.ndarray | List[int]):
        rev_per_min = np.array(data) * 0.229
        return 2 * np.pi * rev_per_min / 60.0

    def _dq_radians_to_pwm(self, data: np.ndarray):
        rev_per_min = 60.0 * data / (2 * np.pi)
        rev_per_min_pwm = (rev_per_min / 0.229).astype(np.int32)
        return rev_per_min_pwm

    def _contained_in_range_q(self, q: np.ndarray):
        return np.all(q >= self.q_range["min"]) and np.all(q <= self.q_range["max"])

    def _contained_in_range_dq(self, dq: np.ndarray):
        return np.all(dq >= self.dq_range["min"]) and np.all(dq <= self.dq_range["max"])

    def _check_range_q(self, q: np.ndarray):
        if not self._contained_in_range_q(q):
            raise ValueError(f"q: {q} is out of range")

    def _check_range_dq(self, dq: np.ndarray):
        if not self._contained_in_range_dq(dq):
            raise ValueError("dq is out of range")

    def _try_make_q_contained_in_range_q(self, q: np.ndarray):
        def _try_contained(q_i: float, q_min: float, q_max: float):
            print(q_i)
            if q_min <= q_i <= q_max:
                return q_i
            elif q_i - 2.0 * np.pi > q_min:
                return q_i - 2.0 * np.pi
            elif q_i + 2.0 * np.pi < q_max:
                return q_i + 2.0 * np.pi
            return q_i

        q_mod = np.fmod(q, 2 * np.pi)
        return np.array(
            [
                _try_contained(q_i, q_min, q_max)
                for q_i, q_min, q_max in zip(
                    q_mod, self.q_range["min"], self.q_range["max"]
                )
            ]
        )

    def make_control_enable(self):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.TorqueEnable,
            [1] * len(self.motor_ids),
            self.retry_num,
        )

    def make_control_disable(self):
        self.client.sync_write(
            self.motor_ids,
            ControlTable.TorqueEnable,
            [0] * len(self.motor_ids),
            self.retry_num,
        )

    def move_q(
        self,
        q: np.ndarray,
        speed: float = 0.4,
        convergence_timeout_seconds=4.0,
        atol=0.01,
    ):
        self._check_data_length(q)
        self._check_range_q(q)
        self.move_q_path([q], speed, convergence_timeout_seconds, atol)

    def move_q_ik(
        self,
        xyz: np.ndarray | None,
        rpy: np.ndarray | None,
        speed: float = 0.4,
        ee_transform: np.ndarray = np.identity(4),
        convergence_timeout_seconds=4.0,
        xyz_convergence_tol=0.001,
        rpy_convergence_tol=0.001,
        atol=0.01,
    ):
        q = self.inverse_kinematics(
            xyz,
            rpy,
            self.get_present_q(),
            ee_transform,
            xyz_convergence_tol,
            rpy_convergence_tol,
        )
        if q is None:
            raise RuntimeError("Failed to obtain inverse kinematics solution")
        self.move_q(q, speed, convergence_timeout_seconds, atol)

    def move_p(
        self,
        xyz: np.ndarray | None,
        rpy: np.ndarray | None,
        ee_transform: np.ndarray = np.identity(4),
        xyz_max_speed=0.1,
        rpy_max_speed=1.55,
        convergence_timeout_seconds=4.0,
        xyz_convergence_tol=0.001,
        rpy_convergence_tol=0.001,
        atol=0.01,
    ):
        q_path = plan_ik_q_trajectory(
            self.dh_params,
            [xyz] if xyz is not None else None,
            [rpy] if rpy is not None else None,
            self.get_present_q(),
            self.control_cycle,
            ee_transform=ee_transform,
            xyz_max_speed=xyz_max_speed,
            rpy_max_speed=rpy_max_speed,
            xyz_convergence_tolerance=xyz_convergence_tol,
            rpy_convergence_tolerance=rpy_convergence_tol,
        )
        self.move_q_path(q_path, 30.0, convergence_timeout_seconds, atol)

    def move_q_path(
        self,
        q_path: List[np.ndarray],
        speed: float,
        convergence_timeout_seconds=4.0,
        atol=0.01,
    ):
        current_q = self.get_present_q()
        for q in q_path:
            diff_q = q - current_q
            times_taken = np.max(np.abs(diff_q)) / speed
            num_steps = int(times_taken / self.control_cycle) + 2
            step_q = diff_q / num_steps
            for t in range(num_steps):
                target_q = current_q + step_q * t
                self.servo_q(target_q)
            current_q = q

        start = time.time()
        while time.time() - start < convergence_timeout_seconds and not np.allclose(
            current_q, q_path[-1], atol=atol
        ):
            current_q = self.get_present_q()

    def is_control_enabled(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.TorqueEnable, self.retry_num
        )
        return all([x == 1 for x in ret])

    def inverse_kinematics(
        self,
        xyz: np.ndarray | None,
        rpy: np.ndarray | None,
        init_q: np.ndarray,
        ee_transform: np.ndarray = np.identity(4),
        xyz_convergence_tol=0.001,
        rpy_convergence_tol=0.001,
    ):
        q = inverse_kienmatics(
            self.dh_params,
            xyz,
            rpy,
            init_q,
            ee_transform,
            update_step=1,
            xyz_convergence_tolerance=xyz_convergence_tol,
            rpy_convergence_tolerance=rpy_convergence_tol,
        )
        if q is None:
            print("Inverse kinematics did not converge")
            return None

        q = np.fmod(q, 2 * np.pi)
        q = self._try_make_q_contained_in_range_q(q)
        if not self._contained_in_range_q(q):
            print(f"Inverse kinematics solution {q} is out of range")
            return None
        return q

    def forward_kinematics(
        self, q: np.ndarray, ee_transform: np.ndarray = np.identity(4)
    ):
        self._check_data_length(q)
        return forward_kinematics(self.dh_params, q, ee_transform)

    def forward_kinematics_all_links(
        self,
        q: np.ndarray,
        ee_transform: np.ndarray = np.identity(4),
        convergence_tol=1e-4,
    ):
        self._check_data_length(q)
        return forward_kinematics_all_links(
            self.dh_params, q, ee_transform, convergence_tol
        )

    def get_jacobian(self, q):
        return calculate_basic_jacobian_xyz_omega(self.dh_params, q)

    def servo_q(self, q: np.ndarray):
        start = time.time()
        self._check_data_length(q)
        self._check_range_q(q)
        q = self._q_radians_to_pwm(q)
        self.client.sync_write(
            self.motor_ids, ControlTable.GoalPosition, q, self.retry_num
        )
        end = time.time()
        time.sleep(max(0, self.control_cycle - (end - start)))

    def servo_dq(self, dq: np.ndarray):
        start = time.time()
        self._check_data_length(dq)
        self._check_range_dq(dq)
        dq = self._dq_radians_to_pwm(dq)
        self.client.sync_write(
            self.motor_ids, ControlTable.GoalVelocity, dq, self.retry_num
        )
        end = time.time()
        time.sleep(max(0, self.control_cycle - (end - start)))

    def set_q_p_gains(self, p_gains: np.ndarray):
        self._check_data_length(p_gains)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.PositionPGain,
            p_gains.astype(np.int32),
            self.retry_num,
        )

    def set_q_i_gains(self, i_gains: np.ndarray):
        self._check_data_length(i_gains)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.PositionIGain,
            i_gains.astype(np.int32),
            self.retry_num,
        )

    def set_q_d_gains(self, d_gains: np.ndarray):
        self._check_data_length(d_gains)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.PositionDGain,
            d_gains.astype(np.int32),
            self.retry_num,
        )

    def set_dq_p_gains(self, p_gains: np.ndarray):
        self._check_data_length(p_gains)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.VelocityPGain,
            p_gains.astype(np.int32),
            self.retry_num,
        )

    def set_dq_i_gains(self, i_gains: np.ndarray):
        self._check_data_length(i_gains)
        self.client.sync_write(
            self.motor_ids,
            ControlTable.VelocityIGain,
            i_gains.astype(np.int32),
            self.retry_num,
        )

    def get_q_p_gains(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionPGain, self.retry_num
        )

    def get_q_i_gains(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionIGain, self.retry_num
        )

    def get_q_d_gains(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.PositionDGain, self.retry_num
        )

    def get_dq_p_gains(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.VelocityPGain, self.retry_num
        )

    def get_dq_i_gains(self):
        return self.client.sync_read(
            self.motor_ids, ControlTable.VelocityIGain, self.retry_num
        )

    def get_present_q(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.PresentPosition, self.retry_num
        )
        return self._pwm_to_q_radians(ret)

    def get_present_p(self):
        ret = self.get_present_q()
        fk = self.forward_kinematics(ret)
        return transform_to_xyz_rpy(fk)

    def get_present_dq(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.PresentVelocity, self.retry_num
        )
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
        self.client.sync_write(
            self.motor_ids,
            ControlTable.OperatingMode,
            [operating_mode] * len(self.motor_ids),
            self.retry_num,
        )

    def get_operation_mode(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.OperatingMode, self.retry_num
        )
        return ret[0]

    def get_present_tempratures(self):
        ret = self.client.sync_read(
            self.motor_ids, ControlTable.PresentTemperature, self.retry_num
        )
        return ret
