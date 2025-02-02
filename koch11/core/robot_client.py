import numpy as np
import dataclasses
import time

from koch11.core.kinematics.kinematics import (
    DhParam,
    forward_kinematics,
    forward_kinematics_all_links,
    calculate_basic_jacobian_xyz_omega,
    inverse_kienmatics,
    plan_ik_q_trajectory,
)
from koch11.core.kinematics.math_utils import (
    rotation_matrix_rpy,
    rotation_matrix_to_axis_and_angle,
    transform_to_xyz_rpy,
)
from typing import List, Any, Dict
from enum import IntEnum
from abc import ABC, abstractmethod


class OperatingMode(IntEnum):
    VelocityControl = 1
    PositionControl = 3
    PwmControl = 16


@dataclasses.dataclass
class QRange:
    min: np.ndarray
    max: np.ndarray


class RobotClient(ABC):
    _not_implemented_message: str = "This function is not implemented"

    def __init__(
        self,
        dh_params: List[DhParam],
        link_q_indices: List[int],
        q_range: Dict[str, np.ndarray],
        dq_range: Dict[str, np.ndarray],
        control_cycle: float,
    ):
        if len(dh_params) != len(link_q_indices):
            raise ValueError("Length of dh_params, link_q_indices should be the same")

        if not (
            len(q_range["min"])
            == len(q_range["max"])
            == len(dq_range["min"])
            == len(dq_range["max"])
        ):
            raise ValueError("Length of q_range, dq_range should be the same")

        self.link_q_indices = link_q_indices
        self.link_q_range = {
            "min": q_range["min"][link_q_indices],
            "max": q_range["max"][link_q_indices],
        }
        self.link_dq_range = {
            "min": dq_range["min"][link_q_indices],
            "max": dq_range["max"][link_q_indices],
        }

        self.dh_params = dh_params
        self.q_range = q_range
        self.dq_range = dq_range
        self.control_cycle = control_cycle

    def _check_data_length(self, data: np.ndarray | List[Any]):
        if len(data) != len(self.q_range["min"]):
            raise ValueError(
                f"Different length data is given. Expected length is {len(self.motor_ids)}"
            )

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

    def _check_control_enabled(self):
        if not self.is_control_enabled():
            raise RuntimeError(
                "Control is not enabled. Enable control first by calling `make_control_enable`"
            )

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
        self._make_control_enable_impl()

    def is_control_enabled(self):
        return self._is_control_enabled_impl()

    def make_control_disable(self):
        self._make_control_disable_impl()

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
        rpy_max_speed=4.55,
        convergence_timeout_seconds=4.0,
        xyz_convergence_tol=0.001,
        rpy_convergence_tol=0.001,
        atol=0.01,
    ):
        q_path = plan_ik_q_trajectory(
            self.dh_params,
            self.link_q_indices,
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
        if q_path is None:
            raise RuntimeError("Failed to obtain inverse kinematics solution")

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

    def move_p_path(
        self,
        xyz: List[np.ndarray] | None,
        rpy: List[np.ndarray] | None,
        ee_transform: np.ndarray = np.identity(4),
        xyz_max_speed=0.1,
        rpy_max_speed=4.55,
        convergence_timeout_seconds=4.0,
        xyz_convergence_tol=0.001,
        rpy_convergence_tol=0.001,
        atol=0.01,
    ):
        q_path = plan_ik_q_trajectory(
            self.dh_params,
            self.link_q_indices,
            xyz,
            rpy,
            self.get_present_q(),
            self.control_cycle,
            ee_transform=ee_transform,
            xyz_max_speed=xyz_max_speed,
            rpy_max_speed=rpy_max_speed,
            xyz_convergence_tolerance=xyz_convergence_tol,
            rpy_convergence_tolerance=rpy_convergence_tol,
        )
        if q_path is None:
            raise RuntimeError("Failed to obtain inverse kinematics solution")

        self.move_q_path(q_path, 30.0, convergence_timeout_seconds, atol)

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
            self.link_q_indices,
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
        return forward_kinematics(self.dh_params, q[self.link_q_indices], ee_transform)

    def forward_kinematics_all_links(
        self,
        q: np.ndarray,
        ee_transform: np.ndarray = np.identity(4),
        convergence_tol=1e-4,
    ):
        self._check_data_length(q)
        return forward_kinematics_all_links(
            self.dh_params, q[self.link_q_indices], ee_transform, convergence_tol
        )

    def get_jacobian(self, q: np.ndarray):
        return calculate_basic_jacobian_xyz_omega(
            self.dh_params, q[self.link_q_indices]
        )

    def servo_q(self, q: np.ndarray):
        start = time.time()
        self._check_data_length(q)
        self._check_range_q(q)
        self._servo_q_impl(q)
        end = time.time()
        time.sleep(max(0, self.control_cycle - (end - start)))

    def servo_p(self, xyz: np.ndarray | None, rpy: np.ndarray | None):
        q = self.inverse_kinematics(xyz, rpy, self.get_present_q())
        if q is None:
            raise RuntimeError("Failed to obtain inverse kinematics solution")
        self.servo_q(q)

    def servo_dq(self, dq: np.ndarray):
        start = time.time()
        self._check_data_length(dq)
        self._check_range_q(dq)
        self._servo_dq_impl(dq)
        end = time.time()
        time.sleep(max(0, self.control_cycle - (end - start)))

    def servo_dp(self, dxyz: np.ndarray, drpy: np.ndarray, epsilon=1e-5):
        J = self.get_jacobian(self.get_present_q())
        if dxyz is None and drpy is None:
            raise ValueError("dxyz and drpy cannot be both None")

        if dxyz is None:
            drot = rotation_matrix_rpy(drpy[0], drpy[1], drpy[2])
            daxis, dangle = rotation_matrix_to_axis_and_angle(drot)
            dq = (
                J[3:6, :].T
                @ np.linalg.inv(J[3:6, :] @ J[3:6, :].T + epsilon * np.identity(3))
                @ daxis
                * dangle
            )
        elif drpy is None:
            dq = (
                J[:3, :].T
                @ np.linalg.inv(J[:3, :] @ J[:3, :].T + epsilon * np.identity(3))
                @ dxyz
            )
        else:
            drot = rotation_matrix_rpy(drpy[0], drpy[1], drpy[2])
            daxis, dangle = rotation_matrix_to_axis_and_angle(drot)
            dq = (
                J.T
                @ np.linalg.inv(J @ J.T + epsilon * np.identity(6))
                @ np.concatenate([dxyz, daxis * dangle])
            )

        self.servo_dq(dq)

    def servo_voltage(self, v: np.ndarray):
        self._check_data_length(v)
        self._servo_voltage_impl()

    def get_present_q(self):
        return self._get_present_q_impl()

    def get_present_p(self):
        ret = self.get_present_q()
        fk = self.forward_kinematics(ret)
        return transform_to_xyz_rpy(fk)

    def get_present_dq(self):
        return self._get_present_dq_impl()

    def get_present_dp(self) -> None:
        ret = self.get_present_dq()
        J = self.get_jacobian(self.get_present_q())
        return J @ ret[self.link_q_indices]

    def get_present_voltage(self):
        self._get_present_voltage_impl()

    def set_q_p_gains(self, gains: np.ndarray):
        self._check_data_length(gains)
        self._set_q_p_gains_impl(gains)

    def set_q_i_gains(self, gains: np.ndarray):
        self._check_data_length(gains)
        self._set_q_i_gains_impl(gains)

    def set_q_d_gains(self, gains: np.ndarray):
        self._check_data_length(gains)
        self._set_q_d_gains_impl(gains)

    def get_q_p_gains(self):
        self._get_q_p_gains_impl()

    def get_q_i_gains(self):
        self._get_q_i_gains_impl()

    def get_q_d_gains(self):
        self._get_q_d_gains_impl()

    def get_dq_p_gains(self):
        self._get_dq_p_gains_impl()

    def get_dq_i_gains(self):
        self._get_dq_i_gains_impl()

    def get_q_range(self):
        return self.q_range

    def get_dq_range(self):
        return self.dq_range

    def set_operation_mode(self, operating_mode: OperatingMode):
        self._set_operation_mode_impl(operating_mode)

    def get_operation_mode(self) -> OperatingMode:
        return self._get_operation_mode_impl()

    def get_present_tempratures(self):
        return self._get_present_tempratures_impl()

    def reboot(self):
        self.make_control_disable()
        self._reboot_impl()

    @abstractmethod
    def _reboot_impl(self):
        pass

    @abstractmethod
    def _is_control_enabled_impl(self):
        pass

    @abstractmethod
    def _make_control_enable_impl(self):
        pass

    @abstractmethod
    def _make_control_disable_impl(self):
        pass

    @abstractmethod
    def _servo_q_impl(self, q: np.ndarray):
        pass

    @abstractmethod
    def _servo_dq_impl(self, dq: np.ndarray):
        pass

    @abstractmethod
    def _servo_voltage_impl(self, v: np.ndarray):
        pass

    @abstractmethod
    def _get_present_q_impl(self):
        pass

    @abstractmethod
    def _get_present_dq_impl(self):
        pass

    @abstractmethod
    def _get_present_voltage_impl(self):
        pass

    @abstractmethod
    def _set_q_p_gains_impl(self, gains: np.ndarray):
        pass

    @abstractmethod
    def _set_q_i_gains_impl(self, gains: np.ndarray):
        pass

    @abstractmethod
    def _set_q_d_gains_impl(self, gains: np.ndarray):
        pass

    @abstractmethod
    def _get_q_p_gains_impl(self):
        pass

    @abstractmethod
    def _get_q_i_gains_impl(self):
        pass

    @abstractmethod
    def _get_q_d_gains_impl(self):
        pass

    @abstractmethod
    def _get_dq_p_gains_impl(self):
        pass

    @abstractmethod
    def _get_dq_i_gains_impl(self):
        pass

    @abstractmethod
    def _set_operation_mode_impl(self, operating_mode: OperatingMode):
        pass

    @abstractmethod
    def _get_operation_mode_impl(self) -> OperatingMode:
        pass

    @abstractmethod
    def _get_present_tempratures_impl(self):
        pass
