import numpy as np
import dataclasses

from typing import List
from enum import IntEnum


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

    def make_control_enable(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def make_control_disable(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def move_q(self, q: np.ndarray, speed: float):
        print(self._not_implemented_message)
        raise NotImplementedError

    def move_q_ik(self, p: np.ndarray, speed: float):
        print(self._not_implemented_message)
        raise NotImplementedError

    def move_p(self, q: np.ndarray, speed: float):
        print(self._not_implemented_message)
        raise NotImplementedError

    def move_p_ik(self, p: np.ndarray, speed: float):
        print(self._not_implemented_message)
        raise NotImplementedError

    def move_q_path(self, q_path: List[np.ndarray], speed: float):
        print(self._not_implemented_message)
        raise NotImplementedError

    def move_p_path(self, p_path: List[np.ndarray], speed: float):
        print(self._not_implemented_message)
        raise NotImplementedError

    def inverse_kinematics(self, p: np.ndarray, convergence_tol=10e-4):
        print(self._not_implemented_message)
        raise NotImplementedError

    def forward_kinematics(
        self, q: np.ndarray, ee_transform: np.ndarray = np.identity(4)
    ):
        print(self._not_implemented_message)
        raise NotImplementedError

    def forward_kinematics_all_links(
        self,
        q: np.ndarray,
        ee_transform: np.ndarray = np.identity(4),
        convergence_tol=1e-4,
    ):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_jacobian(self, q: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def servo_q(self, q: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def servo_p(self, p: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def servo_dq(self, dq: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def servo_dp(self, dp: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def servo_voltage(self, v: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_present_q(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_present_p(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_present_dq(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_present_dp(self) -> None:
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_present_voltage(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def set_q_p_gains(self, gains: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def set_q_i_gains(self, gains: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def set_q_d_gains(self, gains: np.ndarray):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_dq_p_gains(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_dq_i_gains(self):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_q_range(self) -> QRange:
        print(self._not_implemented_message)
        raise NotImplementedError

    def set_operation_mode(self, operating_mode: OperatingMode):
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_operation_mode(self) -> OperatingMode:
        print(self._not_implemented_message)
        raise NotImplementedError

    def get_present_tempratures(self):
        print(self._not_implemented_message)
        raise NotImplementedError
