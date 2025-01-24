import numpy as np

from koch11.core.kinematics.kinematics import DhParam
from koch11.dynamixel.robot_client import DynamixelRobotClient

dh_params = [
    DhParam(0.0, 0.0, 0.0533, 0.0),
    DhParam(0.0, np.deg2rad(-90.0), 0.0, np.deg2rad(-82.21830862329406)),
    DhParam(0.10930658717570502, 0.0, 0.0, np.deg2rad(-6.23676770828996)),
    DhParam(0.10051653844019899, 0.0, -0.00135, np.deg2rad(-91.5449236684159757)),
    DhParam(0.0, np.deg2rad(-90.0), 0.0450, np.deg2rad(180.0)),
]

q_range = {
    "min": np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -7 * np.pi / 10, -np.pi]),
    "max": np.array([np.pi / 2, np.pi / 2, np.pi / 2, 7 * np.pi / 10, np.pi]),
}

dq_range = {
    "min": np.array([-2 * np.pi, -2 * np.pi, -3 * np.pi, -3 * np.pi, -3 * np.pi]),
    "max": np.array([2 * np.pi, 2 * np.pi, 3 * np.pi, 3 * np.pi, 3 * np.pi]),
}


def make_client(
    port_name: str = "/dev/ttyACM0", baud_rate=2000000
) -> DynamixelRobotClient:
    return DynamixelRobotClient(
        [1, 2, 3, 4, 5],
        dh_params,
        q_range,
        dq_range,
        port_name=port_name,
        baud_rate=baud_rate,
        retry_num=30,
    )
