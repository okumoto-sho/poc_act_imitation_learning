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
    "min": np.array(
        [
            -2 * np.pi,
            -np.pi / 2 - 0.1,
            -np.pi / 2 - 0.1,
            -7 * np.pi / 10,
            -np.pi,
            -np.pi / 6,
        ]
    ),
    "max": np.array(
        [2 * np.pi, np.pi / 2 + 0.1, np.pi, 7 * np.pi / 10, np.pi, np.pi / 2]
    ),
}

dq_range = {
    "min": np.array(
        [-2 * np.pi, -2 * np.pi, -3 * np.pi, -3 * np.pi, -3 * np.pi, -3 * np.pi]
    ),
    "max": np.array([2 * np.pi, 2 * np.pi, 3 * np.pi, 3 * np.pi, 3 * np.pi, 3 * np.pi]),
}

q_offsets_follower = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi / 2, -np.pi])
q_rot_direction_follower = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

q_offsets_leader = np.array([-np.pi, 0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi])
q_rot_direction_leader = np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0])


def make_follower_client(
    port_name: str = "/dev/ttyACM2", baud_rate=1000000
) -> DynamixelRobotClient:
    return DynamixelRobotClient(
        [1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4],
        dh_params,
        q_range,
        dq_range,
        port_name=port_name,
        baud_rate=baud_rate,
        retry_num=30,
        q_offsets=q_offsets_follower,
        q_rot_direction=q_rot_direction_follower,
    )


def make_leader_client(
    port_name: str = "/dev/ttyACM1", baud_rate=2000000
) -> DynamixelRobotClient:
    return DynamixelRobotClient(
        [1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4],
        dh_params,
        q_range,
        dq_range,
        port_name=port_name,
        baud_rate=baud_rate,
        retry_num=30,
        q_offsets=q_offsets_leader,
        q_rot_direction=q_rot_direction_leader,
    )
