import numpy as np

from numpy.typing import ArrayLike


def rotation_matrix_x(radians: float):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(radians), -np.sin(radians), 0],
            [0, np.sin(radians), np.cos(radians), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_matrix_y(radians: float):
    return np.array(
        [
            [np.cos(radians), 0, np.sin(radians), 0],
            [0, 1, 0, 0],
            [-np.sin(radians), 0, np.cos(radians), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_matrix_z(radians: float):
    return np.array(
        [
            [np.cos(radians), -np.sin(radians), 0, 0],
            [np.sin(radians), np.cos(radians), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_matrix_rpy(roll_radians: float, pitch_radians: float, yaw_radians: float):
    return (
        rotation_matrix_z(yaw_radians)
        @ rotation_matrix_y(pitch_radians)
        @ rotation_matrix_x(roll_radians)
    )


def translation_matrix(d: ArrayLike):
    return np.array([[1, 0, 0, d[0]], [0, 1, 0, d[1]], [0, 0, 1, d[2]], [0, 0, 0, 1]])


def skew(axis: np.ndarray):
    ax, ay, az = axis
    return np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])


def rotation_around_axis(axis: np.ndarray, radians: float):
    K = skew(axis / np.linalg.norm(axis))
    rot = np.identity(4)
    rot[0:3, 0:3] = np.identity(3) + np.sin(radians) * K + (1 - np.cos(radians)) * K @ K
    return rot


def rotation_matrix_to_rpy_radians(matrix: np.array, yaw_radians_on_singular: float):
    c_theta = np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2)

    if np.isclose(c_theta, 0.0, atol=1e-5):
        phi = yaw_radians_on_singular
        theta = 90.0
        psi = np.arctan2(matrix[1, 0], matrix[1, 1])
    else:
        phi = np.arctan2(matrix[1, 0], matrix[0, 0])
        theta = np.arctan2(-matrix[2, 0], c_theta)
        psi = np.arctan2(matrix[2, 1], matrix[2, 2])

    return np.array([psi, theta, phi])


def transform_to_xyz_rpy(T: np.array):
    position = np.array([T[0, 3], T[1, 3], T[2, 3]])
    rpy = rotation_matrix_to_rpy_radians(T[0:3, 0:3], 0.0)
    return np.concatenate([position, rpy])


def rotation_matrix_to_axis_and_angle(matrix: np.array):
    trace = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    radians = np.arccos((trace - 1) / 2.0)
    axis = -np.array(
        [
            matrix[1, 2] - matrix[2, 1],
            matrix[2, 0] - matrix[0, 2],
            matrix[0, 1] - matrix[1, 0],
        ]
    )
    if np.linalg.norm(axis) > 1e-5:
        axis = axis / np.linalg.norm(axis)
    return axis, radians


def mult_quat(quat_1: np.array, quat_2: np.array):
    qw, qx, qy, qz = quat_1[0], quat_1[1], quat_1[2], quat_1[3]
    pw, px, py, pz = quat_2[0], quat_2[1], quat_2[2], quat_2[3]
    return np.array(
        [
            qw * pw - qx * px - qy * py - qz * pz,
            qw * px + qx * pw + qy * pz - qz * py,
            qw * py - qx * pz + qy * pw + qz * px,
            qw * pz + qx * py - qy * px + qz * pw,
        ]
    )


def quat_around_axis(axis: np.array, theta: float):
    lambda_x, lambda_y, lambda_z = axis
    theta_rad = theta
    return np.array(
        [
            np.cos(theta_rad / 2.0),
            lambda_x * np.sin(theta_rad / 2.0),
            lambda_y * np.sin(theta_rad / 2.0),
            lambda_z * np.sin(theta_rad / 2.0),
        ]
    )


def quat_around_x(theta_radians: float):
    return quat_around_axis(np.array([1, 0, 0]), theta_radians)


def quat_around_y(theta_radians: float):
    return quat_around_axis(np.array([0, 1, 0]), theta_radians)


def quat_around_z(theta_radians: float):
    return quat_around_axis(np.array([0, 0, 1]), theta_radians)


def quat_rpy_radians(roll_radians: float, pitch_radians: float, yaw_radians: float):
    quat_x = quat_around_x(roll_radians)
    quat_y = quat_around_y(pitch_radians)
    quat_z = quat_around_z(yaw_radians)
    return mult_quat(quat_z, mult_quat(quat_y, quat_x))


def quat_to_rotation_matrix(quat: np.array):
    # 必要に応じて四元数を正規化（長さが非常に小さい場合は単位行列を返すなどの処理も考慮）
    norm_q = np.linalg.norm(quat)
    if norm_q < 1e-15:
        # 値が小さすぎる場合は回転なし(単位行列)とみなす
        return np.eye(3)

    qw, qx, qy, qz = quat / norm_q

    # 回転行列を計算
    R = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

    return R


def transform_to_quat(T: np.array):
    rpy = rotation_matrix_to_rpy_radians(T[0:3, 0:3], 0.0)
    quat = quat_rpy_radians(rpy[0], rpy[1], rpy[2])
    return quat


def transform_to_pos_quat(T: np.array):
    position = np.array([T[0, 3], T[1, 3], T[2, 3]])
    rpy = rotation_matrix_to_rpy_radians(T[0:3, 0:3], 0.0)
    quat = quat_rpy_radians(rpy[0], rpy[1], rpy[2])
    return np.concatenate([position, quat])
