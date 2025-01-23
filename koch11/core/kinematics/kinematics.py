import numpy as np
import dataclasses

from typing import List
from koch11.core.kinematics.math_utils import (
    rotation_matrix_x,
    rotation_matrix_z,
    translation_matrix,
    rotation_matrix_rpy,
    rotation_matrix_to_axis_and_angle,
    transform_to_xyz_rpy,
)


@dataclasses.dataclass
class DhParam:
    a: float
    alpha: float
    d: float
    theta: float


def forward_kinematics(
    dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)
):
    fk_all_links = forward_kinematics_all_links(dh_params, q_radians, ee_transform)
    return fk_all_links[-1]


def forward_kinematics_xyz_rpy(
    dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)
):
    fk_all_links = forward_kinematics_all_links(dh_params, q_radians, ee_transform)
    return transform_to_xyz_rpy(fk_all_links[-1])


def forward_kinematics_all_links(
    dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)
):
    if len(dh_params) != len(q_radians):
        raise ValueError("Given different length dh_params and q_radians")

    T = np.identity(4)
    fk_all_links = []
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


def calculate_z_directions_all_links(
    dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)
):
    fk_all_links = forward_kinematics_all_links(dh_params, q_radians, ee_transform)
    return [T[0:3, 2] for T in fk_all_links]


def calculate_basic_jacobian_xyz_omega(
    dh_params: List[DhParam], q_radians, ee_transform=np.identity(4)
):
    transform_all_links = forward_kinematics_all_links(
        dh_params, q_radians, ee_transform
    )
    ee_pos = transform_all_links[-1][0:3, 3]
    jacobian_xyz_omega = np.zeros((6, len(q_radians)))
    for i, transform in enumerate(transform_all_links[:-1]):
        link_pos = transform[0:3, 3]
        rel_ee_link = ee_pos - link_pos
        z = transform[0:3, 2]
        jacobian_xyz_omega[0:3, i] = np.cross(z, rel_ee_link)
        jacobian_xyz_omega[3:6, i] = z
    return jacobian_xyz_omega


def _inverse_kinematics_xyz(
    dh_params: List[DhParam],
    xyz: np.ndarray,
    init_q_radians: np.ndarray,
    ee_transform=np.identity(4),
    update_step=1.0,
    max_iter=1000,
    xyz_convergence_tolerance=0.00001,
    epsilon=1e-5,
):
    current_q_radians = init_q_radians
    goal_pos = xyz
    for _ in range(max_iter):
        J = calculate_basic_jacobian_xyz_omega(
            dh_params, current_q_radians, ee_transform
        )[0:3, :]
        cur_transform = forward_kinematics(dh_params, current_q_radians, ee_transform)

        diff_pos = goal_pos - cur_transform[0:3, 3]
        diff = diff_pos
        if np.linalg.norm(diff_pos) < xyz_convergence_tolerance:
            break

        dq = J.T @ np.linalg.inv(J @ J.T + epsilon * np.identity(3)) @ diff
        current_q_radians = current_q_radians + update_step * dq
    return current_q_radians, np.linalg.norm(diff_pos)


def _inverse_kinematics_rpy(
    dh_params: List[DhParam],
    rpy: np.ndarray,
    init_q_radians: np.ndarray,
    ee_transform=np.identity(4),
    update_step=1.0,
    max_iter=1000,
    rpy_convergence_tolerance=0.001,
    epsilon=1e-5,
):
    current_q_radians = init_q_radians
    goal_rotation = rotation_matrix_rpy(rpy[0], rpy[1], rpy[2])
    for _ in range(max_iter):
        J = calculate_basic_jacobian_xyz_omega(
            dh_params, current_q_radians, ee_transform
        )[3:6, :]
        cur_transform = forward_kinematics(dh_params, current_q_radians, ee_transform)

        diff_axis, diff_radians = rotation_matrix_to_axis_and_angle(
            goal_rotation[0:3, 0:3] @ cur_transform[0:3, 0:3].T
        )
        diff = diff_axis * diff_radians
        if np.linalg.norm(diff_radians) < rpy_convergence_tolerance:
            break

        dq = J.T @ np.linalg.inv(J @ J.T + epsilon * np.identity(3)) @ diff
        current_q_radians = current_q_radians + update_step * dq

    return current_q_radians, np.linalg.norm(diff_radians)


def _inverse_kinematics_xyz_rpy(
    dh_params: List[DhParam],
    xyz: np.ndarray,
    rpy: np.ndarray,
    init_q_radians: np.ndarray,
    ee_transform=np.identity(4),
    update_step=1.0,
    max_iter=1000,
    xyz_convergence_tolerance=0.00001,
    rpy_convergence_tolerance=0.001,
    epsilon=1e-5,
):
    current_q_radians = init_q_radians
    goal_pos = xyz
    goal_rotation = rotation_matrix_rpy(rpy[0], rpy[1], rpy[2])
    for _ in range(max_iter):
        J = calculate_basic_jacobian_xyz_omega(
            dh_params, current_q_radians, ee_transform
        )
        cur_transform = forward_kinematics(dh_params, current_q_radians, ee_transform)

        diff_pos = goal_pos - cur_transform[0:3, 3]
        diff_axis, diff_radians = rotation_matrix_to_axis_and_angle(
            goal_rotation[0:3, 0:3] @ cur_transform[0:3, 0:3].T
        )
        diff = np.concatenate((diff_pos, diff_axis * diff_radians))
        if (
            np.linalg.norm(diff_pos) < xyz_convergence_tolerance
            and np.linalg.norm(diff_radians) < rpy_convergence_tolerance
        ):
            break

        dq = J.T @ np.linalg.inv(J @ J.T + epsilon * np.identity(6)) @ diff
        current_q_radians = current_q_radians + update_step * dq

    return current_q_radians, np.linalg.norm(diff_pos), np.linalg.norm(diff_radians)


def inverse_kienmatics(
    dh_params: List[DhParam],
    xyz: np.ndarray | None,
    rpy: np.ndarray | None,
    init_q_radians: np.ndarray,
    ee_transform=np.identity(4),
    update_step=1.0,
    max_iter=1000,
    xyz_convergence_tolerance=0.00001,
    rpy_convergence_tolerance=0.001,
    epsilon=1e-5,
) -> List[np.ndarray]:
    if xyz is None and rpy is None:
        raise ValueError("At least one of xyz or rpy must be given")

    diff_xyz, diff_rpy = 0.0, 0.0
    if xyz is None:
        q, diff_rpy = _inverse_kinematics_rpy(
            dh_params,
            rpy,
            init_q_radians,
            ee_transform,
            update_step,
            max_iter,
            rpy_convergence_tolerance,
            epsilon,
        )

    elif rpy is None:
        q, diff_xyz = _inverse_kinematics_xyz(
            dh_params,
            xyz,
            init_q_radians,
            ee_transform,
            update_step,
            max_iter,
            xyz_convergence_tolerance,
            epsilon,
        )
    else:
        q, diff_xyz, diff_rpy = _inverse_kinematics_xyz_rpy(
            dh_params,
            xyz,
            rpy,
            init_q_radians,
            ee_transform,
            update_step,
            max_iter,
            xyz_convergence_tolerance,
            rpy_convergence_tolerance,
            epsilon,
        )

    if diff_xyz > xyz_convergence_tolerance or diff_rpy > rpy_convergence_tolerance:
        print("Inverse kinematics did not converge")
        return None

    return q
