import numpy as np
import genesis as gs
import time

from genesis.engine.entities.rigid_entity import RigidEntity
from koch11.core.kinematics.kinematics import (
    forward_kinematics_xyz_rpy,
    inverse_kienmatics,
)
from koch11.dynamixel.koch11 import dh_params


def prepare_simulation(show_viewer: bool = True):
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=30,
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=show_viewer,
    )
    robot: RigidEntity = scene.add_entity(
        gs.morphs.URDF(
            file="./assets/urdf/koch11.urdf",
            pos=(0.0, 0.0, 0.0),
            euler=(0, 0, 0),
            fixed=True,
        ),
    )
    scene.build()

    return scene, robot


scene, robot = prepare_simulation(show_viewer=True)
direction = 1
dofs_index = [0, 1, 2, 3, 4]
delta = 0.01
q = np.array([0.0, np.deg2rad(10.0), np.deg2rad(45.0), np.deg2rad(90.0), 0.0])
pos = forward_kinematics_xyz_rpy(dh_params, q)
pos[4] = np.deg2rad(30.0)
dpos = np.array([0.001, 0.001, 0.0, 0.0, 0.0, 0.0])
epsilon = 1e-6

print(pos)
while 1:
    if ((pos + dpos)[0] > 0.15 and pos[0] < 0.15) or (
        (pos + dpos)[0] < 0.03 and pos[0] > 0.03
    ):
        dpos *= -1

    pos += dpos

    start = time.time()
    q = inverse_kienmatics(
        dh_params, pos[0:3], None, q, xyz_convergence_tolerance=0.0001, update_step=1.0
    )
    end = time.time()
    print(f"{(end - start) * 1000} {robot.get_links_pos()[4]}")

    robot.set_dofs_position(q, dofs_index)
    scene.step()
