import numpy as np

from koch11.dynamixel.koch11 import make_client

client = make_client()
client.make_control_enable()

p = client.get_present_p()
print(p)

max_xyz_speed = 0.1
max_q_speed = 5.0
for _ in range(100):
    max_xyz_speed += 0.03
    max_q_speed += 0.05
    client.move_p(
        None, np.array([0.0, 0.0, np.pi / 2]), atol=0.0001, rpy_max_speed=10.0
    )
    client.move_q(np.array([0, 0, 0, 0, 0]), atol=0.0001, speed=max_q_speed)
    client.move_p(
        None, np.array([0.0, 0.0, -np.pi / 2]), atol=0.0001, rpy_max_speed=10.0
    )
