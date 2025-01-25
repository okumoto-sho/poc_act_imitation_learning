import numpy as np

from koch11.dynamixel.koch11 import make_client

client = make_client()
client.make_control_enable()

p = client.get_present_p()
dpos = np.array([0.15, 0.05, -0.10])
print(p)

max_xyz_speed = 0.1
max_q_speed = 0.5
for _ in range(100):
    max_xyz_speed += 0.03
    max_q_speed += 0.05
    client.move_p(p[0:3] + dpos, None, xyz_max_speed=max_xyz_speed, atol=0.0001)
    print(client.get_present_p()[0:3] - p[0:3])
    client.move_q(np.array([0, 0, 0, 0, 0]), atol=0.0001, speed=max_q_speed)

# client.make_control_disable()
