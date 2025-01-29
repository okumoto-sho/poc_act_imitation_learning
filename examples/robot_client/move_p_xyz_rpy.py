import numpy as np

from koch11.dynamixel.koch11 import make_follower_client

client = make_follower_client()
client.make_control_enable()

p = client.get_present_p()
dpos = np.array([0.08, 0.0, -0.08, 0.0, 0.0, 0.0])
print(p)

max_xyz_speed = 0.1
max_q_speed = 0.5
for _ in range(100):
    max_xyz_speed += 0.03
    max_q_speed += 0.05
    cur_p = p + dpos
    client.move_p(cur_p[0:3], cur_p[3:6], xyz_max_speed=max_xyz_speed, atol=0.0001)
    client.move_q(np.array([0, 0, 0, 0, 0, 0]), atol=0.0001, speed=max_q_speed)
