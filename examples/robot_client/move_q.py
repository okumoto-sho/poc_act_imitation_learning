import numpy as np

from koch11.dynamixel.koch11 import make_follower_client

client = make_follower_client()
client.make_control_enable()

q = np.array([0, 0, 0, 0, 0, np.pi / 2])
client.move_q(q, 2.0)

# client.make_control_disable()
