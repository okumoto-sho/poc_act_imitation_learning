import numpy as np

from koch11.dynamixel.koch11 import make_follower_client

client = make_follower_client()
client.make_control_enable()

q = np.array([0, 0, 0, 0, 0])
client.move_q(q, 1.0)

# client.make_control_disable()
