import numpy as np

from koch11.dynamixel.koch11 import make_leader_client, make_follower_client


follower = make_follower_client()
follower.make_control_enable()
follower.move_q(np.array([0, 0, 0, 0, 0, 0]))

leader = make_leader_client()

while True:
    q = leader.get_present_q()
    follower.servo_q(q)
    print(q)
