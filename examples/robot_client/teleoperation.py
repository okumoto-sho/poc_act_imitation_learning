import numpy as np
import time

from koch11.dynamixel.koch11 import make_leader_client, make_follower_client


follower = make_follower_client()
follower.make_control_enable()
follower.move_q(np.array([0, 0, 0, 0, 0, 0]), speed=2.0)

leader = make_leader_client()
q_pos = leader.get_present_q()
follower.move_q(q_pos, speed=2.0)

while True:
    start = time.time()
    q_pos = follower.get_present_q()
    q = leader.get_present_q()
    follower.servo_q(q)
    end = time.time()
    print(
        q_pos,
        q,
        f"FPS: {1 / (end - start)}",
    )
