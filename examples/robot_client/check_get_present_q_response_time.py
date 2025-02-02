import time

from koch11.dynamixel.koch11 import make_follower_client, make_leader_client

follower = make_follower_client()
follower.make_control_disable()

leader = make_leader_client()
leader.make_control_disable()

while True:
    start_follower = time.time()
    follower_q = follower.get_present_q()
    end_follower = time.time()

    start_leader = time.time()
    leader_q = leader.get_present_q()
    end_leader = time.time()

    if (end_follower - start_follower) > 0.01:
        print(
            f"FPS_Follower: {1 / (end_follower - start_follower)} FPS_Leader: {1 / (end_leader - start_leader)}"
        )
