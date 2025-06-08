from koch11.dynamixel.koch11 import make_leader_client, make_follower_client

leader_client = make_leader_client()
follower_client = make_follower_client()

while True:
    leader_q, follower_q = (
        leader_client.get_present_q(),
        follower_client.get_present_q(),
    )

    print(f"Follower current q: {follower_q}, Leader current q: {leader_q}")
