from koch11.dynamixel.koch11 import make_leader_client

client = make_leader_client()

while True:
    q = client.get_present_q()
    print(q)
