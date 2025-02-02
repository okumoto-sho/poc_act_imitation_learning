from koch11.dynamixel.koch11 import make_follower_client

client = make_follower_client()

client.make_control_enable()
print(f"Control enabled: {client.is_control_enabled()}")
client.make_control_disable()
print(f"Control enabled: {client.is_control_enabled()}")
