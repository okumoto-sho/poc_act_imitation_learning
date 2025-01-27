import numpy as np
import time

from koch11.dynamixel.koch11 import make_client
from koch11.core.robot_client import OperatingMode

client = make_client()
client.make_control_disable()
client.set_operation_mode(OperatingMode.VelocityControl)
client.make_control_enable()

present_q = client.get_present_q()
client.move_q(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
start = time.time()
client.servo_dq(np.array([1.5, 0.0, 0.0, 0.0, 0.0]))
while time.time() - start < 10:
    prev_q = present_q
    present_q = client.get_present_q()
    if present_q[0] >= np.pi / 2 and prev_q[0] < np.pi / 2:
        client.servo_dq(np.array([-1.5, 0.0, 0.0, 0.0, 0.0]))
    elif present_q[0] <= -np.pi / 2 and prev_q[0] > -np.pi / 2:
        client.servo_dq(np.array([1.5, 0.0, 0.0, 0.0, 0.0]))

client.make_control_disable()
client.set_operation_mode(OperatingMode.PositionControl)
client.make_control_enable()
