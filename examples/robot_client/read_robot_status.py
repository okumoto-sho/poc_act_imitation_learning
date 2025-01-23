from koch11.dynamixel.koch11 import make_client

client = make_client()

status = {
  "joint position": client.get_present_q(),
  "joint velocity": client.get_present_dq(),
  "cartesian position": client.get_present_p(),
  "cartesian velocity": client.get_present_dp(),
  "temperature": client.get_present_tempratures(),
  "joint P gains": client.get_q_p_gains(),
  "joint I gains": client.get_q_i_gains(),
  "joint D gains": client.get_q_d_gains(),
  "velocity P gains": client.get_dq_p_gains(),
  "velocity I gains": client.get_dq_i_gains(),
  "q range": client.get_q_range(),
  "dq range": client.get_dq_range(),
  "control enabled": client.is_control_enabled(),
  "operation mode": client.get_operation_mode(),
}

for key, value in status.items():
  print(f"{key}: {value}")