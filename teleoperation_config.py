import numpy as np

camera_config: dict = {
    "fourcc": "MJPG",
    "fps": 120.0,
    "width": 640,
    "height": 480,
    "device_id": 0,
}

teleoperation_config: dict = {
    "leader_device": "/dev/ttyACM1",
    "follower_device": "/dev/ttyACM2",
    "episode_len_time_steps": 1000,
    "control_cycle": 0.01,
    "camera_configs": [camera_config],
    "follower_initail_q": np.array([0, 0, 0, 0, 0, 0]),
}
