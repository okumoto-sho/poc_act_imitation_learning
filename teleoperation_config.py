import numpy as np

camera_config_1: dict = {
    "fourcc": "MJPG",
    "fps": 120.0,
    "width": 640,
    "height": 480,
    "device_id": 0,
    "device_name": "0",
}
camera_config_2: dict = {
    "fourcc": "MJPG",
    "fps": 120.0,
    "width": 640,
    "height": 480,
    "device_id": 2,
    "device_name": "2",
}

teleoperation_config: dict = {
    "leader_device": "/dev/ttyACM0",
    "follower_device": "/dev/ttyACM1",
    "episode_len_time_steps": 500,
    "control_cycle": 0.025,
    "camera_configs": [camera_config_1],
    "follower_initail_q": np.array([0, 0, 0, 0, 0, 0]),
}

camera_names = [
    config["device_name"] for config in teleoperation_config["camera_configs"]
]
