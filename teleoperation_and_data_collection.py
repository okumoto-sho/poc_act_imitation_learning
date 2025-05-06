import cv2 as cv
import time
import numpy as np

from absl import flags, app
from koch11.dynamixel.koch11 import make_leader_client, make_follower_client
from teleoperation_config import teleoperation_config
from koch11.camera import Camera
from dataset import save_one_episode_data
from typing import List

FLAGS = flags.FLAGS
flags.DEFINE_integer("initial_episode_id", 0, "Initial episode ID.")
flags.DEFINE_string(
    "dataset_dir", "./datasets/pick_and_place/train", "Directory to save the dataset."
)
flags.DEFINE_integer(
    "num_episodes", 50, "Number of episodes to collect data for teleoperation."
)


def execute_single_teleoperation_step(
    episode_id: int, dataset_dir: str, follower, leader, cameras: List[Camera]
):
    # wait for press 'q' to start the data collection
    control_cycle = teleoperation_config["control_cycle"]
    print("Start the teleoperation.")
    print("Press 'q' to start the data collection.")
    while cv.waitKey(1) != ord("q"):
        start = time.time()
        for camera in cameras:
            frame = camera.read(flip=True)
            cv.imshow(f"Frame {camera.device_name}", frame)
        end = time.time()
        action = leader.get_present_q()
        follower.servo_q(action)

        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))
        print(f"FPS: {1 / (end - start)}")

    # prepare data buffers
    max_time_steps = teleoperation_config["episode_len_time_steps"]
    qpos_data = np.zeros(shape=(max_time_steps, 6), dtype=np.float32)
    action_data = np.zeros(shape=(max_time_steps, 6), dtype=np.float32)
    images_data = {}
    for cam in teleoperation_config["camera_configs"]:
        images_data[cam["device_name"]] = np.zeros(
            shape=(max_time_steps, cam["height"], cam["width"], 3), dtype=np.uint8
        )

    # start the data collection
    print("Start data collection.")
    for step in range(max_time_steps):
        start = time.time()
        qpos = follower.get_present_q()
        action = leader.get_present_q()

        qpos_data[step, :] = qpos
        action_data[step, :] = action
        for camera in cameras:
            frame = camera.read(flip=True)
            images_data[camera.device_name][step, :] = frame

        follower.servo_q(action)
        end = time.time()
        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))
        print(f"Coillecting data. Step: {step}, FPS: {1 / (end - start)}")

    #  save the collected data
    save_one_episode_data(
        dataset_dir=dataset_dir,
        episode_id=episode_id,
        qpos_data=qpos_data,
        action_data=action_data,
        images_data=images_data,
        control_cycle=control_cycle,
    )


def main(_):
    # setting robots
    follower = make_follower_client(teleoperation_config["follower_device"])
    follower.make_control_enable()
    follower.move_q(teleoperation_config["follower_initail_q"], speed=2.0)
    leader = make_leader_client(teleoperation_config["leader_device"])

    # setting the camera settings
    cameras = []
    for camera_config in teleoperation_config["camera_configs"]:
        cameras.append(Camera(**camera_config))

    # Repeat the teleoperation and data collection for the specified number of episodes
    for episode_id in range(
        FLAGS.initial_episode_id, FLAGS.initial_episode_id + FLAGS.num_episodes
    ):
        execute_single_teleoperation_step(
            episode_id, FLAGS.dataset_dir, follower, leader, cameras
        )


if __name__ == "__main__":
    app.run(main)
