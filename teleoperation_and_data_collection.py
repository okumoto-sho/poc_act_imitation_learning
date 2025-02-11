import argparse
import cv2 as cv
import time
import h5py
import numpy as np

from koch11.dynamixel.koch11 import make_leader_client, make_follower_client
from teleoperation_config import teleoperation_config
from koch11.camera import Camera


def execute_single_teleoperation_step(
    episode_id: int, task_name: str, dataset_dir: str, follower, leader, cameras
):
    # wait for press 'q' to start the data collection
    control_cycle = teleoperation_config["control_cycle"]
    print("Start the teleoperation.")
    print("Press 'q' to start the data collection.")
    while cv.waitKey(1) != ord("q"):
        start = time.time()
        command_q = leader.get_present_q()
        follower.servo_q(command_q)
        for camera in cameras:
            frame = camera.read(flip=True)
            cv.imshow("Frame", frame)
        end = time.time()
        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))

    # prepare the data dictionary
    max_time_steps = teleoperation_config["episode_len_time_steps"]
    data_dict = {
        "/observations/qpos": np.zeros(shape=(max_time_steps, 6)),
        "/action": np.zeros(shape=(max_time_steps, 6)),
    }
    for cam in teleoperation_config["camera_configs"]:
        data_dict[f"/observations/images/{cam['device_id']}"] = np.zeros(
            shape=(max_time_steps, cam["height"], cam["width"], 3)
        )

    # start the data collection
    print("Start data collection.")
    for step in range(max_time_steps):
        start = time.time()
        qpos = follower.get_present_q()
        command_q = leader.get_present_q()

        data_dict["/observations/qpos"][step, :] = qpos
        data_dict["/action"][step, :] = command_q
        for i, camera in enumerate(cameras):
            frame = camera.read(flip=True)
            data_dict[f"/observations/images/{i}"][step, :] = frame

        follower.servo_q(command_q)
        end = time.time()
        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))
        print(f"Coillecting data. Step: {step}, FPS: {1 / (end - start)}")

    #  save the collected data
    with h5py.File(f"{dataset_dir}/{task_name}_{episode_id}.h5", "w") as f:
        obs = f.create_group("observations")
        images = obs.create_group("images")
        for cam in teleoperation_config["camera_configs"]:
            images.create_dataset(
                f"{cam['device_id']}",
                (max_time_steps, cam["height"], cam["width"], 3),
                dtype="uint8",
            )
        obs.create_dataset("qpos", (max_time_steps, 6), dtype="float32")
        f.create_dataset("action", (max_time_steps, 6), dtype="float32")

        for name, array in data_dict.items():
            f[name][...] = array


def main(args):
    # setting robots
    follower = make_follower_client(teleoperation_config["follower_device"])
    follower.make_control_enable()
    follower.move_q(teleoperation_config["follower_initail_q"], speed=2.0)
    leader = make_leader_client(teleoperation_config["leader_device"])

    # setting the camera settings
    cameras = []
    for camera_config in teleoperation_config["camera_configs"]:
        cameras.append(
            Camera(
                camera_config["device_id"],
                camera_config["fps"],
                camera_config["width"],
                camera_config["height"],
                camera_config["fourcc"],
            )
        )

    # Repeat the teleoperation and data collection for the specified number of episodes
    for episode_id in range(
        args.initial_episode_id, args.initial_episode_id + args.num_episodes
    ):
        execute_single_teleoperation_step(
            episode_id, args.task_name, args.dataset_dir, follower, leader, cameras
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_episode_id", type=int, default=0)
    parser.add_argument("--dataset_dir", type=str, default="train_dataset")
    parser.add_argument("--task_name", type=str, default="teleoperation")
    parser.add_argument("--num_episodes", type=int, default=50)
    args = parser.parse_args()
    main(args)
