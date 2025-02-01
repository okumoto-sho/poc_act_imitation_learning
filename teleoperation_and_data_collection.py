import argparse
import cv2 as cv
import time
import h5py

from koch11.dynamixel.koch11 import make_leader_client, make_follower_client
from teleoperation_config import teleoperation_config


def main(args):
    # setting robots
    follower = make_follower_client(teleoperation_config["follower_device"])
    follower.make_control_enable()
    follower.move_q(teleoperation_config["follower_initail_q"], speed=2.0)
    leader = make_leader_client(teleoperation_config["leader_device"])

    # setting the camera settings
    cameras = []
    for camera_config in teleoperation_config["camera_configs"]:
        cap = cv.VideoCapture(camera_config["device_id"])
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*camera_config["fourcc"]))
        cap.set(cv.CAP_PROP_FPS, camera_config["fps"])
        cap.set(cv.CAP_PROP_FRAME_WIDTH, camera_config["width"])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
        cameras.append(cap)

    # get the follower robot move to initial position
    follower.move_q(teleoperation_config["follower_initail_q"], speed=1.0)

    # wait for press 'q' to start the data collection
    control_cycle = teleoperation_config["control_cycle"]
    print("Start the teleoperation.")
    print("Press 'q' to start the data collection.")
    while cv.waitKey(1) != ord("q"):
        start = time.time()
        leader_q = leader.get_present_q()
        follower.servo_q(leader_q)
        for cap in cameras:
            ret, frame = cap.read()
            frame = cv.flip(frame, 0)
            frame = cv.flip(frame, 1)
            cv.imshow("Frame", frame)
        end = time.time()
        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))

    # prepare the data dictionary
    data_dict = {"/observations/qpos": [], "/action": []}
    for cam in teleoperation_config["camera_configs"]:
        data_dict[f"/observations/images/{cam['device_id']}"] = []

    # start the data collection
    print("Start data collection.")
    max_time_steps = teleoperation_config["episode_len_time_steps"]
    for step in range(max_time_steps):
        start = time.time()
        qpos = follower.get_present_q()
        leader_q = leader.get_present_q()

        data_dict["/observations/qpos"].append(qpos)
        data_dict["/action"].append(leader_q)
        for i, cap in enumerate(cameras):
            ret, frame = cap.read()
            frame = cv.flip(frame, 0)
            frame = cv.flip(frame, 1)
            data_dict[f"/observations/images/{i}"].append(frame)

        follower.servo_q(leader_q)
        end = time.time()
        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))

    with h5py.File(
        f"{args.dataset_dir}/{args.task_name}_{args.episode_id}.h5", "w"
    ) as f:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_id", type=int)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--task_name", type=str)
    args = parser.parse_args()
    main(args)
