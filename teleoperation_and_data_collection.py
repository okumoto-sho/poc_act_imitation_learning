import cv2 as cv
import time
import h5py
import numpy as np

from absl import flags, app
from koch11.dynamixel.koch11 import make_leader_client, make_follower_client
from teleoperation_config import teleoperation_config
from koch11.camera import Camera

FLAGS = flags.FLAGS
flags.DEFINE_integer("initial_episode_id", 0, "Initial episode ID.")
flags.DEFINE_string(
    "dataset_dir", "./datasets/pick_and_place/train", "Directory to save the dataset."
)
flags.DEFINE_integer(
    "num_episodes", 50, "Number of episodes to collect data for teleoperation."
)


def execute_single_teleoperation_step(
    episode_id: int, dataset_dir: str, follower, leader, cameras
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
        command_q = leader.get_present_q()
        follower.servo_q(command_q)

        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))
        print(f"FPS: {1 / (end - start)}")

    # prepare the data dictionary
    max_time_steps = teleoperation_config["episode_len_time_steps"]
    data_dict = {
        "/observations/qpos": np.zeros(shape=(max_time_steps, 6)),
        "/action": np.zeros(shape=(max_time_steps, 6)),
    }
    for cam in teleoperation_config["camera_configs"]:
        data_dict[f"/observations/images/{cam['device_name']}"] = np.zeros(
            shape=(max_time_steps, cam["height"], cam["width"], 3), dtype=np.uint8
        )

    # start the data collection
    print("Start data collection.")
    for step in range(max_time_steps):
        start = time.time()
        qpos = follower.get_present_q()
        command_q = leader.get_present_q()

        data_dict["/observations/qpos"][step, :] = qpos
        data_dict["/action"][step, :] = command_q
        for camera in cameras:
            frame = camera.read(flip=True)
            data_dict[f"/observations/images/{camera.device_name}"][step, :] = frame

        follower.servo_q(command_q)
        end = time.time()
        if (end - start) < control_cycle:
            time.sleep(control_cycle - (end - start))
        print(f"Coillecting data. Step: {step}, FPS: {1 / (end - start)}")

    #  save the collected data
    with h5py.File(f"{dataset_dir}/{episode_id}.h5", "w") as f:
        obs = f.create_group("observations")
        images = obs.create_group("images")
        for cam in teleoperation_config["camera_configs"]:
            images.create_dataset(
                f"{cam['device_name']}",
                data=f"{dataset_dir}/images/{episode_id}_{cam['device_name']}.mp4",
            )
        obs.create_dataset("qpos", (max_time_steps, 6), dtype="float32")
        f.create_dataset("action", (max_time_steps, 6), dtype="float32")

        f["/observations/qpos"][:] = data_dict["/observations/qpos"]
        f["/action"][:] = data_dict["/action"]
        for cam in teleoperation_config["camera_configs"]:
            out = cv.VideoWriter(
                f"{dataset_dir}/images/{episode_id}_{cam['device_name']}.mp4",
                cv.VideoWriter_fourcc(*"mp4v"),
                1.0 / teleoperation_config["control_cycle"],
                (cam["width"], cam["height"]),
            )
            if not out.isOpened():
                raise ValueError(
                    f"Error: Cannot open {episode_id}_{cam['device_name']}.mp4"
                )

            for step in range(max_time_steps):
                frame = data_dict[f"/observations/images/{cam['device_name']}"][step, :]
                out.write(frame)
            out.release()


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
