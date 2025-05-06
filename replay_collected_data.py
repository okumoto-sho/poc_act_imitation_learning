import cv2 as cv
import time

from absl import flags, app
from koch11.dynamixel.koch11 import make_follower_client
from teleoperation_config import teleoperation_config
from dataset import read_full_steps_data

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dataset_dir",
    "./datasets/pick_and_place/train",
    "Path to the dataset.",
)
flags.DEFINE_integer(
    "episode_id",
    0,
    "Episode ID to replay.",
)
flags.DEFINE_boolean(
    "move_robot",
    False,
    "If true, the robot will be moved to the recorded qpos.",
)


def main(_):
    cameras_config = teleoperation_config["camera_configs"]
    device_names = [config["device_name"] for config in cameras_config]
    dataset_dict = read_full_steps_data(
        dataset_dir=FLAGS.dataset_dir,
        episode_id=FLAGS.episode_id,
        camera_devices=device_names,
    )

    if FLAGS.move_robot:
        follower = make_follower_client()
        follower.move_q(dataset_dict["qpos"][0])

    control_cycle: float = teleoperation_config["control_cycle"]
    print("Press 'q' to quit")
    print(f"Replay the recorded data with length {len(dataset_dict['action'])}")
    for i in range(len(dataset_dict["action"])):
        start = time.time()
        for cam in cameras_config:
            image = dataset_dict[f"/images/{cam['device_name']}"][i]
            cv.imshow(str(cam["device_name"]), image)

        print("Send the action {} to the robot.".format(dataset_dict["action"][i]))
        if FLAGS.move_robot:
            follower.servo_q(dataset_dict["action"][i])

        if cv.waitKey(1) == ord("q"):
            break
        end = time.time()
        if end - start < control_cycle:
            time.sleep(control_cycle - (end - start))

        print(f"Step: {i} FPS: {1 / (end - start)}")


if __name__ == "__main__":
    app.run(main)
