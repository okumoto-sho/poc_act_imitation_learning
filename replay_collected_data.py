import argparse
import cv2 as cv
import time

from koch11.dynamixel.koch11 import make_follower_client
from teleoperation_config import teleoperation_config
from dataset import read_h5_dataset


def main(args):
    cameras_config = teleoperation_config["camera_configs"]
    device_names = [config["device_name"] for config in cameras_config]
    dataset_dict = read_h5_dataset(args.dataset_path, device_names)

    if args.move_robot:
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

        if args.move_robot:
            follower.servo_q(dataset_dict["action"][i])

        if cv.waitKey(1) == ord("q"):
            break
        end = time.time()
        if end - start < control_cycle:
            time.sleep(control_cycle - (end - start))
            print("hi")

        print(f"Step: {i} FPS: {1 / (end - start)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--move_robot", action="store_true")
    args = parser.parse_args()
    main(args)
