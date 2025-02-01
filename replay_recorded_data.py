import argparse
import cv2 as cv
import h5py
import time

from koch11.dynamixel.koch11 import make_follower_client
from teleoperation_config import teleoperation_config


def main(args):
    cameras_config = teleoperation_config["camera_configs"]
    dataset_dict = {}
    with h5py.File(args.dataset_path, "r") as f:
        dataset_dict["/observations/qpos"] = f["/observations/qpos"][...]
        dataset_dict["/action"] = f["/action"][...]
        for cam in cameras_config:
            dataset_dict[f"/observations/images/{cam['device_id']}"] = f[
                f"/observations/images/{cam['device_id']}"
            ][...]

    if args.move_robot:
        follower = make_follower_client()
        follower.move_q(teleoperation_config["follower_initail_q"])

    control_cycle: float = teleoperation_config["control_cycle"]
    print("Press 'q' to quit")
    print(f"Replay the recorded data with length {len(dataset_dict['/action'])}")
    for i in range(len(dataset_dict["/action"])):
        start = time.time()
        for cam in cameras_config:
            image = dataset_dict[f"/observations/images/{cam['device_id']}"][i]
            cv.imshow(str(cam["device_id"]), image)

        if args.move_robot:
            follower.servo_q(dataset_dict["/action"][i])

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
