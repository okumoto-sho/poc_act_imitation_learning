import argparse
import torch
import time
import numpy as np
import cv2 as cv

from koch11.dynamixel.robot_client import DynamixelRobotClient
from teleoperation_config import teleoperation_config
from model_config import model_config
from models import ActPolicy
from koch11.dynamixel.koch11 import make_follower_client
from koch11.camera import Camera


def main(args):
    policy = ActPolicy(
        model_config["image_feat_seq_length"],
        model_config["action_chunk_size"],
        model_config["action_dim"],
        model_config["qpos_dim"],
        model_config["emb_dim"],
        z_dim=model_config["z_dim"],
        n_enc_layers=model_config["n_enc_layers"],
        n_dec_layers=model_config["n_dec_layers"],
        n_heads=model_config["n_heads"],
        feedforward_dim=model_config["feedforward_dim"],
    ).cuda()

    checkpoint = args.checkpoint
    state_dict = torch.load(checkpoint)
    policy.encoder.load_state_dict(state_dict["encoder_state_dict"])
    policy.decoder.load_state_dict(state_dict["decoder_state_dict"])
    policy.backbone.load_state_dict(state_dict["backbone_state_dict"])

    policy.eval()

    episode_len = args.episode_len
    cameras_config = teleoperation_config["camera_configs"]
    cameras = []
    for camera_config in cameras_config:
        cameras.append(
            Camera(
                camera_config["device_id"],
                camera_config["fps"],
                camera_config["width"],
                camera_config["height"],
                camera_config["fourcc"],
            )
        )

    m = model_config["temporal_ensemble_log_discount"]
    action_sum_buffer = np.zeros((episode_len, model_config["action_dim"]))
    denominator_buffer = np.zeros((episode_len, 1))
    weights = np.exp(-m * np.arange(model_config["action_chunk_size"])).reshape(-1, 1)

    robot: DynamixelRobotClient = make_follower_client(
        teleoperation_config["follower_device"]
    )
    robot.make_control_enable()

    # Apply initial action to the robot by move_q. This is neccesary to avoid the sudden movement of the robot.
    qpos = robot.get_present_q()
    qpos_data = torch.tensor(qpos).float().cuda().unsqueeze(0)
    image = cameras[0].read(flip=True)
    images_data = torch.tensor(image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)
    actions_pred = policy.inference(qpos_data, images_data)[0].cpu().detach().numpy()
    robot.move_q(actions_pred[0, :])

    for i in range(episode_len):
        start = time.time()
        qpos = robot.get_present_q()
        qpos_data = torch.tensor(qpos).float().cuda().unsqueeze(0)

        image = cameras[0].read(flip=True)
        images_data = (
            torch.tensor(image / 255.0).float().cuda().unsqueeze(0).unsqueeze(0)
        )

        actions_pred = (
            policy.inference(qpos_data, images_data)[0].cpu().detach().numpy()
        )

        cv.imshow("0", image)
        if cv.waitKey(1) == ord("q"):
            break

        action_length = (
            model_config["action_chunk_size"]
            if i + model_config["action_chunk_size"] < episode_len
            else episode_len - i
        )
        action_sum_buffer[i : i + action_length, :] += (
            weights[0:action_length] * actions_pred[0:action_length, :]
        )
        denominator_buffer[i : i + action_length] += weights[0:action_length]
        action_taken = action_sum_buffer[i] / denominator_buffer[i]

        robot.servo_q(action_taken)

        end = time.time()
        if (end - start) < teleoperation_config["control_cycle"]:
            time.sleep(teleoperation_config["control_cycle"] - (end - start))
        print(f"Step: {i}, FPS: {1 / (end - start)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--episode_len", type=int, default=10000)
    args = parser.parse_args()
    main(args)
