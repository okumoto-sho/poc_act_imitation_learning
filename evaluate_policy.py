import torch
import time
import numpy as np
import cv2 as cv

from absl import flags, app
from koch11.dynamixel.robot_client import DynamixelRobotClient
from teleoperation_config import teleoperation_config, camera_device_names
from model_config import model_config
from models.act import ActPolicy
from koch11.dynamixel.koch11 import make_follower_client
from koch11.camera import Camera

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "./checkpoints/act_policy.pth", "Checkpoint path")
flags.DEFINE_integer("episode_len", 10000, "Episode length")
flags.DEFINE_enum("dtype", "float16", ["float16", "float32"], "Data type")
flags.DEFINE_bool("record", False, "Record video")
flags.DEFINE_string("record_dir", "./records", "Record directory")


def main(_):
    if FLAGS.dtype == "float32":
        dtype = torch.float32
    elif FLAGS.dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Invalid dtype: {FLAGS.dtype}")

    policy = (
        ActPolicy(
            camera_device_names,
            model_config["action_chunk_size"],
            model_config["action_dim"],
            model_config["qpos_dim"],
            model_config["emb_dim"],
            z_dim=model_config["z_dim"],
            n_enc_layers=model_config["n_enc_layers"],
            n_dec_layers=model_config["n_dec_layers"],
            n_heads=model_config["n_heads"],
            feedforward_dim=model_config["feedforward_dim"],
        )
        .cuda()
        .to(dtype=dtype)
    )

    checkpoint = FLAGS.checkpoint
    state_dict = torch.load(checkpoint)
    policy.load_state_dict(state_dict["parameters_state_dict"])

    policy.eval()

    episode_len = FLAGS.episode_len
    cameras_configs = teleoperation_config["camera_configs"]
    cameras = {}
    for camera_config in cameras_configs:
        cameras[camera_config["device_name"]] = Camera(**camera_config)

    m = model_config["temporal_ensemble_log_discount"]
    action_sum_buffer = np.zeros((episode_len, model_config["action_dim"]))
    denominator_buffer = np.zeros((episode_len, 1))
    weights = np.exp(-m * np.arange(model_config["action_chunk_size"])).reshape(-1, 1)

    robot: DynamixelRobotClient = make_follower_client(
        teleoperation_config["follower_device"]
    )
    # robot.make_control_enable()

    # prepare vidoe capture
    if FLAGS.record:
        out = {}
        for camera_config in cameras_configs:
            fourcc = cv.VideoWriter_fourcc(*camera_config["fourcc"])
            out[camera_config["device_name"]] = cv.VideoWriter(
                f"{FLAGS.record_dir}/{camera_config['device_name']}.avi",
                fourcc,
                30,
                (camera_config["width"], camera_config["height"]),
            )

    # Apply initial action to the robot by move_q. This is neccesary to avoid the sudden movement of the robot.
    qpos = robot.get_present_q()
    qpos_data = torch.tensor(qpos).to(dtype=dtype).cuda().unsqueeze(0)

    images = {}
    for camera_name in camera_device_names:
        image = cameras[camera_name].read(flip=True)
        images_data = torch.tensor(image / 255.0).to(dtype=dtype).cuda().unsqueeze(0)
        images[camera_name] = images_data

    actions_pred = policy.inference(qpos_data, images)[0].cpu().detach().numpy()
    robot.move_q(actions_pred[0, :])

    for i in range(episode_len):
        start = time.time()
        qpos = robot.get_present_q()
        qpos_data = torch.tensor(qpos).to(dtype=dtype).cuda().unsqueeze(0)

        images = {}
        for camera_name in camera_device_names:
            image = cameras[camera_name].read(flip=True)
            images_data = (
                torch.tensor(image / 255.0).to(dtype=dtype).cuda().unsqueeze(0)
            )
            images[camera_name] = images_data
            cv.imshow(camera_name, image)
            if FLAGS.record:
                out[camera_name].write(image)

        with torch.inference_mode():
            actions_pred = policy.inference(qpos_data, images)[0].cpu().detach().numpy()

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

    for cap in out.values():
        cap.release()


if __name__ == "__main__":
    app.run(main)
