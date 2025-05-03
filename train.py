import torch

from absl import flags, app
from dataset import EpisodicDataset
from models.act import ActPolicy, ActOutput
from model_config import model_config
from teleoperation_config import camera_device_names
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_epochs", 10000, "number of epochs to train")
flags.DEFINE_string(
    "train_dataset_dir",
    "./train_dataset/pick_and_place_two_cameras/",
    "Dataset directory for training",
)
flags.DEFINE_string(
    "test_dataset_dir",
    "./test_dataset/pick_and_place_two_cameras/",
    "Dataset directory for testing",
)
flags.DEFINE_integer("num_episodes_train", 24, "number of episodes to train")
flags.DEFINE_integer("num_episodes_test", 5, "number of episodes to test")
flags.DEFINE_string("initial_checkpoint", None, "initial checkpoint path")
flags.DEFINE_string("checkpoints_dir", "./checkpoints", "checkpoints directory")
flags.DEFINE_float("learning_rate", 1e-5, "learning rate")
flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_string("tensorboard_log_dir", "./logs", "log directory")


def main(_):
    train_dataset = EpisodicDataset(
        dataset_dir=FLAGS.train_dataset_dir,
        num_episodes=FLAGS.num_episodes_train,
        camera_names=camera_device_names,
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size, shuffle=True
    )
    test_dataset = EpisodicDataset(
        dataset_dir=FLAGS.test_dataset_dir,
        num_episodes=FLAGS.num_episodes_test,
        camera_names=camera_device_names,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=FLAGS.batch_size, shuffle=True
    )

    act_policy = ActPolicy(
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
        kl_loss_weight=model_config["kl_loss_weight"],
    ).cuda()
    if FLAGS.initial_checkpoint is not None:
        checkpoint = torch.load(FLAGS.initial_checkpoint)
        act_policy.load_state_dict(checkpoint["parameters_state_dict"])
    act_policy.train()

    optimizer = torch.optim.AdamW(
        [
            {"params": act_policy.parameters()},
        ],
        lr=FLAGS.learning_rate,
    )

    writer = SummaryWriter(log_dir=FLAGS.tensorboard_log_dir)
    global_steps = 0
    for epoch in tqdm(range(FLAGS.num_epochs)):
        for data in train_data_loader:
            qpos, images, actions = data
            output: ActOutput = act_policy(qpos, actions, images)

            loss_total = output.loss_total

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            global_steps += 1
            writer.add_scalar("train/loss_total", loss_total.item(), global_steps)
            writer.add_scalar("train/loss_l2", output.loss_l2.item(), global_steps)
            writer.add_scalar("train/loss_kl", output.loss_kl.item(), global_steps)

        if epoch % 50 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "parameters_state_dict": act_policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": output.loss_total.item(),
                    "kl": output.loss_kl.item(),
                    "l2": output.loss_l2.item(),
                },
                f"{FLAGS.checkpoints_dir}/checkpoint_epoch_{epoch}.pth",
            )

            with torch.inference_mode():
                act_policy.eval()
                print("Validation: ")

                l2_err, kl, total_loss = 0.0, 0.0, 0.0
                num_validation_steps = 0
                for data in test_data_loader:
                    qpos, images, actions = data

                    output: ActOutput = act_policy(qpos, actions, images)
                    l2_err += output.loss_l2.item()
                    kl += output.loss_kl.item()
                    total_loss += output.loss_total.item()
                    num_validation_steps += 1

                print(f"Validation Loss: {l2_err / num_validation_steps}")
                print(f"Validation KL: {kl / num_validation_steps}")
                print(f"Validation L2: {total_loss / num_validation_steps}")
                writer.add_scalar(
                    "validation/loss_total", total_loss / num_validation_steps, epoch
                )
                writer.add_scalar(
                    "validation/loss_l2", l2_err / num_validation_steps, epoch
                )
                writer.add_scalar(
                    "validation/loss_kl", kl / num_validation_steps, epoch
                )

            act_policy.train()


if __name__ == "__main__":
    app.run(main)
