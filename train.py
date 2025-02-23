import torch
import argparse
import time

from dataset import EpisodicDataset
from models import (
    ActPolicy,
    l2_error,
    kl_divergence,
)
from model_config import model_config
from teleoperation_config import camera_device_names, teleoperation_config
from experimental.dataset import BatchStreamingQueue, BatchStreamingDataset


def execute_update_step(data, act_policy, optimizer, beta):
    qpos, images, actions = data

    pred_action, mu_z, logvar_z = act_policy(qpos, actions, images)

    # loss calculation
    l2_err = l2_error(pred_action, actions)
    kl = kl_divergence(mu_z, logvar_z)
    loss = l2_err + beta * kl

    # optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {"l2_err": l2_err, "kl": kl, "loss": loss}


def main(args):
    try:
        train_queue = None
        if args.dataset == "streaming":
            train_queue = BatchStreamingQueue(
                dataset_dir=args.train_dataset_dir,
                num_episodes=args.num_episodes_train,
                camera_names=camera_device_names,
                batch_size=args.batch_size,
                action_chunk_size=model_config["action_chunk_size"],
                episode_len=teleoperation_config["episode_len_time_steps"],
                data_queue_buffer_size=32,
                num_workers=4,
            )
            test_queue = BatchStreamingQueue(
                dataset_dir=args.test_dataset_dir,
                num_episodes=args.num_episodes_test,
                camera_names=camera_device_names,
                batch_size=args.batch_size,
                action_chunk_size=model_config["action_chunk_size"],
                episode_len=teleoperation_config["episode_len_time_steps"],
                data_queue_buffer_size=32,
                num_workers=4,
            )
            train_data_loader = BatchStreamingDataset(
                train_queue, args.num_episodes_train // args.batch_size
            )
            test_data_loader = BatchStreamingDataset(
                test_queue,
                1,
            )
        elif args.dataset == "episodic":
            train_dataset = EpisodicDataset(
                dataset_dir=args.train_dataset_dir,
                num_episodes=args.num_episodes_train,
                camera_names=camera_device_names,
            )
            train_data_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True
            )
            test_dataset = EpisodicDataset(
                dataset_dir=args.test_dataset_dir,
                num_episodes=args.num_episodes_test,
                camera_names=camera_device_names,
            )
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=True
            )
        else:
            raise ValueError(f"Invalid dataset type: {args.dataset}")

        beta = model_config["beta"]
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
        ).cuda()
        if args.initial_checkpoint is not None:
            checkpoint = torch.load(args.initial_checkpoint)
            act_policy.load_state_dict(checkpoint["parameters_state_dict"])
        act_policy.train()

        optimizer = torch.optim.AdamW(
            [
                {"params": act_policy.parameters()},
            ],
            lr=args.learning_rate,
        )
        for epoch in range(args.num_epochs):
            start_data_load = time.time()
            for data in train_data_loader:
                end_data_load = time.time()
                start_update = end_data_load
                result = execute_update_step(data, act_policy, optimizer, beta)
                end_update = time.time()

                print(
                    f"Loss: {result['l2_err'].item()} Elapsed time for model update: {end_update - start_update},"
                    f"Elapsed time for data load: {end_data_load - start_data_load}"
                )
                start_data_load = time.time()

            if epoch % 50 == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "parameters_state_dict": act_policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": result["loss"].item(),
                        "kl": result["kl"].item(),
                        "l2": result["l2_err"].item(),
                    },
                    f"{args.checkpoints_dir}/checkpoint_epoch_{epoch}.pth",
                )

                with torch.inference_mode():
                    act_policy.eval()
                    print("Validation: ")
                    l2_errs = []
                    kls = []
                    losses = []
                    for data in test_data_loader:
                        qpos, images, actions = data

                        pred_action, mu_z, logvar_z = act_policy(qpos, actions, images)
                        l2_err = l2_error(pred_action, actions).item()
                        kl = kl_divergence(mu_z, logvar_z).item()
                        l2_errs.append(l2_err)
                        kls.append(kl)
                        losses.append(l2_err + beta * kl)

                    print(f"Validation Loss: {sum(losses) / len(losses)}")
                    print(f"Validation KL: {sum(kls) / len(kls)}")
                    print(f"Validation L2: {sum(l2_errs) / len(l2_errs)}")

                act_policy.train()
    except Exception as e:
        if train_queue is not None:
            train_queue.close()
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--train_dataset_dir", type=str, default="./train_dataset")
    parser.add_argument("--test_dataset_dir", type=str, default="./test_dataset")
    parser.add_argument("--num_episodes_train", type=int, default=24)
    parser.add_argument("--num_episodes_test", type=int, default=5)
    parser.add_argument("--initial_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="streaming")
    args = parser.parse_args()
    main(args)
