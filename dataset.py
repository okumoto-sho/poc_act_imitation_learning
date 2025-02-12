import torch
import h5py
import os
import numpy as np

from typing import List


def read_h5_dataset(dataset_path: str, camera_device_names: List[str]):
    dataset_dict = {}
    with h5py.File(dataset_path, "r") as f:
        dataset_dict["qpos"] = f["/observations/qpos"][:]
        for device_name in camera_device_names:
            dataset_dict[f"images/{device_name}"] = f[
                f"/observations/images/{device_name}"
            ][:]
        dataset_dict["action"] = f["/action"][:]
    return dataset_dict


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        task_name: str,
        num_episods: int,
        camera_names: List[str],
        action_chunk_size: int = 100,
        random_sampling: bool = True,
        index_offset=0,
        device: str = "cuda:0",
    ):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.action_chunk_size = action_chunk_size
        self.task_name = task_name
        self.num_episodes = num_episods
        self.camera_names = camera_names
        self.random_sampling = random_sampling
        self.device = device
        self.index_offset = index_offset

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset with {len(self)} episodes"
            )

        episode_path = os.path.join(
            self.dataset_dir, f"{self.task_name}_{index + self.index_offset}.h5"
        )
        with h5py.File(episode_path, "r") as f:
            episode_len = len(f["/observations/qpos"])
            if self.random_sampling:
                start_ts = np.random.choice(episode_len - self.action_chunk_size)
            else:
                start_ts = 0

            qpos = f["/observations/qpos"][start_ts]
            images = {}
            for camera_name in self.camera_names:
                images[camera_name] = (
                    torch.from_numpy(
                        f[f"/observations/images/{camera_name}"][start_ts] / 255.0
                    )
                    .float()
                    .to(self.device)
                )
            action = f["/action"][start_ts : self.action_chunk_size + start_ts]

        qpos = torch.from_numpy(qpos).float().to(self.device)
        action_data = torch.from_numpy(action).float().to(self.device)

        return qpos, images, action_data
