import torch
import h5py
import os
import numpy as np

from typing import List


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        task_name: str,
        num_episods: int,
        camera_names: List[str],
    ):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.task_name = task_name
        self.num_episodes = num_episods
        self.camera_names = camera_names

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset with {len(self)} episodes"
            )

        episode_path = os.path.join(self.dataset_dir, f"{self.task_name}_{index}.h5")
        with h5py.File(episode_path, "r") as f:
            episode_len = len(f["/observations/qpos"])
            action_shape = f["/action"].shape
            start_ts = np.random.choice(episode_len)

            qpos = f["/observations/qpos"][start_ts]
            images = []
            for camera_name in self.camera_names:
                images.append(f[f"/observations/images/{camera_name}"][start_ts])
            action = f["/action"][start_ts:]

        padded_action = np.zeros(action_shape, dtype=np.float32)
        padded_action[: len(action)] = action
        is_pad = np.zeros(episode_len)
        is_pad[len(action) :] = 1

        qpos = torch.from_numpy(qpos).float()
        images = np.stack(images, axis=0)
        images = torch.from_numpy(images).float()
        action_data = torch.from_numpy(padded_action).float()

        return qpos, images, action_data, is_pad
