import torch
import h5py
import os
import numpy as np
import cv2 as cv

from typing import List


def read_h5_dataset(dataset_path: str, camera_device_names: List[str]):
    dataset_dict = {}
    with h5py.File(dataset_path, "r") as f:
        dataset_dict["qpos"] = f["/observations/qpos"][:]
        dataset_dict["action"] = f["/action"][:]
        for device_name in camera_device_names:
            image_dataset_path = f[f"/observations/images/{device_name}"].asstr()[...]
            cap = cv.VideoCapture(str(image_dataset_path))
            if not cap.isOpened():
                return None
            dataset_dict[f"/images/{device_name}"] = []
            ret, frame = cap.read()
            while ret:
                dataset_dict[f"/images/{device_name}"].append(frame)
                ret, frame = cap.read()

    return dataset_dict


def read_one_step_data(
    frame_index: int,
    action_chunk_size: int,
    camera_devices: List[int],
    h5_dataset_path: str,
):
    data = {}
    with h5py.File(h5_dataset_path, "r") as f:
        for camera_device in camera_devices:
            image_dataset_path = f[f"/observations/images/{camera_device}"].asstr()[...]
            cap = cv.VideoCapture(str(image_dataset_path))
            if not cap.isOpened():
                return None
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            data[f"/images/{camera_device}"] = frame

        data["qpos"] = f["/observations/qpos"][frame_index, ...]
        data["action"] = f["/action"][frame_index : frame_index + action_chunk_size]
    return data


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_episodes: int,
        camera_names: List[str],
        action_chunk_size: int = 100,
        random_sampling: bool = True,
        index_offset=0,
        device: str = "cuda:0",
    ):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.action_chunk_size = action_chunk_size
        self.num_episodes = num_episodes
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

        episode_path = os.path.join(self.dataset_dir, f"{index}.h5")
        with h5py.File(episode_path, "r") as f:
            episode_len = len(f["/observations/qpos"])
            if self.random_sampling:
                start_ts = np.random.choice(episode_len - self.action_chunk_size)
            else:
                start_ts = 0

            data = read_one_step_data(
                start_ts,
                self.action_chunk_size,
                self.camera_names,
                episode_path,
            )
            images = {}
            for camera_name in self.camera_names:
                images[camera_name] = (
                    torch.from_numpy(data[f"/images/{camera_name}"])
                    .to(self.device)
                    .float()
                    / 255.0
                )

        qpos = torch.from_numpy(data["qpos"]).to(self.device)
        action_data = torch.from_numpy(data["action"]).to(self.device)

        return qpos, images, action_data
