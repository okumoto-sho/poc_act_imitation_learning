import torch
import h5py
import os
import numpy as np
import cv2 as cv

from typing import List, Dict, Optional


def read_full_steps_data(
    dataset_dir: str,
    episode_id: int,
    camera_devices: List[int],
    image_size=(480, 640, 3),
):
    data = {}
    with h5py.File(f"{dataset_dir}/{episode_id}.h5", "r") as f:
        n_timesteps = f["/action"].shape[0]
        data["qpos"] = f["/observations/qpos"][...]
        data["action"] = f["/action"][...]
        for camera_device in camera_devices:
            data[f"/images/{camera_device}"] = np.zeros(
                (n_timesteps, image_size[0], image_size[1], image_size[2]),
                dtype=np.uint8,
            )
            relative_images_path = f[f"/observations/images/{camera_device}"].asstr()[0]
            cap = cv.VideoCapture(f"{dataset_dir}/{relative_images_path}")
            if not cap.isOpened():
                raise RuntimeError(
                    f"Error: Cannot open video file {dataset_dir}/{relative_images_path}"
                )
            for step in range(n_timesteps):
                _, frame = cap.read()
                data[f"/images/{camera_device}"][step, :] = cv.resize(
                    frame, (image_size[1], image_size[0])
                )
            cap.release()
    return data


def read_one_step_data(
    dataset_dir: str,
    episode_id: int,
    frame_index: int,
    camera_devices: List[int],
    action_chunk_size: Optional[int] = None,
    image_size=(480, 640, 3),
):
    data = {}
    h5_dataset_path = f"{dataset_dir}/{episode_id}.h5"

    with h5py.File(h5_dataset_path, "r") as f:
        for camera_device in camera_devices:
            relative_images_path = f[f"/observations/images/{camera_device}"].asstr()[0]
            cap = cv.VideoCapture(f"{dataset_dir}/{relative_images_path}")
            if not cap.isOpened():
                return None
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            data[f"/images/{camera_device}"] = cv.resize(
                frame, (image_size[1], image_size[0])
            )

        data["qpos"] = f["/observations/qpos"][frame_index, ...]
        if action_chunk_size is not None:
            data["action"] = f["/action"][frame_index : frame_index + action_chunk_size]
        else:
            data["action"] = f["/action"][frame_index:]
    return data


def save_one_episode_data(
    dataset_dir: str,
    episode_id: int,
    qpos_data: np.ndarray,
    action_data: np.ndarray,
    images_data: Dict[str, np.ndarray],
    control_cycle: float,
):
    """
    Args:
        dataset_dir (str): Directory to save the dataset.
        episode_id (int): Episode ID.
        qpos_data (np.ndarray): Qpos data with shape `(time_steps, DoF_robot_arms)`.
        action_data (np.ndarray): Action data with shape `(time_steps, DoF_robot_arms)`.
        images_data (Dict[str, np.ndarray]):
        Images data with shape `{<camera_device_id> : np.ndarray(time_steps, height, width, n_channels)}`.
    """
    n_timesteps, dof_robot_arms = qpos_data.shape[0], qpos_data.shape[1]

    # Execute validation for checking the shape consistency of given data.
    if n_timesteps != action_data.shape[0]:
        raise ValueError(
            f"Time steps of qpos data {n_timesteps} and action data {action_data.shape[0]} do not match."
        )

    for camera_id, images in images_data.items():
        if n_timesteps != images.shape[0]:
            raise ValueError(
                f"Time steps of qpos data {n_timesteps} and images data {images.shape[0]} of camera id {camera_id} do"
                "not match."
            )

    if dof_robot_arms != action_data.shape[1]:
        raise ValueError(
            f"DoF of qpos data {dof_robot_arms} and action data {action_data.shape[1]} do not match."
        )

    with h5py.File(f"{dataset_dir}/{episode_id}.h5", "w") as f:
        f["/observations/qpos"] = qpos_data
        f["/action"] = action_data
        for camera_id, images in images_data.items():
            f[f"/observations/images/{camera_id}"] = [
                f"images/{episode_id}_{camera_id}.mp4"
            ]

            width, height = images.shape[2], images.shape[1]
            out = cv.VideoWriter(
                f"{dataset_dir}/images/{episode_id}_{camera_id}.mp4",
                cv.VideoWriter_fourcc(*"mp4v"),
                1.0 / control_cycle,
                (width, height),
            )
            if not out.isOpened():
                raise ValueError(f"Error: Cannot open {episode_id}_{camera_id}.mp4")

            for step in range(n_timesteps):
                frame = images[step, :, :, :]
                out.write(frame)
            out.release()


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
        image_size=(480, 640, 3),
    ):
        super(EpisodicDataset).__init__()
        self.dataset_dir = dataset_dir
        self.action_chunk_size = action_chunk_size
        self.num_episodes = num_episodes
        self.camera_names = camera_names
        self.random_sampling = random_sampling
        self.device = device
        self.index_offset = index_offset
        self.image_size = image_size

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
                dataset_dir=self.dataset_dir,
                episode_id=index,
                frame_index=start_ts,
                camera_devices=self.camera_names,
                action_chunk_size=self.action_chunk_size,
                image_size=self.image_size,
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
