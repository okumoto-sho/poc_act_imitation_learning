import torch
import numpy as np

from typing import List, Tuple, Dict

import torch.utils.data.dataset
from dataset import read_one_step_data
from experimental.shared_queue.shared_queue import SharedQueue, EntryType
from multiprocessing import Event, Process


def stream_image(
    data_queue: SharedQueue,
    stop_event,
    episode_len: int,
    action_chunk_size: int,
    camera_devices: List[int],
    h5_dataset_path_candidates: List[str],
    image_size: Tuple[int, int, int],
):
    while not stop_event.is_set():
        h5_dataset_path = np.random.choice(h5_dataset_path_candidates)
        frame_index = np.random.randint(0, episode_len)
        data = read_one_step_data(
            frame_index, action_chunk_size, camera_devices, h5_dataset_path, image_size
        )
        data_queue.try_put(data)


class BatchStreamingQueue:
    def __init__(
        self,
        dataset_dir: str,
        num_episodes: int,
        camera_names: List[str],
        batch_size: int = 8,
        action_chunk_size: int = 100,
        image_size=(480, 640, 3),
        episode_len: int = 500,
        index_offset=0,
        device: str = "cuda:0",
        data_queue_buffer_size=32,
        num_workers=4,
    ):
        if num_workers > num_episodes:
            raise ValueError("num_workers should not be greater than num_episodes")
        if len(camera_names) == 0:
            raise ValueError("camera_names should not be empty")
        if action_chunk_size > episode_len:
            raise ValueError("action_chunk_size should be less than episode_len")
        if batch_size > data_queue_buffer_size:
            raise ValueError("batch_size should be less than data_queue_buffer")

        self.dataset_dir = dataset_dir
        self.action_chunk_size = action_chunk_size
        self.num_episodes = num_episodes
        self.episode_len = episode_len
        self.camera_names = camera_names
        self.device = device
        self.index_offset = index_offset
        self.batch_size = batch_size
        self.image_size = image_size
        self.queue_data_structure = {
            f"/images/{camera_name}": EntryType(np.uint8, image_size)
            for camera_name in camera_names
        }
        self.queue_data_structure["qpos"] = EntryType(np.float32, (6,))
        self.queue_data_structure["action"] = EntryType(
            np.float32, (action_chunk_size, 6)
        )
        self.data_queue = SharedQueue(self.queue_data_structure, data_queue_buffer_size)
        self.stop_event = Event()

        self.workers = []
        covered_indexes_per_worker = max(num_episodes // num_workers, 1)
        for worker_index in range(num_workers):
            p = Process(
                target=stream_image,
                args=(
                    self.data_queue,
                    self.stop_event,
                    self.episode_len - self.action_chunk_size,
                    action_chunk_size,
                    camera_names,
                    [
                        f"{dataset_dir}/{index + index_offset}.h5"
                        for index in range(
                            worker_index * covered_indexes_per_worker,
                            (worker_index + 1) * covered_indexes_per_worker,
                        )
                    ],
                    self.image_size,
                ),
            )
            p.start()
            self.workers.append(p)

    def pop(self) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        data = self.data_queue.try_get_some(self.batch_size)
        if data is None:
            raise ValueError("Faile to get data from the queue")

        images = {}
        for camera_name in self.camera_names:
            images[camera_name] = (
                torch.from_numpy(data[f"/images/{camera_name}"]).to(self.device).float()
                / 255.0
            )

        qpos = torch.from_numpy(data["qpos"]).to(self.device)
        action_data = torch.from_numpy(data["action"]).to(self.device)

        return qpos, images, action_data

    def close(self):
        for worker in self.workers:
            worker.terminate()
        self.data_queue.close()


class BatchStreamingDataset:
    def __init__(self, streaming_queue: BatchStreamingQueue, iter_num: int):
        self.streaming_queue = streaming_queue
        self.cur_iter_num = 0
        self.iter_num = iter_num

    def __iter__(self):
        self.cur_iter_num = 0
        return self

    def __next__(self):
        if self.cur_iter_num >= self.iter_num:
            raise StopIteration
        self.cur_iter_num += 1
        return self.streaming_queue.pop()
