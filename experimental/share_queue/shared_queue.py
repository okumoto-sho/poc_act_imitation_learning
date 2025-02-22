import numpy as np
import dataclasses
import time

from typing import Tuple, Dict, List
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import shared_memory, Lock


@dataclasses.dataclass
class EntryType:
    dtype: np.uint8 | np.int32 | np.float32
    shape: Tuple[int] | Tuple[int, int] | Tuple[int, int, int]

    def __init__(self, dtype, shape):
        if dtype not in [np.uint8, np.int32, np.float32]:
            raise ValueError(f"Invalid dtype: {dtype}")
        if len(shape) not in [1, 2, 3]:
            raise ValueError(f"Invalid shape: {shape}")
        if not all([elem > 0 for elem in shape]):
            raise ValueError(
                f"Invalid shape {shape}: elements of shape must be greater than 0"
            )
        self.dtype = dtype
        self.shape = shape

    def entry_size(self):
        """
        Returns the size of the entry in bytes.
        """
        if len(self.shape) == 1:
            entry_count = self.shape[0]
        elif len(self.shape) == 2:
            entry_count = self.shape[0] * self.shape[1]
        elif len(self.shape) == 3:
            entry_count = self.shape[0] * self.shape[1] * self.shape[2]

        if self.dtype == np.uint8:
            return entry_count * 1
        elif self.dtype == np.int32:
            return entry_count * 4
        elif self.dtype == np.float32:
            return entry_count * 4


class SharedIndexQueue:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.mem_manager = SharedMemoryManager()
        self.mem_manager.start()
        self.forward_chain = self.mem_manager.ShareableList(
            [-1 for _ in range(capacity + 2)]
        )
        self.first_sentinel = capacity
        self.last_sentinel = capacity + 1
        self.forward_chain[self.first_sentinel] = self.last_sentinel
        self.lock = Lock()

    def _unsafe_is_empty(self):
        return self._unsafe_count() == 0

    def _unsafe_get_last_index(self):
        cur = self.first_sentinel
        while self.forward_chain[cur] != self.last_sentinel:
            cur = self.forward_chain[cur]
        return cur

    def _unsafe_get_first_index(self):
        return self.forward_chain[self.first_sentinel]

    def _unsafe_get_available_index(self):
        for i in range(self.capacity):
            if self.forward_chain[i] == -1:
                return i
        return None

    def _unsafe_contains(self, index):
        cur = self.first_sentinel
        while cur != self.last_sentinel:
            cur = self.forward_chain[cur]
            if cur == index:
                return True
        return False

    def is_empty(self):
        with self.lock:
            return self._unsafe_is_empty()

    def is_full(self):
        with self.lock:
            return self._unsafe_count() == self.capacity

    def pop(self):
        index = self.try_pop()
        if index is None:
            raise ValueError("Failed to pop index")
        return index

    def try_pop_some(
        self, num: int, timeout: float = 2.0, polling_interval: float = 0.002
    ):
        start = time.time()
        indexes = []
        while len(indexes) < num and time.time() - start < timeout:
            with self.lock:
                indexes = self._unsafe_try_pop_some(num)
            time.sleep(polling_interval)
        return indexes

    def try_pop(self):
        with self.lock:
            return self._unsafe_try_pop()

    def _unsafe_try_pop_some(self, num: int):
        if self._unsafe_count() < num:
            return []
        indexes = []
        for _ in range(num):
            cur = self._unsafe_try_pop()
            assert cur is not None
            indexes.append(cur)
        return indexes

    def _unsafe_try_pop(self):
        if self._unsafe_is_empty():
            return None
        first = self._unsafe_get_first_index()
        self.forward_chain[self.first_sentinel] = self.forward_chain[first]
        return first

    def contains(self, index):
        with self.lock:
            return self._unsafe_contains(index)

    def push(self, index):
        if not self.try_push(index):
            raise ValueError(f"Failed to push index {index}")

    def push_some(self, indexes: List[int]):
        if not self.try_push_some(indexes):
            raise ValueError(f"Failed to push indexes {indexes}")

    def try_push(self, index):
        with self.lock:
            return self._unsafe_try_push(index)

    def try_push_some(self, indexes: List[int]):
        with self.lock:
            for index in indexes:
                if not self._unsafe_try_push(index):
                    return False
            return True

    def _unsafe_try_push(self, index):
        if self._unsafe_count() == self.capacity:
            return False
        last = self._unsafe_get_last_index()
        self.forward_chain[last] = index
        self.forward_chain[index] = self.last_sentinel
        return True

    def contained_indexes(self):
        with self.lock:
            indexes = []
            cur = self.first_sentinel
            while self.forward_chain[cur] != self.last_sentinel:
                cur = self.forward_chain[cur]
                indexes.append(cur)
            return indexes

    def _unsafe_count(self):
        cur = self.first_sentinel
        count = 0
        while self.forward_chain[cur] != self.last_sentinel:
            cur = self.forward_chain[cur]
            count += 1
        return count

    def count(self):
        with self.lock:
            return self._unsafe_count()

    def close(self):
        self.mem_manager.shutdown()
        del self.mem_manager


class SharedQueue:
    """
    A queue implementation using shared memory.
    """

    MEM_INDEX_CHAIN_FIRST = 0

    def __init__(self, structure: Dict[str, EntryType], capacity: int):
        self.shared_memories_manager = SharedMemoryManager()
        self.shared_memories_manager.start()
        self.available_index_queue = SharedIndexQueue(capacity)
        for i in range(capacity):
            self.available_index_queue.push(i)

        self.occupied_index_queue = SharedIndexQueue(capacity)

        self.capacity_ = capacity
        self.shared_memories: Dict[str, shared_memory.SharedMemory] = {}
        self.structure = structure
        for key, value in structure.items():
            self.shared_memories[key] = self.shared_memories_manager.SharedMemory(
                value.entry_size() * capacity
            )

    def try_put(self, data):
        index = self.available_index_queue.try_pop()
        if index is None:
            return False

        try:
            for key, value in data.items():
                shape = self.structure[key].shape
                dtype = self.structure[key].dtype
                entry_size = self.structure[key].entry_size()
                if value.shape != shape:
                    raise ValueError(
                        f"Shape mismatch for key {key}: expected {shape}, got {value.shape}"
                    )
                if value.dtype != dtype:
                    raise ValueError(
                        f"Dtype mismatch for key {key}: expected {dtype}, got {value.dtype}"
                    )
                self.shared_memories[key].buf[
                    index * entry_size : (index + 1) * entry_size
                ] = value.flatten().tobytes()
        except Exception as e:
            # If an exception occurs, we need to push the index back to the available queue
            self.available_index_queue.push(index)
            raise e

        self.occupied_index_queue.push(index)
        return True

    def _get_with_index(self, index):
        ret_dict = {}
        for key, value in self.shared_memories.items():
            shape = self.structure[key].shape
            dtype = self.structure[key].dtype
            entry_size = self.structure[key].entry_size()
            ret_dict[key] = np.ndarray(
                shape=shape,
                dtype=dtype,
                buffer=value.buf[index * entry_size : (index + 1) * entry_size],
            ).copy()
        return ret_dict

    def try_get(self):
        index = self.occupied_index_queue.try_pop()
        if index is None:
            return None

        try:
            ret_dict = self._get_with_index(index)
        except Exception as e:
            # If an exception occurs, we need to push the index back to the available queue
            self.occupied_index_queue.push(index)
            raise e

        self.available_index_queue.push(index)
        return ret_dict

    def try_get_some(self, num: int, timeout: int = 2, polling_interval: float = 0.002):
        ret_dict_batch = {}
        for key in self.shared_memories.keys():
            batched_shape = (num, *self.structure[key].shape)
            ret_dict_batch[key] = np.zeros(
                shape=batched_shape, dtype=self.structure[key].dtype
            )

        indexes = self.occupied_index_queue.try_pop_some(num, timeout, polling_interval)
        if len(indexes) != num:
            return None

        try:
            for key, value in self.shared_memories.items():
                for ret_index, buffer_index in enumerate(indexes):
                    shape = self.structure[key].shape
                    dtype = self.structure[key].dtype
                    entry_size = self.structure[key].entry_size()
                    ret_dict_batch[key][ret_index, ...] = np.ndarray(
                        shape=shape,
                        dtype=dtype,
                        buffer=value.buf[
                            buffer_index * entry_size : (buffer_index + 1) * entry_size
                        ],
                    )
        except Exception as e:
            print("Exception occured in SharedQueue.try_get_some. Executing roll back.")
            self.occupied_index_queue.push_some(indexes)
            raise e

        self.available_index_queue.push_some(indexes)
        return ret_dict_batch

    def is_full(self):
        return self.available_index_queue.is_empty()

    def is_empty(self):
        return self.available_index_queue.is_full()

    @property
    def available_size(self):
        return self.available_index_queue.count()

    @property
    def occupied_size(self):
        return self.occupied_index_queue.count()

    @property
    def capacity(self):
        return self.capacity_

    def close(self):
        self.shared_memories_manager.shutdown()
        self.available_index_queue.close()
        self.occupied_index_queue.close()
        del self.shared_memories_manager
