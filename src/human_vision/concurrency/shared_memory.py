from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import numpy as np


@dataclass
class SharedFrame:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def to_numpy(self) -> np.ndarray:
        shm = shared_memory.SharedMemory(name=self.name)
        array = np.ndarray(self.shape, dtype=self.dtype, buffer=shm.buf)
        return array


def create_shared_frame(array: np.ndarray) -> SharedFrame:
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    dest = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    dest[:] = array
    return SharedFrame(name=shm.name, shape=array.shape, dtype=str(array.dtype))
