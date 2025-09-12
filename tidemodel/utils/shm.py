"""
共享内存相关工具
"""


from typing import Tuple

import mmap
import numpy as np
import posix_ipc


class ShmNdarray:
    """
    基于共享内存的float32类型的数组
    """

    def __init__(self, name: str, shape: Tuple[int]) -> None:
        self.name: str = name
        self.shape: Tuple[int, ...] = shape
        self.n_bytes: int = int(
            np.prod(self.shape) * np.dtype(np.float32).itemsize
        )
        
        self.buf: mmap.mmap = None
        self.data: np.ndarray = None

    def load(self, ) -> np.ndarray:
        shm = posix_ipc.SharedMemory(name=f"/{self.name}")
        self.buf = mmap.mmap(shm.fd, self.n_bytes, mmap.MAP_SHARED, mmap.PROT_READ)
        shm.close_fd()
        
        self.data = np.frombuffer(self.buf, dtype=np.float32).reshape(self.shape)

    def save(self, data: np.ndarray) -> None:
        shm = posix_ipc.SharedMemory(
            name=f"/{self.name}",
            flags=posix_ipc.O_CREX,
            size=self.n_bytes,
        )
        self.buf = mmap.mmap(shm.fd, self.n_bytes, mmap.MAP_SHARED, mmap.PROT_WRITE)
        shm.close_fd()

        target = np.ndarray(self.shape, dtype=np.float32, buffer=self.buf)
        target[...] = data 
        self.buf.flush()

    def close(self) -> None:
        if self.buf is not None:
            self.buf.close()
            self.buf = None

        try:
            posix_ipc.unlink_shared_memory(f"/{self.name}")
        except posix_ipc.ExistentialError:
            pass
    