"""
模型数据集
"""

from typing import Any, Dict, List, Literal, Tuple, Union

import torch
import numpy as np
from torch.utils.data import Dataset

from ..data import MinuteData


def collate_fn(
    data_list: List[Dict[str, Any]],
) -> Dict[str, Union[torch.Tensor, List[Any]]]:
    """
    将列表中的字典的相同字段合并为一个字典:
    1. numpy矩阵会被合并成tensor
    2. 非numpy矩阵会被合并成列表
    """

    data: Dict[str, List[Any]] = {}
    for item in data_list:
        for k, v in item.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k in data:
        if not isinstance(data[k][0], np.ndarray):
            continue

        if data[k][0].dtype == np.bool_:
            dtype = torch.bool
        elif data[k][0].dtype == np.float32:
            dtype = torch.float32
        elif data[k][0].dtype == np.int64:
            dtype = torch.int64
        else:
            raise ValueError(f"{data[k][0].dtype} is not supported")

        data[k] = np.stack(data[k], axis=0)
        data[k] = torch.tensor(data[k], dtype=dtype) 
    return data


def load_to_device(
    data: Dict[str, Union[torch.Tensor, List[Any]]],
    device: str,
) -> None:
    """
    将tensor矩阵加载到指定设备中
    """
    for k in data:
        if isinstance(data[k], torch.Tensor):
            data[k] = data[k].to(device, non_blocking=True)


class Matrix4DDataset(Dataset):
    """
    从四维矩阵构建数据集
    x: (n_dt, n_step, n_ticker, n_x)
    y: (n_dt, n_step, n_ticker)

    尽量避免对矩阵有过多的操作, 原因如下:
    1. 矩阵占用内存高, 直接操作内存和耗时都会爆炸
    2. 在torch中使用gpu对小batch操作会更优
    """

    def __init__(
        self,
        data: MinuteData,
        slice_tuple: Tuple[slice],
    ) -> None:
        self.data: MinuteData = data

        self.mask: np.bool = np.zeros(, dtype=bool)

        self.dt_mask = self.x_loader.y_loader.get_dt_mask(
            mode=mode,
            open=False,
            slice=False,
        )
        self.indexes: np.ndarray = np.where(self.dt_mask)[0]
        self._init: bool = False

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        if not self._init:
            self.data.load()
            self._init = True

        data_idx: int = self.indexes[idx]
        return {
            "x": self.x_loader.x[data_idx],
            "y": self.x_loader.y_loader.y[data_idx],
            "idx": np.array([data_idx, ], np.int64),
        }
