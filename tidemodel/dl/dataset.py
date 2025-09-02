"""
数据集
"""

from typing import Any, Dict, List, Tuple, Union

import torch
import numpy as np
from torch.utils.data import Dataset

from ..data import HDF5MinuteDataBase, MinuteData, cross_norm


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


class BinMinuteDataset(Dataset):
    """
    基于二进制数据库构建数据集

    尽量避免对矩阵有过多的操作, 原因如下:
    1. 矩阵占用内存高, 直接操作内存和耗时都会爆炸
    2. 在torch中使用gpu对小batch操作会更优
    """

    def __init__(
        self,
        whole_x: MinuteData,
        whole_y: MinuteData,
        start_dt: np.datetime64 | None,
        end_dt: np.datetime64 | None,
        start_minute: int | None,
        end_minute: int | None,
        cross_norm_y: bool = False,
        seq_len: int | None = None,
    ) -> None:
        self._load_x_and_y(
            whole_x=whole_x,
            whole_y=whole_y,
            start_dt=start_dt,
            end_dt=end_dt,
            start_minute=start_minute,
            end_minute=end_minute,
            cross_norm_y=cross_norm_y,
            seq_len=seq_len,
        )

        self.seq_len: int | None = seq_len

    def _load_x_and_y(
        self,
        whole_x: MinuteData,
        whole_y: MinuteData,
        start_dt: np.datetime64 | None,
        end_dt: np.datetime64 | None,
        start_minute: int | None,
        end_minute: int | None,
        cross_norm_y: bool = False,
    ) -> None:
        """
        根据整个x和y矩阵计算y, y_pred和index
        """
        self.whole_x: MinuteData = whole_x
        self.whole_y: MinuteData = whole_y
        if cross_norm_y:
            self.whole_y = cross_norm(self.whole_y)

        self.y: MinuteData = self.whole_y[
            slice(start_dt, end_dt),
            slice(start_minute, end_minute)
        ]
        self.y_pred: MinuteData = MinuteData(
            dates=self.y.dates.data,
            minutes=self.y.minutes.data,
            tickers=self.y.tickers.data,
            names=self.y.names.data,
        )

        dt_slice: slice = self.whole_y.dates.index_range(
            start_dt, end_dt
        )
        self.dt_indexes = list(range(len(self.whole_y.dates)))[dt_slice]

        minute_slice: slice = self.whole_y.minutes.index_range(
            start_minute, end_minute
        )
        self.minute_indexes = list(range(len(
            self.whole_y.minutes
        )))[minute_slice]
    
    def __len__(self) -> int:
        return len(self.dt_indexes) * len(self.minute_indexes)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx0: int = self.dt_indexes[idx // len(self.minute_indexes)]
        idx1: int = self.minute_indexes[idx % len(self.minute_indexes)]

        # 非时序数据返回数据是(n, ...)
        if self.seq_len is None:
            return {
                "x": self.whole_x.data[idx0, idx1],
                "y": self.whole_y.data[idx0, idx1],
                "idx": np.array([idx, ], np.int64),
                "minute": np.array([idx1, ], np.int64),
            }

        # 时序数据返回数据是(t, n, ...)
        x: np.ndarray = self.whole_x.data[
            idx0, max(idx1 - self.seq_len, 0.0): idx1
        ]
        y: np.ndarray = self.whole_y.data[
            idx0, max(idx1 - self.seq_len, 0.0): idx1
        ]

        # 如果不够则需要从前一日补齐
        if x.shape[0] < self.seq_len:
            assert idx0 >= 1, "can't get yesterday sequence data"

            n_pad: int = self.seq_len - x.shape[0]
            pad_x: np.ndarray = self.whole_x.data[idx0 - 1, -n_pad: ]
            pad_y: np.ndarray = self.whole_y.data[idx0 - 1, -n_pad: ]

        return {
            "x": np.concatenate([pad_x, x], axis=0),
            "y": np.concatenate([pad_y, y], axis=0),
            "idx": np.array([idx, ], np.int64),
        }


class HDF5MinuteDataset(BinMinuteDataset):
    """
    基于HDF5分钟数据库构建数据集

    无需主动调用load, 在第一次访问时会自动load
    """

    def __init__(
        self,
        hdf5_db: HDF5MinuteDataBase,
        y_names: str | List[str],
        x_names: List[str] | slice = slice(None),
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
        start_minute: int | None = None,
        end_minute: int | None = None,
        cross_norm_y: bool = False,
        seq_len: int | None = None,
    ) -> None:
        self.hdf5_db = hdf5_db
        self.x_names: List[str] | slice = x_names
        self.y_names: str | List[str] = y_names
        self.start_dt: np.datetime64 | None = start_dt
        self.end_dt: np.datetime64 | None = end_dt
        self.start_minute: int | None = start_minute
        self.end_minute: int | None = end_minute
        self.cross_norm_y: bool = cross_norm_y
        self.seq_len: int | None = seq_len

        # 数据, y常用于评估
        self.whole_x: MinuteData = None
        self.whole_y: MinuteData = None
        self.y: MinuteData = None
        self.dt_indexes: List[int] = None
        self.minute_indexes: List[int] = None

        self._load: bool = False

    def load(self, ) -> None:
        """
        初始化hdf5句柄, 加载数据
        """
        whole_x, whole_y = self.hdf5_db.read_dataset_lazy(
            self.y_names
        )
        
        self._load_x_and_y(
            whole_x=whole_x,
            whole_y=whole_y,
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            start_minute=self.start_minute,
            end_minute=self.end_minute,
            cross_norm_y=self.cross_norm_y,
        )

        # 提前计算因子名索引
        if self.x_names == slice(None):
            self.name_idx_slice: slice | Tuple[int] = slice(None)
        else:
            self.name_idx_slice = whole_x.names.index_list(self.x_names)

        self._load = True

    def __len__(self) -> int:
        if not self._load:
            self.load()

        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        if not self._load:
            self.load()

        data: Dict[str, np.ndarray] = super().__getitem__(idx)

        # 根据选择的因子名对数据索引
        if self.seq_len is None:
            data["x"] = data["x"][:, self.name_idx_slice]
        else:
            data["x"] = data["x"][:, :, self.name_idx_slice]
        return data
