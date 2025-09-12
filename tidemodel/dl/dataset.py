"""
数据集
"""

import itertools
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import torch
import numpy as np
from torch.utils.data import Dataset

from ..data import BinMinuteDataBase, HDF5MinuteDataBase, MinuteData, cross_norm


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


"""
数据集类
"""

class PreloadMinuteDataset(Dataset):
    """
    基于提前加载好的x和y来构建数据集

    尽量避免对矩阵有过多的操作, 原因如下:
    1. 矩阵占用内存高, 直接操作内存和耗时都会爆炸
    2. 在torch中使用gpu对小batch操作会更优
    """

    def __init__(
        self,
        whole_x: MinuteData,
        whole_y: MinuteData,
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        seq_len: int | None = None,
        cross_norm_y: bool = False,
    ) -> None:
        self.seq_len: int | None = seq_len
        self.x_names: List[str] = list(whole_x.names.data)
        self.y_names: List[str] = list(whole_y.names.data)

        self.whole_x: MinuteData = None
        self.whole_y: MinuteData = None
        self.y: MinuteData = None

        self._load_x_and_y(
            whole_x=whole_x,
            whole_y=whole_y,
            date_slice=date_slice,
            minute_slice=minute_slice,
            cross_norm_y=cross_norm_y,
        )

    def _load_x_and_y(
        self,
        whole_x: MinuteData,
        whole_y: MinuteData,
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        cross_norm_y: bool = False,
    ) -> None:
        """
        根据整个x和y矩阵计算y, y_pred和index
        """
        self.whole_x = whole_x
        self.whole_y = whole_y

        if cross_norm_y:
            self.whole_y = cross_norm(self.whole_y)

        self.y = self.whole_y[date_slice, minute_slice]

        date_idx_slice: slice = self.whole_y.dates.index_range(
            date_slice.start, date_slice.stop, date_slice.step
        )
        self.date_indexes = list(range(len(self.whole_y.dates)))[date_idx_slice]

        minute_idx_slice: slice = self.whole_y.minutes.index_range(
            minute_slice.start, minute_slice.stop, minute_slice.step
        )
        self.minute_indexes = list(range(len(self.whole_y.minutes)))[
            minute_idx_slice
        ]
        self.indexes: List[Tuple[int, int]] = list(
            itertools.product(self.date_indexes, self.minute_indexes)
        )

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx0: int = self.indexes[idx][0]
        idx1: int = self.indexes[idx][1]

        # 非时序数据返回数据是(n, ...)
        if self.seq_len is None:
            return {
                "x": self.whole_x.data[idx0, idx1],
                "y": self.whole_y.data[idx0, idx1],
                "idx": np.array([idx, ], np.int64),
            }

        # 时序数据返回数据是(t, n, ...)
        # 计算用到的indxes
        idxes: List[int] = list(range(
            max(idx1 - self.seq_len + 1, 0), idx1 + 1
        ))
        if len(idxes) < self.seq_len:
            n_pad: int = self.seq_len - len(idxes)
            idxes = list(range(-n_pad, 0, 1)) + idxes

        # 获取当日数据
        td_idxes: List[int] = [idx for idx in idxes if idx >= 0]
        x: np.ndarray = self.whole_x.data[idx0, td_idxes]

        # 不够则需要从昨日数据补齐
        yd_idxes: List[int] = [idx for idx in idxes if idx < 0]
        if len(yd_idxes) > 0:
            assert idx0 >= 1, f"can't get yesterday data"
            yd_x: np.ndarray = self.whole_x.data[idx0 - 1, yd_idxes]
            x = np.concatenate([yd_x, x], axis=0)

        return {
            "x": x,
            "y": self.whole_y.data[idx0, idx1],
            "idx": np.array([idx, ], np.int64),
        }


class LazyMinuteDataset(PreloadMinuteDataset):
    """
    基于HDF5分钟数据库构建数据集

    懒加载模式, 无需主动调用load, 在第一次访问时会自动load
    """

    def __init__(
        self,
        hdf5_db: HDF5MinuteDataBase,
        y_names: str | List[str],
        x_names: List[str] | slice = slice(None),
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        seq_len: int | None = None,
        cross_norm_y: bool = False,
    ) -> None:
        self.hdf5_db = hdf5_db
        self.x_names: List[str] | slice = (
            self.hdf5_db.names if x_names == slice(None)
            else x_names
        )
        self.y_names: str | List[str] = y_names
        self.date_slice: slice = date_slice
        self.minute_slice: slice = minute_slice
        self.tickers: List[str] | slice = tickers
        self.seq_len: int | None = seq_len
        self.cross_norm_y: bool = cross_norm_y

        self.whole_x: MinuteData = None
        self.whole_y: MinuteData = None
        self.y: MinuteData = None

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
            date_slice=self.date_slice,
            minute_slice=self.minute_slice,
            cross_norm_y=self.cross_norm_y,
        )

        # 提前计算股票名和因子名索引
        self.ticker_idx_slice = whole_x.tickers.index_list(self.tickers)
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
            data["x"] = data["x"][self.ticker_idx_slice, self.name_idx_slice]
        else:
            data["x"] = data["x"][:, self.ticker_idx_slice, self.name_idx_slice]
        return data


"""
一次性创建训练集, 验证集和测试集
"""

def get_datasets(
    dataset_cls: Type[Dataset],
    train_date_slice: slice = slice(None),
    valid_date_slice: slice = slice(None),
    test_date_slice: slice = slice(None),
    minute_slice: slice = slice(None),
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    根据参数获取训练集, 验证集和测试集
    """
    train_dataset: Dataset = dataset_cls(
        date_slice=train_date_slice,
        minute_slice=minute_slice,
    )
    valid_dataset: Dataset = dataset_cls(
        date_slice=valid_date_slice,
        minute_slice=minute_slice,
    )

    # 测试时步长必须为1
    test_dataset: Dataset = dataset_cls(
        date_slice=slice(
            test_date_slice.start,
            test_date_slice.stop,
            1,
        ),
        minute_slice=slice(
            minute_slice.start,
            minute_slice.stop,
            1,
        ),
    )
    return train_dataset, valid_dataset, test_dataset


def get_preload_datasets(
    bin_folder: str,
    y_names: str | List[str],
    hdf5_folder: str | None = None,
    x_names: List[str] | slice = slice(None),
    train_date_slice: slice = slice(None),
    valid_date_slice: slice = slice(None),
    test_date_slice: slice = slice(None),
    minute_slice: slice = slice(None),
    tickers: slice | List[str] = slice(None),
    seq_len: int | None = None,
    cross_norm_y: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    获取预加载训练集. 验证集和测试集
    """
    if hdf5_folder is None:
        assert isinstance(x_names, list)

        bin_db = BinMinuteDataBase(bin_folder)
        whole_x, whole_y = bin_db.read_dataset(
            x_names=x_names,
            y_names=y_names,
            tickers=tickers,
        )
    else:
        hdf5_db = HDF5MinuteDataBase(hdf5_folder, bin_folder)
        whole_x, whole_y = hdf5_db.read_dataset(
            x_names=x_names,
            y_names=y_names,
            tickers=tickers,
        )

    dataset_cls: Callable = partial(
        PreloadMinuteDataset,
        whole_x=whole_x,
        whole_y=whole_y,
        seq_len=seq_len,
        cross_norm_y=cross_norm_y,
    )
    return get_datasets(
        dataset_cls=dataset_cls,
        train_date_slice=train_date_slice,
        valid_date_slice=valid_date_slice,
        test_date_slice=test_date_slice,
        minute_slice=minute_slice,
    )


def get_lazy_datasets(
    bin_folder: str,
    hdf5_folder: str,
    y_names: str | List[str],
    x_names: List[str] | slice = slice(None),
    train_date_slice: slice = slice(None),
    valid_date_slice: slice = slice(None),
    test_date_slice: slice = slice(None),
    minute_slice: slice = slice(None),
    tickers: slice | List[str] = slice(None),
    seq_len: int | None = None,
    cross_norm_y: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    获取懒加载训练集. 验证集和测试集
    """

    hdf5_db = HDF5MinuteDataBase(
        hdf5_folder, bin_folder
    )
    dataset_cls: Callable = partial(
        LazyMinuteDataset,
        hdf5_db=hdf5_db,
        x_names=x_names,
        y_names=y_names,
        tickers=tickers,
        seq_len=seq_len,
        cross_norm_y=cross_norm_y,
    )
    return get_datasets(
        dataset_cls=dataset_cls,
        train_date_slice=train_date_slice,
        valid_date_slice=valid_date_slice,
        test_date_slice=test_date_slice,
        minute_slice=minute_slice,
    )
