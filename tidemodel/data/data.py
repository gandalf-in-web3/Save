"""
分钟数据类
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import h5py
import numpy as np
import numpy.typing as npt


class Index(ABC):
    """
    由一维numpy数组构成的索引基类
    """

    def __init__(self, data: np.ndarray) -> None:
        assert isinstance(data, np.ndarray)
        assert data.ndim == 1

        self.data: np.ndarray = data

    def __len__(self, ) -> int:
        return len(self.data)

    @abstractmethod
    def index(self, value: Any) -> int:
        """
        获取值所在的位置

        若不存在会抛出ValueError
        """
        pass

    def index_list(self, values: List[Any]) -> List[int] | slice:
        """
        获取一组值所在的位置
        
        若有不存在的值会抛出ValueError

        若所在位置是连续的, 返回slice以避免拷贝
        """
        assert isinstance(values, list)
        assert len(values) > 0

        indexes: List[int] = [self.index(value) for value in values]

        if len(indexes) == 1 or all(
            indexes[i + 1] - indexes[i] == 1
            for i in range(len(indexes) - 1)
        ):
            return slice(indexes[0], indexes[-1] + 1)
        return indexes
    
    @abstractmethod
    def index_range(self, start_value: Any, end_value: Any, step: int = 1) -> slice:
        """
        获取起止值所对应的slice区间

        step值步长, 无单位
        """
        pass


class SortedIndex(Index):
    """
    有序索引, 要求不严格升序
    """

    def index(self, value: Any) -> int:
        idx: int = np.searchsorted(self.data, value, side="left")
        if self.data[idx] != value:
            raise ValueError(f"{value} is not in index")
        return idx

    def index_range(self, start_value: Any, end_value: Any, step: int = 1) -> slice:
        start: int = 0
        if start_value is not None:
            start = np.searchsorted(self.data, start_value, side="left")

        stop: int = len(self.data)
        if end_value is not None:
            stop = np.searchsorted(self.data, end_value, side="left")
        return slice(start, stop, step)


class UniqueIndex(Index):
    """
    无序索引, 要求每个数值唯一
    """

    def index(self, value: Any) -> int:
        try:
            return np.where(self.data == value)[0][0]
        except:
            raise ValueError(f"{value} is not in index")

    def index_range(self, start_value: Any, end_value: Any, step: int = 1) -> slice:
        raise NotImplemented(f"UniqueIndex doesn't support range")


class MinuteData:
    """
    分钟频数据类

    由(dates, minutes, tickers, names)构成的四维索引和四维矩阵构成
    """

    def __init__(
        self,
        dates: np.ndarray,
        minutes: np.ndarray,
        tickers: np.ndarray,
        names: np.ndarray,
        data: npt.NDArray | None = None,
    ) -> None:
        self.dates: Index = SortedIndex(dates)
        self.minutes: Index = SortedIndex(minutes)
        self.tickers: Index = SortedIndex(tickers)
        self.names: Index = SortedIndex(names)
        self.shape: Tuple[int] = (
            len(dates), len(minutes), len(tickers), len(names)
        )

        if data is None:
            self.data: np.ndarray | HDF5Ndarray = np.empty(
                self.shape, dtype=np.float32
            )
        else:
            self.data = data
            assert self.data.shape == self.shape

    def __getitem__(self, value_slices: slice | Tuple[slice]) -> "MinuteData":
        """
        接受四个维度的值索引并返回MinuteData

        注意索引不能改变数据的维度, 前两维的索引不触发拷贝

        接受的值索引格式为:
        1. slice(None) 或 值索引slice(start_dt, end_dt, step) step指个数
        2. slice(None) 或 值索引slice(start_minute, end_minute, step) step指个数
        3. slice(None) 或 值索引列表[ticker1, ticker2...]
        4. slice(None) 或 值索引列表[name1, name2...]
        """
        if isinstance(value_slices, slice):
            value_slices = (value_slices, )
        assert len(value_slices) <= 4
        value_slices = (
            *value_slices,
            *[slice(None) for _ in range(4 - len(value_slices))]
        )

        date_idx_slice: slice = self.dates.index_range(
            value_slices[0].start, value_slices[0].stop, value_slices[0].step,
        )
        minute_idx_slice: slice = self.minutes.index_range(
            value_slices[1].start, value_slices[1].stop, value_slices[1].step,
        )
        ticker_idx_slice: List[int] | slice = (
            slice(None) if value_slices[2] == slice(None)
            else self.tickers.index_list(value_slices[2])
        )
        name_idx_slice: List[int] | slice = (
            slice(None) if value_slices[3] == slice(None)
            else self.names.index_list(value_slices[3])
        )
        return MinuteData(
            dates=self.dates.data[date_idx_slice],
            minutes=self.minutes.data[minute_idx_slice],
            tickers=self.tickers.data[ticker_idx_slice],
            names=self.names.data[name_idx_slice],
            data=self.data[
                date_idx_slice,
                minute_idx_slice,
                ticker_idx_slice,
                name_idx_slice
            ],
        )


class HDF5Ndarray:
    """
    由一组已经初始化好的h5py dataset构成的矩阵

    和numpy矩阵一样支持索引返回numpy矩阵
    """

    def __init__(self, datasets: List[h5py.Dataset]) -> None:
        self.datasets: List[h5py.Dataset] = datasets

        # 每一个dataset的shape都是(1, n_minute, n_ticker, n_name)
        self.shape: Tuple[int] = (
            len(self.datasets), 
        ) + self.datasets[0].shape[1: ]

    def __getitem__(self, slices: Any) -> np.ndarray | float:
        """
        和numpy矩阵完全一致的索引方法
        """
        if isinstance(slices[0], int):
            return self.datasets[slices[0]][0, *slices[1: ]]
        else:
            datasets: List[h5py.Dataset] = self.datasets[slices[0]]
            data: List[float | np.ndarray] = [
                dataset[0, *slices[1: ]] for dataset in datasets
            ]
            return np.stack(data, axis=0)
