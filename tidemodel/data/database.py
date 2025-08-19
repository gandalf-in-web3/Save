"""
保存和读取分钟数据
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

import numpy as np
from tqdm.auto import tqdm

from .data import MinuteData
from .ops import cross_norm
from ..utils import validate_float_to_int


class BinMinuteDataBase:
    """
    以bin文件为核心的分钟数据库

    数据库格式如下:
    datetime.bin: 时间戳
    ticker.bin: 品种名
    x/*: 一个bin文件对应一个因子(n_date, n_minute, n_ticker)
    y/*: 一个bin文件对应一个标签(n_date, n_minute, n_ticker)
    """

    def __init__(self, folder: str) -> None:
        self.folder: str = folder

        # 读取ns时间戳, UTC时区转换为东八区需要加八小时
        timestamps: np.ndarray = np.fromfile(
            os.path.join(self.folder, "datetime.bin"), dtype=np.int64
        ) // (10**9)
        dts: np.ndarray = timestamps.astype("datetime64[s]").astype(
            "datetime64[m]"
        ) + np.timedelta64(8, 'h')

        self.freq: int = validate_float_to_int(
            (dts[1] - dts[0]) / np.timedelta64(1, 'm')
        )
        self.minutes: np.ndarray = np.arange(0, 240 + 1, self.freq)
        self.dates: np.ndarray = dts[::len(self.minutes)].astype("datetime64[D]")

        # 读取品种名
        self.tickers: np.ndarray = np.loadtxt(
            os.path.join(self.folder, "ticker.bin"), dtype=str
        )

    def list_x_names(self, ) -> List[str]:
        """
        获取所有可用因子名
        """
        names: List[str] = os.listdir(os.path.join(self.folder, 'x'))
        return [name for name in names if name[0] != '.']

    def list_y_names(self, ) -> List[str]:
        """
        获取所有可用标签名
        """
        names: List[str] = os.listdir(os.path.join(self.folder, 'y'))
        return [name for name in names if name[0] != '.']
    
    def read_multi_x(
        self,
        names: List[str],
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
        apply_func: Callable[[MinuteData], MinuteData] | None = None,
        n_worker: int = 32,
    ) -> MinuteData:
        """
        多线程读取多个x并合并
        """
        first_x: MinuteData = self.read_x(
            name=names[0],
            start_dt=start_dt,
            end_dt=end_dt,
            apply_func=apply_func,
        )
        x: MinuteData = MinuteData(
            dates=first_x.dates.data,
            minutes=first_x.minutes.data,
            tickers=first_x.tickers.data,
            names=np.array(names),
        )

        def read_and_assign(i: int) -> None:
            x.data[:, :, :, i] = self.read_x(
                name=names[i],
                start_dt=start_dt,
                end_dt=end_dt,
                apply_func=apply_func,
            ).data.squeeze(-1)

        with ThreadPoolExecutor(n_worker) as executor:
            futures = [
                executor.submit(read_and_assign, i) for i in range(len(names))
            ]
            for future in tqdm(as_completed(futures), total=len(names)):
                try:
                    future.result()
                except Exception as e:
                    print(e)
        return x

    def read_x(
        self,
        name: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
        apply_func: Callable[[MinuteData], MinuteData] | None = None,
    ) -> MinuteData:
        """
        读取单个因子
        """
        x: MinuteData = self._read_data(
            file=os.path.join(self.folder, 'x', name),
            start_dt=start_dt,
            end_dt=end_dt,
        )
        if apply_func is not None:
            x = apply_func(x)
        return x

    def read_y(
        self,
        name: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
        apply_func: Callable[[MinuteData], MinuteData] | None = None,
        return_raw: bool = False,
    ) -> MinuteData:
        """
        读取单个标签

        如果return_raw为False的话, 会默认对y做预处理:
        1. 将inf替换为nan
        2. 将trade_mask为False的地方替换为nan
        """
        y: MinuteData = self._read_data(
            file=os.path.join(self.folder, 'y', name),
            start_dt=start_dt,
            end_dt=end_dt,
        )
        if return_raw:
            return y

        trade_mask: np.ndarray = self.read_trade_mask(start_dt, end_dt)
        y.data[np.isinf(y.data) & (~trade_mask)] = np.nan  

        if apply_func is not None:
            y = apply_func(y)
        return y

    def _read_data(
        self,
        file: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> MinuteData:
        """
        从单个bin文件中读取数据
        """
        start: int = 0
        if start_dt is not None:
            start = np.searchsorted(self.dates, start_dt, side="left")

        end: int = len(self.dates)
        if end_dt is not None:
            end = np.searchsorted(self.dates, end_dt, side="left")

        data: np.ndarray = np.fromfile(
            file,
            dtype=np.float32,
            offset=(
                start * len(self.minutes) * len(self.tickers)
                * np.dtype(np.float32).itemsize
            ),
            count=(end - start) * len(self.minutes) * len(self.tickers),
        )
        data = data.reshape(
            end - start,
            len(self.minutes),
            len(self.tickers),
            1,
        )

        return MinuteData(
            dates=self.dates[start: end],
            minutes=self.minutes,
            tickers=self.tickers,
            names=np.array([file.split('/')[-1], ]),
            data=data,
        )


class FlattenBinMinuteDataBase(BinMinuteDataBase):
    """
    以bin文件为核心的分钟数据库

    与BinMinuteDataBase不同点在于x和y去除nan后展平

    常用于训练树模型

    数据库格式如下:
    datetimes.npy: 一维np.datetimes数组
    y/*: 仅有一个bin文件对应一个标签(n_sample, )
    x/*: 一个bin文件对应一个因子(n_sample, )
    """

    def __init__(self, folder: str) -> None:
        self.folder: str = folder

        self.dts: np.ndarray = np.load(
            os.path.join(self.folder, "datetimes.npy")
        )

        y_names: List[str] = os.listdir(os.path.join(self.folder, 'y'))
        assert len(y_names) == 1
        self.y_name: str = y_names[0]

    def list_x_names(self, ) -> List[str]:
        """
        获取所有可用因子名
        """
        names: List[str] = os.listdir(os.path.join(self.folder, 'x'))
        return [name for name in names if name[0] != '.']
    
    def read_x(
        self,
        name: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> np.ndarray:
        return self._read_data(
            os.path.join(self.folder, 'x', name),
            start_dt=start_dt,
            end_dt=end_dt,
        )
    
    def read_y(
        self,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> np.ndarray:
        return self._read_data(
            os.path.join(self.folder, 'y', self.y_name),
            start_dt=start_dt,
            end_dt=end_dt,
        )

    def _read_data(
        self,
        file: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> np.ndarray:
        start: int = 0
        if start_dt is not None:
            start = np.searchsorted(self.dts, start_dt, side="left")

        end: int = len(self.dts)
        if end_dt is not None:
            end = np.searchsorted(self.dts, end_dt, side="left")

        return np.fromfile(
            file,
            dtype=np.float32,
            offset=start * np.dtype(np.float32).itemsize,
            count=end - start,
        )

    def read_multi_x(
        self,
        names: List[str],
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
        n_worker: int = 32,
    ) -> np.ndarray:
        first_x: np.ndarray = self.read_x(
            name=names[0],
            start_dt=start_dt,
            end_dt=end_dt,
        )
        x: np.ndarray = np.empty((len(first_x), len(names)), dtype=np.float32)

        def read_and_assign(i: int) -> None:
            x[:, i] = self.read_x(
                name=names[i],
                start_dt=start_dt,
                end_dt=end_dt,
            )

        with ThreadPoolExecutor(n_worker) as executor:
            futures = [
                executor.submit(read_and_assign, i) for i in range(len(names))
            ]
            for future in tqdm(as_completed(futures), total=len(names)):
                try:
                    future.result()
                except Exception as e:
                    print(e)
        return x


class HDF5MinuteDataBase:
    """
    以h5文件为核心的分钟数据库

    数据库格式如下:
    datetime.bin: 时间戳
    names.txt: 因子名称
    x.h5: 因子数据
        "x": 一组因子(n_date, n_minute, n_ticker, n_name)
        ""
    """

    def __init__(self, folder: str, bin_db: BinMinuteDataBase) -> None:
        self.folder: str = folder
        self.bin_db: BinMinuteDataBase = bin_db
