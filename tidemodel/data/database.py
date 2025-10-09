"""
保存和读取分钟数据
"""

import os
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from functools import partial
from typing import Callable, Dict, List, Literal, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .data import HDF5Ndarray, MinuteData, SortedIndex
from .ops import cross_ic, long_ic, mr
from ..utils import read_txt, validate_float_to_int, write_txt


"""
用于读取数据的数据库类
"""

class BinMinuteDataBase:
    """
    基于bin文件构建的分钟数据库

    数据库格式如下:
    datetime.bin: 时间戳
    ticker.bin: 品种名
    x/*: 一个bin文件对应一个因子(n_date, n_minute, n_ticker)
    y/*: 一个bin文件对应一个标签(n_date, n_minute, n_ticker)

    数据库的内存分布为因子连续, 适合需要读取不同因子的场景,
    如因子选择
    """

    def __init__(self, folder: str) -> None:
        self.folder: str = folder

        self._load_dts()

        # 读取品种名
        self.tickers: np.ndarray = np.loadtxt(
            os.path.join(self.folder, "ticker.bin"), dtype=str
        )

    def _load_dts(self, ) -> None:
        """
        加载时间信息
        """
        try:
            # 读取ns时间戳, UTC时区转换为东八区需要加八小时
            timestamps: np.ndarray = np.fromfile(
                os.path.join(self.folder, "datetime.bin"), dtype=np.int64
            ) // (10**9)
            self.dts: np.ndarray = timestamps.astype("datetime64[s]").astype(
                "datetime64[m]"
            ) + np.timedelta64(8, 'h')
        except FileNotFoundError:
            self.dts: np.ndarray = np.load(
                os.path.join(self.folder, "datetime.npy")
            )

        # 根据datetimes计算频率, 日期和分钟
        self.freq: int = validate_float_to_int(
            (self.dts[1] - self.dts[0]) / np.timedelta64(1, 'm')
        )
        self.minutes: np.ndarray = np.arange(0, 240 + 1, self.freq)
        self.dates: np.ndarray = self.dts[::len(self.minutes)].astype("datetime64[D]")

    def list_x_names(self, ) -> List[str]:
        """
        获取所有可用因子名
        """
        names: List[str] = os.listdir(os.path.join(self.folder, 'x'))
        names = [name for name in names if name[0] != '.']
        return [name for name in names if name[-8: ] != '.feather']

    def list_y_names(self, ) -> List[str]:
        """
        获取所有可用标签名

        有的文件是以.feature结尾的, 也需要去掉
        """
        names: List[str] = os.listdir(os.path.join(self.folder, 'y'))
        names = [name for name in names if name[0] != '.']
        return [name for name in names if name[-8: ] != '.feather']

    def read_x(
        self,
        name: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> MinuteData:
        """
        读取单个因子
        """
        return self._read_data(
            file=os.path.join(self.folder, 'x', name),
            start_dt=start_dt,
            end_dt=end_dt,
        )

    def read_y(
        self,
        name: str,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
        return_raw: bool = False,
    ) -> MinuteData:
        """
        读取单个标签

        return_raw为False时会对y做以下三点处理:
        1. 将inf替换为nan
        2. 将mask为False的地方替换为nan
        """
        y: MinuteData = self._read_data(
            file=os.path.join(self.folder, 'y', name),
            start_dt=start_dt,
            end_dt=end_dt,
        )
        if return_raw:
            return y

        trade_mask: np.ndarray = self.read_mask(start_dt, end_dt)
        y.data[np.isinf(y.data) | (~trade_mask)] = np.nan  
        return y

    def read_mask(
        self,
        start_dt: np.datetime64 | None = None,
        end_dt: np.datetime64 | None = None,
    ) -> np.ndarray:
        """
        读取代表是否可交易的mask
        """
        return self._read_data(
            file=os.path.join(self.folder, "x", "TradableUniv"),
            start_dt=start_dt,
            end_dt=end_dt,
        ).data.astype(bool)

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

    def read_multi_data(
        self,
        mode: Literal['x', 'y'],
        names: str | List[str],
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        return_raw: bool = False,
        n_worker: int = 32,
    ) -> MinuteData:
        """
        多线程读取多个数据并合并返回MinuteData
        """
        if mode == 'x':
            read_func: Callable = self.read_x
        elif mode == 'y':
            read_func = partial(self.read_y, return_raw=return_raw)
        else:
            raise ValueError(f"{mode} must be x or y")

        if isinstance(names, str):
            names = [names]

        # 如果只需读取一个数据, 无需多线程直接返回
        first_data: MinuteData = read_func(
            names[0], date_slice.start, date_slice.stop
        )[:: date_slice.step, minute_slice, tickers]

        if len(names) == 1:
            return first_data

        # 根据读取到的第一个数据初始化
        data: MinuteData = MinuteData(
            dates=first_data.dates.data,
            minutes=first_data.minutes.data,
            tickers=first_data.tickers.data,
            names=np.array(names),
        )

        def read_and_assign(i: int) -> None:
            data.data[:, :, :, i] = read_func(
                names[i], date_slice.start, date_slice.stop
            )[:: date_slice.step, minute_slice, tickers].data.squeeze(-1)

        if n_worker == 0:
            for i in range(len(names)):
                read_and_assign(i)
            return data

        with ThreadPoolExecutor(min(n_worker, len(names))) as executor:
            futures = [
                executor.submit(read_and_assign, i) for i in range(len(names))
            ]
            for future in tqdm(as_completed(futures), total=len(names)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error {e}")
        return data

    def read_dataset(
        self,
        x_names: str | List[str],
        y_names: str | List[str],
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        n_worker: int = 32,
    ) -> Tuple[MinuteData, MinuteData]:
        """
        读取数据集
        """
        if isinstance(x_names, str):
            x_names = [x_names]

        x: MinuteData = self.read_multi_data(
            mode='x',
            names=x_names,
            date_slice=date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
            n_worker=n_worker,
        )

        if isinstance(y_names, str):
            y_names = [y_names]

        y: MinuteData = self.read_multi_data(
            mode='y',
            names=y_names,
            date_slice=date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
            n_worker=n_worker,
        )
        return x, y
    
    def read_flatten_dataset(
        self,
        x_names: str | List[str],
        y_name: str,
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        apply_func_x: Callable[[MinuteData], MinuteData] = lambda x: x,
        apply_func_y: Callable[[MinuteData], MinuteData] = lambda x: x,
        n_worker: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取展平到品种的数据集
        
        展平方法是按y去除nan后展平

        apply_func_x和apply_func_y是针对索引后的MinuteData做操作
        """
        y: MinuteData = self.read_y(
            y_name, date_slice.start, date_slice.stop
        )[:: date_slice.step, minute_slice, tickers]
        y = apply_func_y(y)

        mask: np.ndarray = ~np.isnan(y.data)
        flatten_y: np.ndarray = y.data[mask]

        if isinstance(x_names, str):
            x_names = [x_names]

        flatten_x: np.ndarray = np.empty(
            (len(flatten_y), len(x_names)), dtype=np.float32
        )

        def read_and_assign(i: int) -> None:
            x: MinuteData = self.read_x(
                x_names[i], date_slice.start, date_slice.stop
            )[:: date_slice.step, minute_slice, tickers]
            x = apply_func_x(x)
            flatten_x[:, i] = x.data[mask]

        with ThreadPoolExecutor(min(n_worker, len(x_names))) as executor:
            futures = [
                executor.submit(read_and_assign, i) for i in range(len(x_names))
            ]
            for future in tqdm(as_completed(futures), total=len(x_names)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error {e}")
        return flatten_x, flatten_y


class HDF5MinuteDataBase:
    """
    基于h5文件构建的分钟数据库
    
    每个h5文件对应一个日期的固定因子数据, 内部按照时间点分块

    数据库格式如下:
    names.txt: 因子名
    {date}.h5:
        "x": 因子矩阵(n_date, n_minute, n_ticker, n_name)

    数据库的内存分布为时间连续, 适合需要固定因子读取不同时间的场景,
    如固定因子集的模型训练
    """

    def __init__(self, folder: str, bin_folder: str) -> None:
        self.folder: str = folder
        self.bin_db = BinMinuteDataBase(bin_folder)
        self.names: List[str] = read_txt(os.path.join(self.folder, "names.txt"))

        self.files: List[str] = [
            file for file in os.listdir(self.folder) if file[-3: ] == ".h5"
        ]
        self.files.sort()
        self.dates = np.array(
            [file.split('.')[0] for file in self.files],
            dtype="datetime64[D]"
        )

    def read_dataset(
        self,
        y_names: str | List[str],
        x_names: List[str] | slice = slice(None),
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        n_worker: int = 32,
    ) -> Tuple[MinuteData, MinuteData]:
        """
        读取数据集到内存中
        """
        # 计算需要读取的日期
        date_idx_slice: slice = SortedIndex(self.dates).index_range(
            date_slice.start, date_slice.stop, date_slice.step
        )
        dates: np.ndarray = self.dates[date_idx_slice]

        # 读取y
        y: MinuteData = self.bin_db.read_multi_data(
            mode='y',
            names=y_names,
            date_slice=date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
            n_worker=n_worker,
        )
        
        # 读取x
        x = MinuteData(
            dates=y.dates.data,
            minutes=y.minutes.data,
            tickers=y.tickers.data,
            names=np.array(self.names),
        )[:, :, :, x_names]

        def read_and_assign(i: int) -> None:
            with h5py.File(os.path.join(
                self.folder, f"{np.datetime_as_string(dates[i], unit='D')}.h5"
            ), 'r', locking=False) as f:
                x.data[i: i + 1] = MinuteData(
                    dates=np.array([dates[i], ]),
                    minutes=self.bin_db.minutes,
                    tickers=self.bin_db.tickers,
                    names=np.array(self.names),
                    data=f["x"][:],
                )[:, minute_slice, tickers, x_names].data

        with ThreadPoolExecutor(min(n_worker, len(dates))) as executor:
            futures = [
                executor.submit(read_and_assign, i) for i in range(len(dates))
            ]
            for future in tqdm(as_completed(futures), total=len(dates)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error {e}")
        return x, y
        
    def read_flatten_dataset(
        self,
        y_name: str,
        x_names: List[str] | slice = slice(None),
        date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        tickers: List[str] | slice = slice(None),
        apply_func_x: Callable[[MinuteData], MinuteData] = lambda x: x,
        apply_func_y: Callable[[MinuteData], MinuteData] = lambda x: x,
        n_worker: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取展平到品种的数据集
        
        展平方法是按y去除nan后展平

        apply_func_x和apply_func_y是针对索引后的MinuteData做操作
        """
        # 计算需要读取的日期
        date_idx_slice: slice = SortedIndex(self.dates).index_range(
            date_slice.start, date_slice.stop, date_slice.step
        )
        dates: np.ndarray = self.dates[date_idx_slice]

        # 读取y并展平
        y: MinuteData = self.bin_db.read_y(
            y_name, date_slice.start, date_slice.stop
        )[:: date_slice.step, minute_slice, tickers]
        y = apply_func_y(y)

        mask: np.ndarray = ~np.isnan(y.data)
        flatten_y: np.ndarray = y.data[mask]

        # 读取x并展平
        if x_names == slice(None):
            n_names: int = len(self.names)
        else:
            n_names = len(x_names)

        flatten_x = np.empty(
            (len(flatten_y), n_names), dtype=np.float32
        )

        def read_and_assign(i: int) -> None:
            with h5py.File(os.path.join(
                self.folder, f"{np.datetime_as_string(dates[i], unit='D')}.h5"
            ), 'r', locking=False) as f:
                x: MinuteData = MinuteData(
                    dates=np.array([dates[i], ]),
                    minutes=self.bin_db.minutes,
                    tickers=self.bin_db.tickers,
                    names=np.array(self.names),
                    data=f["x"][:],
                )[:, minute_slice, tickers, x_names]
                x = apply_func_x(x)

                # 计算应该填入的位置
                start = int(mask[: i].sum())
                end = start + int(mask[i].sum())
                flatten_x[start: end] = x.data[mask[i: i + 1].squeeze(-1)]

        with ThreadPoolExecutor(min(n_worker, len(dates))) as executor:
            futures = [
                executor.submit(read_and_assign, i) for i in range(len(dates))
            ]
            for future in tqdm(as_completed(futures), total=len(dates)):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error {e}")
        return flatten_x, flatten_y
        
    def read_dataset_lazy(
        self,
        y_names: str | List[str],
    ) -> Tuple[MinuteData, MinuteData]:
        """
        以懒加载模式读取数据集
        """
        # 读取y
        y: MinuteData = self.bin_db.read_multi_data(
            'y', y_names, n_worker=0
        )

        # 懒加载x, 将dataset组合成HDF5Ndarray
        self.handles: List[h5py.File] = [
            h5py.File(
                os.path.join(self.folder, file), 'r', locking=False
            ) for file in self.files
        ]
        
        x = MinuteData(
            dates=self.dates,
            minutes=y.minutes.data,
            tickers=y.tickers.data,
            names=np.array(self.names),
            data=HDF5Ndarray([handle["x"] for handle in self.handles]),
        )
        return x, y


class StatsDataBase:
    """
    基于pandas dataframe的因子指标数据库
    """

    def __init__(self, file: str) -> None:
        self.df: pd.DataFrame =  pd.read_csv(file, index_col=0)

    def list_x_names(self, ) -> List[str]:
        """
        返回目前已经计算的指标
        """
        return self.df.index.to_list()

    def get_stats(self, col: str, x_names: List[str]) -> np.ndarray:
        """
        返回指定因子的指标
        """
        return self.df[col].loc[x_names].to_numpy().astype(np.float32)


"""
用于新建数据库的函数
"""

def sample_bin_db(
    from_folder: str,
    to_folder: str,
    date_sample_rate: float = 1.0,
    minute_sample_rate: float = 1.0,
    append: bool = False,
    n_worker: int = 32,
) -> None:
    """
    对现有bin数据库采样并新建一个bin数据库
    """

    from_db = BinMinuteDataBase(from_folder)
    x_names: List[str] = from_db.list_x_names()
    y_names: List[str] = from_db.list_y_names()
    date_step: int = int(1 / date_sample_rate)
    minute_step: int = int(1 / minute_sample_rate)

    if append:
        # 添加模式中会保留已经采样过的数据
        to_db = BinMinuteDataBase(to_folder)
        x_names = list(set(x_names) - set(to_db.list_x_names()))
        y_names = list(set(y_names) - set(to_db.list_y_names()))

        # 读取第一个x来检查
        from_first_x: np.ndarray = from_db.read_x(
            to_db.list_x_names()[0]
        )[::date_step, ::minute_step].data
        to_first_x: np.ndarray = to_db.read_x(
            to_db.list_x_names()[0]
        ).data
        assert np.allclose(
            from_first_x[0], to_first_x[0], equal_nan=True, rtol=0, atol=0
        )
    else:
        # 新建模式下必须写入到一个新的目录中
        os.makedirs(to_folder, exist_ok=False)
        os.mkdir(os.path.join(to_folder, 'x'))
        os.mkdir(os.path.join(to_folder, 'y'))

        # 写入datetimes
        np.save(
            os.path.join(to_folder, "datetime.npy"),
            from_db.dts.reshape(
                len(from_db.dates), len(from_db.minutes)
            )[::date_step, ::minute_step].reshape(-1)
        )

        # 写入tickers
        np.savetxt(
            os.path.join(to_folder, "ticker.bin"),
            from_db.tickers,
            fmt="%s",
        )

    def read_and_save_x(name: str) -> None:
        x: np.ndarray = from_db.read_x(name)[::date_step, ::minute_step].data
        x.tofile(os.path.join(to_folder, 'x', name))

    def read_and_save_y(name: str) -> None:
        y: np.ndarray = from_db.read_y(
            name, return_raw=True
        )[::date_step, ::minute_step].data
        y.tofile(os.path.join(to_folder, 'y', name))

    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        print("start sample x")
        futures = [executor.submit(read_and_save_x, name) for name in x_names]
        for future in tqdm(as_completed(futures), total=len(x_names)):
            try:
                future.result()
            except Exception as e:
                print(f"Error {e}")

        print("start sample y")
        futures = [executor.submit(read_and_save_y, name) for name in y_names]   
        for future in tqdm(as_completed(futures), total=len(y_names)):
            try:
                future.result()
            except Exception as e:
                print(f"Error {e}")


def _read_and_save_hdf5(
    from_folder: str,
    to_folder: str,
    x_names: List[str],
    date: np.datetime64,
) -> None:
    """
    从二进制数据库中读取一天的数据并写入到hdf5中
    """
    from_db = BinMinuteDataBase(from_folder)
    x: MinuteData = from_db.read_multi_data(
        'x',
        x_names,
        date_slice=slice(date, date + np.timedelta64(1, 'D')),
        n_worker=1,
    )

    with h5py.File(os.path.join(
        to_folder, f"{np.datetime_as_string(date, unit='D')}.h5"
    ), 'w') as f:
        dataset = f.create_dataset(
            "x",
            data=x.data,
        )


def build_hdf5_db(
    from_folder: str,
    to_folder: str,
    x_names: List[str],
    date_slice: slice = slice(None),
    n_worker: int = 32,
) -> None:
    """
    基于现有bin数据库按指定因子名新建一个hdf5数据库
    """

    from_db = BinMinuteDataBase(from_folder)

    # 新建文件夹并写入hdf5文件
    os.makedirs(to_folder, exist_ok=False)

    # 保存因子名称
    write_txt(os.path.join(to_folder, "names.txt"), x_names)

    # 按日期读取并保存h5文件
    date_indx_slice = SortedIndex(from_db.dates).index_range(
        date_slice.start, date_slice.stop, date_slice.step
    )
    dates: np.ndarray = from_db.dates[date_indx_slice]

    with ProcessPoolExecutor(n_worker) as executor:
        print("start read and save date.h5")
        futures = [executor.submit(
            _read_and_save_hdf5,
            from_folder,
            to_folder,
            x_names,
            date,
        ) for date in dates]
        
        for future in tqdm(as_completed(futures), total=len(dates)):
            try:
                future.result()
            except Exception as e:
                print(f"Error {e}")


def _read_and_compute_norm_stats(
    from_folder: str,
    x_name: str,
    start_dt: np.datetime64 | None,
    end_dt: np.datetime64 | None,
    start_minute: int | None,
    end_minute: int | None,
) -> Dict[str, float]:
    """
    计算因子指定日期的用于标准化的统计指标
    """

    from_db = BinMinuteDataBase(from_folder)
    x: MinuteData = from_db.read_x(x_name, start_dt, end_dt)
    x = x[:, start_minute: end_minute]
    
    x_stats: Dict[str, float] = {}
    (
        x_stats["x_05"],
        x_stats["x_1"],
        x_stats["x_25"],
        x_stats["x_50"],
        x_stats["x_75"],
        x_stats["x_99"],
        x_stats["x_995"]
    ) = np.nanpercentile(x.data, [
        0.5, 1, 25, 50, 75, 99, 99.5
    ])
    return x_stats


def build_norm_stats_db(
    from_folder: str,
    to_file: str,
    start_dt: np.datetime64,
    end_dt: np.datetime64,
    start_minute: int | None,
    end_minute: int | None,
    n_worker: int = 32,
) -> None:
    """
    基于现有bin数据库计算所有因子的用于标准化的统计指标
    """

    from_db = BinMinuteDataBase(from_folder)
    x_names: List[str] = from_db.list_x_names()

    with ProcessPoolExecutor(min(n_worker, len(x_names))) as executor:
        futures = {executor.submit(
            _read_and_compute_norm_stats,
            from_folder,
            name,
            start_dt,
            end_dt,
            start_minute,
            end_minute,
        ): i for i, name in enumerate(x_names)}

        x_stats: List[Dict[str, float]] = [None for _ in range(len(x_names))]

        for future in tqdm(as_completed(futures), total=len(x_names)):
            try:
                x_stats[futures[future]] = future.result()
            except Exception as e:
                print(f"Error {e}")
    
    stats_df = pd.DataFrame(x_stats, index=x_names, dtype=np.float32)
    stats_df.to_csv(to_file)


def _read_and_compute_selection_stats(
    from_folder: str,
    x_name: str,
    y_name: str,
    start_dt: np.datetime64 | None,
    end_dt: np.datetime64 | None,
    start_minute: int | None,
    end_minute: int | None,
) -> Dict[str, float]:
    """
    计算因子指定日期的用于因子选择的统计指标
    """

    from_db = BinMinuteDataBase(from_folder)
    
    x: MinuteData = from_db.read_x(x_name, start_dt, end_dt)
    y: MinuteData = from_db.read_y(y_name, start_dt, end_dt)

    x = x[:, start_minute: end_minute]
    y = y[:, start_minute: end_minute]

    return {
        "long_ic": long_ic(y, x),
        "long_ic30": long_ic(y[:, : 30], x[:, : 30]),
        "cross_ic": cross_ic(y, x),
        "cross_ic30": cross_ic(y[:, : 30], x[:, : 30]),
        "mr": mr(y, x),
        "mr30": mr(y[:, : 30], x[:, : 30]),
    }


def build_selection_stats_db(
    from_folder: str,
    to_file: str,
    y_name: str,
    start_dt: np.datetime64 | None,
    end_dt: np.datetime64 | None,
    start_minute: int | None,
    end_minute: int | None,
    n_worker: int = 32,
) -> None:
    """
    计算所有x的统计指标并写入到指定文件中

    支持更新模式, 仅重新计算文件中不存在的因子
    """

    from_db = BinMinuteDataBase(from_folder)
    x_names = from_db.list_x_names()

    with ProcessPoolExecutor(min(n_worker, max(len(x_names), 1))) as executor:
        futures = {executor.submit(
            _read_and_compute_selection_stats,
            from_folder,
            x_name,
            y_name,
            start_dt,
            end_dt,
            start_minute,
            end_minute
        ): i for i, x_name in enumerate(x_names)}

        x_stats: List[Dict[str, float]] = [None for _ in range(len(x_names))]

        for future in tqdm(as_completed(futures), total=len(x_names)):
            try:
                x_stats[futures[future]] = future.result()
            except Exception as e:
                print(f"Error: {e}")

    stats_df = pd.DataFrame(x_stats, index=x_names, dtype=np.float32)
    stats_df.to_csv(to_file)
