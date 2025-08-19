"""
通过采样新建二进制分钟频数据库
"""

import argparse
import os
from typing import List

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

import tidemodel as tm


def minute_sample(
    from_folder: str,
    to_folder: str,
    date_sample_rate: float,
    minute_sample_rate: float,
    append: bool = False,
    n_worker: int = 32,
) -> None:
    """
    从现有数据库中按采样率在分钟轴上采样
    """

    from_db = tm.data.BinMinuteDataBase(from_folder)
    x_names: List[str] = from_db.list_x_names()
    y_names: List[str] = from_db.list_y_names()
    date_step: int = int(1 / date_sample_rate)
    minute_step: int = int(1 / minute_sample_rate)

    if append:
        # 添加模式中会保留已经采样过的数据
        to_db = tm.data.BinMinuteDataBase(to_folder)
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
        # 东八区转UTC区需要减8小时
        datetimes: np.ndarray = (
            from_db.dates.astype("datetime64[m]")[::date_step, None]
            + from_db.minutes[None, ::minute_step]
        ) - np.timedelta64(8, 'h')
        # 转化为ns时间戳
        timestamps: np.ndarray = datetimes.astype(
            "datetime64[s]"
        ).astype(np.int64) * (10**9)
        timestamps.tofile(os.path.join(to_folder, "datetime.bin"))

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
                print(e)

        print("start sample y")
        futures = [executor.submit(read_and_save_y, name) for name in y_names]   
        for future in tqdm(as_completed(futures), total=len(y_names)):
            try:
                future.result()
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_folder", type=str, required=True)
    parser.add_argument("--date_sample_rate", type=float, default=1.0)
    parser.add_argument("--minute_sample_rate", type=float, default=1.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--n_worker", type=int, default=32)
    args = parser.parse_args()

    minute_sample(
        from_folder=args.from_folder,
        to_folder=args.to_folder,
        date_sample_rate=args.date_sample_rate,
        minute_sample_rate=args.minute_sample_rate,
        append=args.append,
        n_worker=args.n_worker,
    )
