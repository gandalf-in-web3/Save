"""
去除nan后展平二进制分钟频数据库
"""

import argparse
import os
from typing import List

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

import tidemodel as tm


def read_and_save_x(
    name: str,
    from_folder: str,
    to_folder: str,
    y_name: str,
    start_minute: int,
    end_minute: int,
) -> None:
    from_db = tm.data.BinMinuteDataBase(from_folder)

    # 计算mask
    y: tm.data.MinuteData = from_db.read_y(
        name=y_name,
        apply_func=lambda y: tm.data.cross_norm(y[:, start_minute: end_minute]),
    )
    mask: np.ndarray = ~np.isnan(y.data.squeeze(-1))

    x: np.ndarray = from_db.read_x(
        name=name,
        apply_func=lambda x: x[:, start_minute: end_minute],
    ).data.squeeze(-1)
    x[mask].tofile(os.path.join(to_folder, 'x', name))


def flatten(
    from_folder: str,
    to_folder: str,
    y_name: str,
    start_minute: int = 0,
    end_minute: int = 241,
    n_worker: int = 32,
) -> None:
    """
    根据指定label去除nan后展平并写入新的数据库
    """

    os.makedirs(to_folder, exist_ok=False)
    os.mkdir(os.path.join(to_folder, 'x'))
    os.mkdir(os.path.join(to_folder, 'y'))

    from_db = tm.data.BinMinuteDataBase(from_folder)
    
    # 读取y并作截面归一化
    y: tm.data.MinuteData = from_db.read_y(
        name=y_name,
        apply_func=lambda y: tm.data.cross_norm(y[:, start_minute: end_minute]),
    )

    # (n_date, n_minute, n_ticker)
    mask: np.ndarray = ~np.isnan(y.data.squeeze(-1))
    dts: np.ndarray = (
        y.dates.data[:, None].astype("datetime64[m]")
        + y.minutes.data[None, :]
    )
    dts = np.repeat(dts[:, :, None], len(y.tickers), axis=-1)

    # 保存dts
    np.save(os.path.join(to_folder, "datetimes.npy"), dts[mask])

    # 保存y
    y.data.squeeze(-1)[mask].tofile(os.path.join(to_folder, 'y', y_name))

    # 读取x处理并保存
    x_names: List[str] = from_db.list_x_names()

    with ProcessPoolExecutor(max_workers=n_worker) as executor:
        print("start flatten x")
        futures = [executor.submit(
            read_and_save_x,
            name,
            from_folder,
            to_folder,
            y_name,
            start_minute,
            end_minute,
        ) for name in x_names]
        for future in tqdm(as_completed(futures), total=len(x_names)):
            try:
                future.result()
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_folder", type=str, required=True)
    parser.add_argument("--y_name", type=str, required=True)
    parser.add_argument("--start_minute", type=int, default=0)
    parser.add_argument("--end_minute", type=int, default=241)
    parser.add_argument("--n_worker", type=int, default=32)
    args = parser.parse_args()

    flatten(
        from_folder=args.from_folder,
        to_folder=args.to_folder,
        y_name=args.y_name,
        start_minute=args.start_minute,
        end_minute=args.end_minute,
        n_worker=args.n_worker,
    )
