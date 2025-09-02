import os
import datetime
from typing import List

import numpy as np
from tqdm.auto import tqdm

import tidemodel as tm


folder: str = "/data/factor_data/Bar1m_20210101_20231231"


def read_datetimes():
    date_path = os.path.join(folder, "datetime.bin")
    fromtimestamp = lambda x: datetime.datetime.fromtimestamp(x) 
    timestp = np.fromfile(date_path, dtype=int) // (10**9)
    return np.array(
        np.vectorize(fromtimestamp)(timestp[:]), dtype="datetime64[m]"
    )


def read_tickers():
    return np.loadtxt(os.path.join(folder, "ticker.bin"), dtype=str)


def read_x(name: str) -> np.ndarray:
    return np.fromfile(
        os.path.join(folder, 'x', name),
        dtype=np.float32,
    ).reshape(727, 241, 5238)


def read_y(name: str) -> np.ndarray:
    return np.fromfile(
        os.path.join(folder, 'y', name),
        dtype=np.float32,
    ).reshape(-1, 241, 5238)


def read_multi_x(names: List[str]) -> np.ndarray:
    x: np.ndarray = np.empty((727, 241, 5238, len(names)), dtype=np.float32)
    for i, name in enumerate(tqdm(names)):
        x[:, :, :, i] = read_x(name)
    return x


def read_mask() -> np.ndarray:
    return read_x("TradableUniv").astype(bool)


def test_bin_db() -> None:
    db = tm.data.BinMinuteDataBase(folder)

    assert (db.dts == read_datetimes()).all()
    assert len(db.dates) == 727
    assert db.freq == 1 and len(db.minutes) == 241

    assert (db.tickers == read_tickers()).all()

    mask: np.ndarray = read_mask()
    assert (db.read_mask().squeeze(-1) == mask).all()

    y: np.ndarray = read_y("Y0602q")
    norm_y: np.ndarray = y.copy()
    norm_y[(~mask) | np.isinf(norm_y)] = np.nan
    mean: np.ndarray = np.nanmean(norm_y, axis=2, keepdims=True)
    std: np.ndarray = np.nanstd(norm_y, axis=2, keepdims=True)
    norm_y = (norm_y - mean) / (std + 1e-8)

    # 检查读取数据y
    assert np.array_equal(
        y[: 10],
        db.read_y("Y0602q", return_raw=True).data.squeeze(-1)[: 10],
        equal_nan=True,
    )

    # 检查读取数据y并做预处理
    assert np.array_equal(
        norm_y[: 10],
        tm.data.cross_norm(db.read_y("Y0602q")).data.squeeze(-1)[: 10],
        equal_nan=True,
    )

    # 检查按时间读取数据y
    assert np.array_equal(
        norm_y[20: -124][: 10],
        tm.data.cross_norm(db.read_y(
            "Y0602q",
            np.datetime64("2021-02-01"),
            np.datetime64("2023-07-01"),
        ).data.squeeze(-1))[: 10],
        equal_nan=True,
    )

    # 检查读取多个x
    x_names: List[str] = db.list_x_names()
    assert np.array_equal(
        read_multi_x(x_names[: 4])[20: -124, 30: 210][: 10],
        db.read_multi_data(
            'x',
            x_names[: 4],
            date_slice=slice(np.datetime64("2021-02-01"), np.datetime64("2023-07-01")),
            minute_slice=slice(30, 210),
        ).data[: 10],
        equal_nan=True,
    )


def test_sample_bin_db() -> None:
    db = tm.data.BinMinuteDataBase(folder)
    sample_db = tm.data.BinMinuteDataBase(
        "/data/wangwx/Bar1m_20210101_20231231/sample0.04"
    )

    # 检查时间
    assert (sample_db.dts == db.dts.reshape(727, 241)[
        ::5, ::5
    ].reshape(-1)).all()
    assert sample_db.freq == 5
    assert (sample_db.dates == db.dates[::5]).all()
    assert (sample_db.minutes == db.minutes[::5]).all()

    # 检查品种
    assert (sample_db.tickers == db.tickers).all()

    # 检查mask
    assert (sample_db.read_mask() == db.read_mask()[::5, ::5]).all()

    # 检查y
    assert np.array_equal(
        tm.data.cross_norm(
            sample_db.read_y("Y0602q")
        ).data[:10],
        tm.data.cross_norm(
            db.read_y("Y0602q")[::5, ::5]
        ).data[:10],
        equal_nan=True,
    )

    # 检查x
    x_names: List[str] = db.list_x_names()
    assert np.array_equal(
        sample_db.read_multi_data('x', x_names[: 4]).data[: 10],
        db.read_multi_data(
            'x',
            x_names[: 4],
            date_slice=slice(None, None, 5),
            minute_slice=slice(None, None, 5)
        ).data[: 10],
        equal_nan=True,
    )


def test_hdf5_db() -> None:
    hdf5_db = tm.data.HDF5MinuteDataBase(
        "/data/wangwx/Bar1m_20210101_20231231/x611", folder
    )

    # 检查数据集读取
    x1, y1 = hdf5_db.read_dataset(
        "Y0602q",
        date_slice=slice(
            np.datetime64("2021-02-01"), np.datetime64("2021-02-03")
        ),
        minute_slice=slice(30, 210),
    )

    x2, y2 = hdf5_db.bin_db.read_dataset(
        hdf5_db.names,
        "Y0602q",
        date_slice=slice(
            np.datetime64("2021-02-01"), np.datetime64("2021-02-03")
        ),
        minute_slice=slice(30, 210),
    )

    assert np.array_equal(x1.data, x2.data, equal_nan=True)
    assert np.array_equal(y1.data, y2.data, equal_nan=True)

    # 检查数据集懒加载
    x3, y3 = hdf5_db.read_dataset_lazy("Y0602q")
    assert np.array_equal(x1.data, x3[
        np.datetime64("2021-02-01"): np.datetime64("2021-02-03"),
        30: 210,
    ].data, equal_nan=True)

    assert np.array_equal(
        y3.data, hdf5_db.bin_db.read_y("Y0602q").data, equal_nan=True
    )

    # 检查读取展平数据集
    x4, y4 = hdf5_db.read_flatten_dataset(
        "Y0602q",
        date_slice=slice(
            np.datetime64("2021-02-01"), np.datetime64("2021-02-03")
        ),
        minute_slice=slice(30, 210),
    )

    x5, y5 = hdf5_db.bin_db.read_flatten_dataset(
        hdf5_db.names,
        "Y0602q",
        date_slice=slice(
            np.datetime64("2021-02-01"), np.datetime64("2021-02-03")
        ),
        minute_slice=slice(30, 210),
    )

    assert np.array_equal(x4, x5, equal_nan=True)
    assert np.array_equal(y4, y5, equal_nan=True)


if __name__ == "__main__":
    test_bin_db()
    test_sample_bin_db()
    test_hdf5_db()
