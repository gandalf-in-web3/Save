import time

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import tidemodel as tm
tm.utils.set_seed(int(time.time()))


n_step: int = 10

num_workers: int = 392

hdf5_folder: str = "/mnt/3fs/data/wangwx/1m_2020_2024/x1316"


if __name__ == "__main__":
    bin_db = tm.data.BinMinuteDataBase(
        "/intraday/factor_new/Bar1m_20200101_20241231/"
    )
    whole_y: tm.data.MinuteData = bin_db.read_multi_data(
        'y', ["Y0602q"], n_worker=0
    )
    hdf5_db = tm.data.HDF5MinuteDataBase(
        hdf5_folder,
        "/intraday/factor_new/Bar1m_20200101_20241231/",
    )
    dataset = tm.dl.LazyMinuteDataset(
        hdf5_db,
        whole_y=whole_y,
        date_slice=slice(np.datetime64("2020-01-01"), None),
        minute_slice=slice(None, 210),
        tickers=slice(None),
        seq_len=32,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=tm.dl.collate_fn,
        num_workers=num_workers,
        shuffle=True,
    )

    t1 = time.time()
    current_step: int = 0

    for data in tqdm(dataloader):
        current_step += 1
        if current_step == n_step:
            break

    print(f"{n_step} step cost {time.time() - t1}s")
