"""
新建HDF5分钟频数据库
"""

import argparse

import tidemodel as tm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_folder", type=str, required=True)
    parser.add_argument("--x_names_file", type=str, default=None)
    parser.add_argument("--n_worker", type=int, default=64)
    args = parser.parse_args()

    tm.data.build_hdf5_db(
        from_folder=args.from_folder,
        to_folder=args.to_folder,
        x_names=tm.utils.read_txt(args.x_names_file),
        n_worker=args.n_worker,
    )
