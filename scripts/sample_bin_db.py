"""
用于采样新建二进制分钟频数据库的脚本
"""

import argparse

import tidemodel as tm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_folder", type=str, required=True)
    parser.add_argument("--date_sample_rate", type=float, default=1.0)
    parser.add_argument("--minute_sample_rate", type=float, default=1.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--n_worker", type=int, default=32)
    args = parser.parse_args()

    tm.data.sample_bin_db(
        from_folder=args.from_folder,
        to_folder=args.to_folder,
        date_sample_rate=args.date_sample_rate,
        minute_sample_rate=args.minute_sample_rate,
        append=args.append,
        n_worker=args.n_worker,
    )
