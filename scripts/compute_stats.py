"""
计算因子指标
"""

import argparse

import numpy as np

import tidemodel as tm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--from_folder", type=str, required=True)
    parser.add_argument("--to_file", type=str, required=True)
    parser.add_argument("--y_name", type=str, default=None)
    parser.add_argument("--start_dt", type=str, default=None)
    parser.add_argument("--end_dt", type=str, default=None)
    parser.add_argument("--start_minute", type=int, default=None)
    parser.add_argument("--end_minute", type=int, default=None)
    parser.add_argument("--n_worker", type=int, default=128)
    args = parser.parse_args()

    if args.type == "norm":
        tm.data.build_norm_stats_db(
            from_folder=args.from_folder,
            to_file=args.to_file,
            start_dt=(
                None if args.start_dt is None
                else np.datetime64(args.start_dt)
            ),
            end_dt=(
                None if args.end_dt is None
                else np.datetime64(args.end_dt)
            ),
            start_minute=args.start_minute,
            end_minute=args.end_minute,
            n_worker=args.n_worker,
        )
    elif args.type == "selection":
        assert args.y_name is not None

        tm.data.build_selection_stats_db(
            from_folder=args.from_folder,
            to_file=args.to_file,
            y_name=args.y_name,
            start_dt=(
                None if args.start_dt is None
                else np.datetime64(args.start_dt)
            ),
            end_dt=(
                None if args.end_dt is None
                else np.datetime64(args.end_dt)
            ),
            start_minute=args.start_minute,
            end_minute=args.end_minute,
            n_worker=args.n_worker,
        )
    else:
        raise ValueError(f"{args.type} must be in ['norm', 'selection']")
