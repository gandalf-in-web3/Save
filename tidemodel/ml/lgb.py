import logging
import os
import shutil
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import BinMinuteDataBase, MinuteData, cross_ic
from ..utils import get_logger, read_txt, reset_logger, set_seed


class LGBExperiment:
    """
    LightGBM实验
    """

    config: Dict[str, Any] = {
        "seed": 42,
        "folder": None,
        "train": True,

        "bin_folder": None,
        "x_names_file": None,
        "y_name": None,
        "train_date_slice": None,
        "valid_date_slice": None,
        "test_date_slice": None,
        "minute_slice": slice(None),
        "tickers": slice(None),

        "params": {},
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config.update(config)
        
        self._set_seed()
        self._init_folder()

        # 数据集
        self.train_dataset: lgb.Dataset | None = None

        self.valid_y: MinuteData | None = None
        self.valid_y_pred: MinuteData | None = None
        self.valid_dataset: lgb.Dataset | None = None
        
        self.flatten_test_x: np.ndarray | None = None
        self.test_y: MinuteData | None = None
        self.test_y_pred: MinuteData | None = None

        # 模型
        self.model: lgb.Booster = None

        # 其他
        self.metric: Dict[str, float] = {}
        self.importance_df: pd.DataFrame = None

    def _set_seed(self, ) -> None:
        """
        设置随机数保证可以复现
        """
        set_seed(self.config["seed"])
        self.config["params"]["seed"] = self.config["seed"]
        self.config["params"]["deterministic"] = True

    def _init_folder(self, ) -> None:
        """
        创建文件夹和日志
        """
        if self.config["train"]:
            # 训练时创建文件夹
            if os.path.exists(self.config["folder"]):
                shutil.rmtree(self.config["folder"], ignore_errors=True)
            os.makedirs(self.config["folder"])

        # 设置lightGBM的logger
        self.logger: logging.Logger = get_logger(
            self.config["folder"], "main"
        )
        lgb.register_logger(self.logger) 
        self.logger.info(f"config is {self.config}")

    def load_data(self, ) -> None:
        """
        基于二进制数据库加载数据集
        """
        bin_db = BinMinuteDataBase(self.config["bin_folder"])
        x_names: List[str] = read_txt(self.config["x_names_file"])

        # 训练集按照y中非nan的地方展平用于加速训练
        if self.config["train_date_slice"] is not None:
            train_x, train_y = bin_db.read_flatten_dataset(
                x_names=x_names,
                y_name=self.config["y_name"],
                date_slice=self.config["train_date_slice"],
                minute_slice=self.config["minute_slice"],
                tickers=self.config["tickers"],
            )
            self.train_dataset = lgb.Dataset(
                train_x,
                label=train_y,
                feature_name=x_names,
            )

            # 验证集无需去除nan用于计算截面IC
            assert self.config["valid_date_slice"] is not None
            valid_x, self.valid_y = bin_db.read_dataset(
                x_names=x_names,
                y_names=self.config["y_name"],
                date_slice=self.config["valid_date_slice"],
                minute_slice=self.config["minute_slice"],
                tickers=self.config["tickers"],
            )
            self.valid_y_pred = MinuteData(
                dates=self.valid_y.dates.data,
                minutes=self.valid_y.minutes.data,
                tickers=self.valid_y.tickers.data,
                names=np.array([self.config["y_name"], ]),
            )
            self.valid_dataset = lgb.Dataset(
                valid_x.data.reshape(-1, len(x_names)),
                label=self.valid_y.data.reshape(-1),
                feature_name=x_names,
            )
            self.logger.info(
                f"{len(train_y)} train, "
                f"{len(self.valid_y.data.reshape(-1))} valid"
            )

        # 测试集无需去除nan用于计算截面IC
        if self.config["test_date_slice"] is not None:
            test_x, self.test_y = bin_db.read_dataset(
                x_names=x_names,
                y_names=self.config["y_name"],
                date_slice=self.config["test_date_slice"],
                minute_slice=self.config["minute_slice"],
                tickers=self.config["tickers"],
            )
            self.flatten_test_x = test_x.reshape(-1, len(x_names))
            self.test_y_pred = MinuteData(
                dates=self.test_y.dates.data,
                minutes=self.test_y.minutes.data,
                tickers=self.test_y.tickers.data,
                names=np.array([self.config["y_name"], ]),
            )
            self.logger.info(f"{len(self.flatten_test_x.shape[0])} test")

    def train(self, ) -> None:
        """
        训练模型
        """
        self.model = lgb.train(
            self.config["params"],
            self.train_dataset,
            valid_sets=self.valid_dataset,
            feval=self._valid,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=1),
            ],
        )
        self.model.save_model(
            os.path.join(self.config["folder"], "model.txt"),
            num_iteration=self.model.best_iteration
        )

    def _valid(self, y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple:
        """
        在验证集上计算横截面IC
        """
        self.valid_y_pred.data = y_pred.reshape(self.valid_y_pred.shape)
        return "cross_ic", cross_ic(self.valid_y, self.valid_y_pred), True

    def test(self, ) -> None:
        """
        在测试集上计算横截面IC
        """
        self.model = lgb.Booster(
            model_file=os.path.join(self.config["folder"], "model.txt")
        )

        self.test_y_pred = self.model.predict(
            self.flatten_test_x,
            num_iteration=self.model.best_iteration
        ).reshape(self.test_y_pred.shape)
        
        self.metric = {"cross_ic": cross_ic(self.test_y, self.test_y_pred)}
        self.logger.info(f"metric: {self.metric}")

    def compute_importance(self, ) -> None:
        """
        根据训练结果计算重要度得分
        """
        feature_names = self.model.feature_name()
        gain = self.model.feature_importance(importance_type="gain")
        split = self.model.feature_importance(importance_type="split")
        
        self.importance_df = pd.DataFrame({
            "gain": gain, "split": split
        }, index=feature_names)

        self.importance_df.to_csv(
            os.path.join(self.config["folder"], "importance_df.csv")
        )

    def close(self, ) -> None:
        """
        主动调用以释放资源
        """
        reset_logger(self.logger)
