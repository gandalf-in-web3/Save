"""
Catboost模型
"""

import copy
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from ..ml.experiment import Experiment
from ..data import BinMinuteDataBase, MinuteData, cross_ic


class CatboostExperiment(Experiment):
    """
    Catboost实验
    """

    params = {
        "thread_count": -1,
        "seed": 42,
        "random_seed": 42,

        "loss_function": "RMSE",
        "eval_metric": "RMSE",

        "iterations": 1000,
        "learning_rate": 0.05,

        "border_count": 31,

        "depth": 10,
        "min_data_in_leaf": 10000,
        
        "subsample": 0.7,
        "rsm": 0.7,
        "l2_leaf_reg": 0.0,
    }

    def __init__(
        self,
        folder: str,
        create_folder: bool = True,
        params: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            folder=folder,
            create_folder=create_folder,
            params=params,
        )

        # 数据
        self.x_names: List[str] = None
        self.y_name: str = None

        self.train_dataset: Pool | None = None
        self.valid_dataset: Pool | None = None
        self.test_dataset: Pool | None = None
        
        # 未展平的数据用于计算测试集截面IC
        self.test_x: MinuteData | None = None
        self.test_y: MinuteData | None = None

        # 模型
        self.model: CatBoostRegressor = None

    def load_data(
        self,
        bin_folder: str,
        x_names: str | List[str],
        y_name: str,
        train_date_slice: slice = slice(None),
        valid_date_slice: slice = slice(None),
        test_date_slice: slice = slice(None),
        minute_slice: slice = slice(None),
        tickers: slice | List[str] = slice(None),
    ) -> None:
        """
        基于二进制数据库加载数据集
        """
        bin_db = BinMinuteDataBase(bin_folder)

        if isinstance(x_names, str):
            x_names = [x_names]
        self.x_names = x_names
        self.y_name = y_name

        # catboost的数据集中不允许出现nan
        # 按照y中非nan的地方展平
        flatten_train_x, flatten_train_y = bin_db.read_flatten_dataset(
            x_names=x_names,
            y_name=y_name,
            date_slice=train_date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
        )
        self.train_dataset = Pool(
            flatten_train_x,
            label=flatten_train_y,
            feature_names=self.x_names,
        )

        flatten_valid_x, flatten_valid_y = bin_db.read_flatten_dataset(
            x_names=self.x_names,
            y_name=self.y_name,
            date_slice=valid_date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
        )
        self.valid_dataset = Pool(
            flatten_valid_x,
            label=flatten_valid_y,
            feature_names=self.x_names,
        )

        self.test_x, self.test_y = bin_db.read_dataset(
            x_names=self.x_names,
            y_names=self.y_name,
            date_slice=test_date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
        )
        flatten_test_x, flatten_test_y = bin_db.read_flatten_dataset(
            x_names=self.x_names,
            y_name=self.y_name,
            date_slice=test_date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
        )
        self.test_dataset = Pool(
            flatten_test_x,
            label=flatten_test_y,
            feature_names=self.x_names,
        )

        self.logger.info(
            f"{len(flatten_train_y)} train, "
            f"{len(flatten_valid_y)} valid, "
            f"{len(flatten_test_y)} test"
        )

    def train(self, ) -> None:
        """
        训练模型
        """
        self.params.pop("seed")
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            self.train_dataset,
            eval_set=self.valid_dataset,
            use_best_model=True,
            early_stopping_rounds=50,
            log_cout=self.logger.info,
            log_cerr=self.logger.info,
        )
        self.model.save_model(os.path.join(self.folder, "model.cbm"))

    def test(self, ) -> Dict[str, float]:
        """
        在测试集上计算横截面IC
        """
        self.model = CatBoostRegressor()
        self.model.load_model(os.path.join(self.folder, "model.cbm"))

        test_y_pred: MinuteData = copy.copy(self.test_y)
        test_y_pred.data = self.model.predict(
            self.test_x.data.reshape(-1, self.test_x.data.shape[-1]),
        ).reshape(self.test_y.shape)

        metric: Dict[str, float] = {
            "cross_ic": cross_ic(self.test_y, test_y_pred),
        }
        self.logger.info(f"metric: {metric}")
        return metric

    def compute_importance(self, ) -> pd.DataFrame:
        """
        根据训练结果计算重要度得分
        """
        feature_names = self.model.feature_names_

        # 计算置换重要度得分
        perm_importance = self.model.get_feature_importance(
            self.test_dataset,
            type="LossFunctionChange"
        )

        importance_df = pd.DataFrame({
            "perm": perm_importance,
        }, index=feature_names)
        importance_df.to_csv(
            os.path.join(self.folder, "importance_df.csv")
        )
        return importance_df
