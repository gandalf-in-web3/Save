"""
LightGBM模型
"""

import copy
import os
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..ml.experiment import Experiment
from ..data import BinMinuteDataBase, MinuteData, t_cross_norm, long_ic


class LGBMExperiment2(Experiment):
    """
    LightGBM实验
    """

    params: Dict[str, Any] = {
        "num_threads": -1,
        "seed": 42,
        "deterministic": True,

        "objective": "regression",
        "metric": "None",

        "boosting": "gbdt",
        "num_iterations": 1500,
        "learning_rate": 0.05,

        "max_bin": 31,
        "min_data_in_bin": 1000,

        "max_depth": 12,
        "num_leaves": 511,
        "min_data_in_leaf": 10000,

        "bagging_freq": 5,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
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

        # 设置lgb的logger
        lgb.register_logger(self.logger)

        # 数据
        self.x_names: List[str] = None
        self.y_name: str = None

        self.train_dataset: lgb.Dataset | None = None

        self.valid_dataset: lgb.Dataset | None = None
        self.valid_y: MinuteData | None = None

        self.flatten_test_x: np.ndarray | None = None
        self.test_y: MinuteData | None = None

        # 模型
        self.model: lgb.Booster = None

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
        cross_norm_y: bool = True,
    ) -> None:
        """
        基于二进制数据库加载数据集
        """
        bin_db = BinMinuteDataBase(bin_folder)

        if isinstance(x_names, str):
            x_names = [x_names]
        self.x_names = x_names
        self.y_name = y_name

        # 训练集按照y中非nan的地方展平用于加速训练
        train_x, train_y = bin_db.read_flatten_dataset(
            x_names=x_names,
            y_name=y_name,
            date_slice=train_date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
            apply_func_y=(
                lambda x: t_cross_norm(x) if cross_norm_y else lambda x: x
            ),
        )
        self.train_dataset = lgb.Dataset(
            train_x,
            label=train_y,
            feature_name=self.x_names,
        )

        # 验证集无需去除nan用于计算截面IC
        valid_x, self.valid_y = bin_db.read_dataset(
            x_names=self.x_names,
            y_names=self.y_name,
            date_slice=valid_date_slice,
            minute_slice=minute_slice,
            tickers=tickers,
        )
        if cross_norm_y:
            self.valid_y = t_cross_norm(self.valid_y)

        self.valid_dataset = lgb.Dataset(
            valid_x.data.reshape(-1, len(self.x_names)),
            label=self.valid_y.data.reshape(-1),
            feature_name=self.x_names,
        )

        # 测试集无需去除nan用于计算截面IC
        if test_date_slice == valid_date_slice:
            test_x = valid_x
            self.test_y = self.valid_y
        else:
            test_x, self.test_y = bin_db.read_dataset(
                x_names=self.x_names,
                y_names=self.y_name,
                date_slice=test_date_slice,
                minute_slice=minute_slice,
                tickers=tickers,
            )
            if cross_norm_y:
                self.test_y = t_cross_norm(self.test_y)

        self.flatten_test_x = test_x.data.reshape(-1, len(self.x_names))

        self.logger.info(
            f"{len(train_y)} train, "
            f"{len(self.valid_y.data.reshape(-1))} valid, "
            f"{len(self.flatten_test_x)} test"
        )

    def train(self, ) -> None:
        """
        训练模型
        """
        self.model = lgb.train(
            self.params,
            self.train_dataset,
            valid_sets=self.valid_dataset,
            feval=self._valid,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=1),
            ],
        )
        self.model.save_model(
            os.path.join(self.folder, "model.txt"),
            num_iteration=self.model.best_iteration
        )

    def _valid(self, y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple:
        """
        在验证集上计算横截面IC
        """
        valid_y_pred: MinuteData = copy.copy(self.valid_y)
        valid_y_pred.data = y_pred.reshape(valid_y_pred.shape)
        return "long_ic", long_ic(self.valid_y, valid_y_pred), True

    def test(self, ) -> Dict[str, float]:
        """
        在测试集上计算横截面IC
        """
        self.model = lgb.Booster(
            model_file=os.path.join(self.folder, "model.txt")
        )

        test_y_pred: MinuteData = copy.copy(self.test_y)
        test_y_pred.data = self.model.predict(
            self.flatten_test_x,
            num_iteration=self.model.best_iteration
        ).reshape(self.test_y.shape)

        metric: Dict[str, float] = {
            "long_ic": long_ic(self.test_y, test_y_pred),
        }
        self.logger.info(f"metric: {metric}")
        return metric

    def compute_importance(self, ) -> pd.DataFrame:
        """
        根据训练结果计算重要度得分
        """
        feature_names = self.model.feature_name()

        # 计算gain得分
        gain_importance = self.model.feature_importance(importance_type="gain")

        # 计算shap得分
        # 采样100万样本用于计算
        rng = np.random.default_rng(42)
        n: int = len(self.flatten_test_x)
        idx: np.ndarray = rng.choice(n, min(1000000, n), replace=False)

        contrib = self.model.predict(
            self.flatten_test_x[idx],
            num_iteration=self.model.best_iteration,
            pred_contrib=True
        )
        shap_importance = np.mean(np.abs(contrib[:, : -1]), axis=0)

        importance_df = pd.DataFrame({
            "gain": gain_importance,
            "shap": shap_importance,
        }, index=feature_names)
        importance_df.to_csv(
            os.path.join(self.folder, "importance_df.csv")
        )
        return importance_df

    def compute_bins(self, ) -> Dict[str, List[float]]:
        """
        根据训练结果计算树的分裂分箱
        """
        df: pd.DataFrame = self.model.trees_to_dataframe()
        df = df[df["split_gain"].notnull()]
        df["threshold_value"] = pd.to_numeric(df["threshold"], errors="coerce")
        df = df.dropna(subset=["threshold_value"])

        x_names = list(self.model.feature_name())
        bins: Dict[str, List[float]] = {name: [] for name in x_names}

        for feat, g in df.groupby("split_feature"):
            if feat in bins:
                vals = np.unique(
                    g["threshold_value"].to_numpy(dtype=np.float32)
                )
                bins[feat] = vals.tolist()
        
        np.save(os.path.join(self.folder, "bins.npy"), bins, allow_pickle=True)
        return bins
