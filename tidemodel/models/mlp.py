"""
多层感知机模型
"""

from typing import Any, Dict, List

import torch
import numpy as np
from torch import nn

from ..data import StatsDataBase


class MLP(nn.Module):

    def __init__(
        self,
        dim: int,
        x_min: np.ndarray,
        x_median: np.ndarray,
        x_max: np.ndarray,
    ) -> None:
        super(MLP, self).__init__()

        self.register_buffer("x_min", torch.tensor(x_min))
        self.register_buffer("x_median", torch.tensor(x_median))
        self.register_buffer("x_max", torch.tensor(x_max))

        self.linear = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.7),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 1),
        )

        self._init_weight()

    def _init_weight(self, ) -> None:
        """
        自定义初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_uniform_(
                        m.weight,
                        a=0.01,
                        mode="fan_in",
                        nonlinearity="leaky_relu",
                    )
                    nn.init.constant_(m.bias, 0)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # 预处理x: (b, n, d)
        x: torch.Tensor = data["x"]
        x = torch.where(
            torch.isnan(x),
            self.x_median.reshape(1, 1, -1),
            x,
        )
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.clamp(x, min=0, max=1)

        # 如果min和max中包含nan的话，那么会出现新的nan
        x[torch.isnan(x)] = 0.0

        # (b, n, d)
        ori_shape = x.shape

        x = x.reshape(-1, ori_shape[-1])
        y_pred = self.linear(x)
        y_pred = y_pred.reshape(*ori_shape[: -1], -1)

        mean: np.ndarray = y_pred.mean(dim=-2, keepdim=True)
        std: np.ndarray = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}


def get_mlp_model(
    x_names: List[str],
    x_norm_stats_file: str,
    model_cls: nn.Module = MLP,
) -> nn.Module:
    stats_db = StatsDataBase(x_norm_stats_file)
    x_min: np.ndarray = stats_db.get_stats("x_05", x_names)
    x_median: np.ndarray = stats_db.get_stats("x_50", x_names)
    x_max: np.ndarray = stats_db.get_stats("x_995", x_names)

    return model_cls(
        dim=len(x_min),
        x_min=x_min,
        x_median=x_median,
        x_max=x_max,
    )
