"""
Deep Lasso模型用于因子选择
"""

from typing import Any, Dict, List

import torch
import numpy as np
from torch import nn

from ..data import StatsDataBase
from ..dl import DLExperiment, cross_ic_loss, nanmedian


class DeepLasso(nn.Module):
    def __init__(
        self,
        dim: int,
        x_min: np.ndarray,
        x_median: np.ndarray,
        x_max: np.ndarray,
        x_names: List[str] | None = None,
        weight: float = 0.1,
    ) -> None:
        super().__init__()

        self.x_names: List[str] | None = x_names
        self.weight: float = weight

        self.register_buffer("x_min", torch.tensor(x_min))
        self.register_buffer("x_median", torch.tensor(x_median))
        self.register_buffer("x_max", torch.tensor(x_max))

        self.linear = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(0.8),

            nn.Linear(2048, 1024),
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

    def _init_weight(self,) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_uniform_(m.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
                    nn.init.constant_(m.bias, 0)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # 预处理
        x: torch.Tensor = data["x"]
        x = torch.where(
            torch.isnan(x),
            nanmedian(x, dim=1, keepdim=True)[0],
            x,
        )
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.clamp(x, min=0, max=1)
        x[torch.isnan(x)] = 0.0

        # 让标准化后的x参与 Deep Lasso 求梯度
        x = x.detach().requires_grad_(True)

        # 前向推导
        b, n, d = x.shape
        y_pred = self.linear(x.reshape(-1, d)).reshape(b, n, -1)

        # 对输出做标准化
        mean: np.ndarray = y_pred.mean(dim=-2, keepdim=True)
        std:  np.ndarray = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"x": x, "y_pred": y_pred}

    def loss_func(
        self,
        exp: DLExperiment,
        data: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ic_loss = cross_ic_loss(data["y"], output["y_pred"], dim=-2)

        if self.training:
            # Deep Lasso reg loss: sum_j || d(ic_loss)/d x^{(j)} ||_2
            grads = torch.autograd.grad(
                ic_loss, output["x"], create_graph=True, only_inputs=True
            )[0]    
            reg_loss = torch.linalg.vector_norm(grads, ord=2, dim=(0, 1)).sum()
        else:
            reg_loss = torch.tensor(0.0, device=ic_loss.device)

        return {
            "loss": ic_loss + self.weight * reg_loss,
            "ic_loss": ic_loss,
            "reg_loss": reg_loss,
        }


def get_deep_lasso_model(
    x_names: List[str],
    y_names: List[str],
    x_norm_stats_file: str,
    model_cls: nn.Module = DeepLasso,
) -> nn.Module:
    stats_db = StatsDataBase(x_norm_stats_file)
    x_min: np.ndarray = stats_db.get_stats("x_1", x_names)
    x_median: np.ndarray = stats_db.get_stats("x_50", x_names)
    x_max: np.ndarray = stats_db.get_stats("x_99", x_names)

    return model_cls(
        dim=len(x_min),
        x_min=x_min,
        x_median=x_median,
        x_max=x_max,
    )
