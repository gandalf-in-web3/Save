"""
用于因子选择的GFSNetwork

https://arxiv.org/pdf/2503.13304
"""

from typing import Any, Dict, List

import torch
import numpy as np
from torch import nn

from ..data import StatsDataBase
from ..dl import DLExperiment, cross_ic_loss, nanmedian


class GumbelSigmoidGate(nn.Module):
    """
    基于Gumbel Sigmoid的特征门控

    模型参数遵循论文中的原始参数
    """

    def __init__(
        self,
        dim: int,
        tau: float = 2.0,
        min_tau: float = 0.1,
        alpha: float = 0.9,
        keep_ratio: float = 0.03,
    ) -> None:
        super().__init__()

        self.tau: float = tau
        self.min_tau: float = min_tau
        self.alpha: float = alpha
        self.keep_ratio: float = keep_ratio

        self.logits = nn.Parameter(torch.zeros(dim))
        self.hard: bool = False

    def get_importance(self, ) -> torch.Tensor:
        """
        获取特征的重要性得分
        """
        return torch.sigmoid(self.logits)

    def decay_tau(self, ) -> None:
        """
        按照一定的比例衰减tau
        """
        self.tau = max(self.tau * self.alpha, self.min_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == len(self.logits)

        if self.training and not self.hard:
            gumbels: torch.Tensor = -torch.empty_like(
                self.logits
            ).exponential_().log()
            gumbels = (self.logits + gumbels) / self.tau
            mask_soft: torch.Tensor = gumbels.sigmoid()
        else:
            mask_soft = (self.logits / self.tau).sigmoid()

        if self.hard:
            q: torch.Tensor = torch.quantile(mask_soft, 1.0 - self.keep_ratio)
            mask_hard = (mask_soft >= q).float()
            mask: torch.Tensor = mask_hard - mask_soft.detach() + mask_soft
        else:
            mask = mask_soft

        mask = mask.view(*([1] * (x.dim() - 1)), -1)
        return x * mask, mask_soft


class GFS(nn.Module):

    def __init__(
        self,
        dim: int,
        x_min: np.ndarray,
        x_median: np.ndarray,
        x_max: np.ndarray,
        x_names: List[str] | None = None,
        weight: float = 0.1,
        keep_ratio: float = 0.03,
    ) -> None:
        super().__init__()

        self.x_names: List[str] | None = x_names
        self.weight: float = weight

        self.register_buffer("x_min", torch.tensor(x_min))
        self.register_buffer("x_median", torch.tensor(x_median))
        self.register_buffer("x_max", torch.tensor(x_max))

        self.keep_ratio: float = keep_ratio
        self.gate = GumbelSigmoidGate(dim, keep_ratio=keep_ratio)

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
            # self.x_median.reshape(1, 1, -1),
            nanmedian(x, dim=1, keepdim=True)[0],
            x,
        )
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.clamp(x, min=0, max=1)

        # 如果min和max中包含nan的话，那么会出现新的nan
        x[torch.isnan(x)] = 0.0

        # 推理得到输出
        b, n, d = x.shape
        x, mask = self.gate(x)
        y_pred = self.linear(x.reshape(-1, d)).reshape(b, n, -1)
        
        # 对输出标准化
        mean: np.ndarray = y_pred.mean(dim=-2, keepdim=True)
        std: np.ndarray = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred, "mask": mask}

    def loss_func(
        self,
        exp: DLExperiment,
        data: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ic_loss = cross_ic_loss(data["y"], output["y_pred"], dim=-2)

        if self.gate.hard:
            lambda_loss: torch.Tensor = torch.zeros_like(output["mask"].mean())
        else:
            lambda_loss = torch.relu(output["mask"].mean() - self.keep_ratio)

        return {
            "loss": ic_loss + 0.2 * lambda_loss,
            "ic_loss": ic_loss,
            "lambda_loss": lambda_loss,
        }

    def on_epoch_end(self, exp: DLExperiment) -> None:
        if exp.epoch == 2:
            self.gate.hard = True
            exp.logger.info("change gate to hard")

        self.gate.decay_tau()
        exp.logger.info(f"change tau to be {self.gate.tau}")


def get_gfs_model(
    x_names: List[str],
    y_names: List[str],
    x_norm_stats_file: str,
    model_cls: nn.Module = GFS,
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
        x_names=x_names,
    )
