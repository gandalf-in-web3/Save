"""
多层感知机模型
"""

from typing import Any, Dict, List

import torch
import numpy as np
from torch import nn

from ..data import StatsDataBase
from ..dl import cross_rank, apply_split, nanmedian


class MLP(nn.Module):

    def __init__(
        self,
        dim: int,
        output_dim: int,
        x_min: np.ndarray,
        x_median: np.ndarray,
        x_max: np.ndarray,
        x_names: List[str] | None = None,
        with_rank: bool = False,
        rank_x_names: List[str] | None = None,
        with_lgb: bool = False,
        lgb_x_names: List[str] | None = None,
        lgb_bins: Dict[str, List[float]] | None = None,
    ) -> None:
        super().__init__()

        self.register_buffer("x_min", torch.tensor(x_min))
        self.register_buffer("x_median", torch.tensor(x_median))
        self.register_buffer("x_max", torch.tensor(x_max))

        self.dim: int = dim
        self.raw_dim: int = dim
        self.x_names: List[str] | None = x_names

        # rank预处理
        self.with_rank: bool = with_rank
        self.rank_x_names: List[int] | None = rank_x_names
        if self.with_rank:
            if self.rank_x_names is None:
                self.rank_x_slice: slice | list = slice(None)
                self.dim += self.raw_dim
            else:
                self.rank_x_slice = [
                    self.x_names.index(name) for name in self.rank_x_names
                ]
                self.dim += len(self.rank_x_names)

        # lgb预处理
        self.with_lgb: bool = with_lgb
        self.lgb_x_names: List[int] | None = lgb_x_names
        if self.with_lgb:
            if self.lgb_x_names is None:
                self.lgb_x_slice: slice | list = slice(None)
                self.dim += self.raw_dim
            else:
                self.lgb_x_slice = [
                    self.x_names.index(name) for name in self.lgb_x_names
                ]
                self.dim += len(self.lgb_x_names)

            # 分裂点个数padding到最长的并记录真实长度
            if self.lgb_x_names is None:
                lgb_bins = {name: lgb_bins[name] for name in self.x_names}
            else:
                lgb_bins = {name: lgb_bins[name] for name in self.lgb_x_names}
            
            max_bin: int = max((
                len(bins) for _, bins in lgb_bins.items()
            ), default=0)
            lgb_bins_pad: torch.Tensor = torch.zeros(
                (len(lgb_bins), max_bin), dtype=torch.float32
            )
            lgb_bins_len: torch.Tensor = torch.zeros(
                (len(lgb_bins), ), dtype=torch.int64
            )
    
            for i, bins in enumerate(list(lgb_bins.values())):
                if bins:
                    t = torch.tensor(bins, dtype=torch.float32)
                    lgb_bins_pad[i, : t.numel()] = t
                    lgb_bins_len[i] = t.numel()

            self.register_buffer("lgb_bins_pad", lgb_bins_pad)
            self.register_buffer("lgb_bins_len", lgb_bins_len)

        self.linear = nn.Sequential(
            nn.Linear(self.dim, 2048),
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

            nn.Linear(128, output_dim),
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

        if self.with_rank:
            x = torch.cat([
                x, cross_rank(data["x"][:, :, self.rank_x_slice], dim=1)
            ], dim=-1)

        if self.with_lgb:
            x = torch.cat([
                x, apply_split(
                    data["x"][:, :, self.lgb_x_slice],
                    self.lgb_bins_pad,
                    self.lgb_bins_len,
                )
            ], dim=-1)

        # 推理得到输出
        b, n, d = x.shape
        y_pred = self.linear(x.reshape(-1, d)).reshape(b, n, -1)
        
        # 对输出标准化
        mean: np.ndarray = y_pred.mean(dim=-2, keepdim=True)
        std: np.ndarray = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}


def get_mlp_model(
    x_names: List[str],
    y_names: List[str],
    x_norm_stats_file: str,
    model_cls: nn.Module = MLP,
    with_rank: bool = False,
) -> nn.Module:
    stats_db = StatsDataBase(x_norm_stats_file)
    x_min: np.ndarray = stats_db.get_stats("x_1", x_names)
    x_median: np.ndarray = stats_db.get_stats("x_50", x_names)
    x_max: np.ndarray = stats_db.get_stats("x_99", x_names)

    return model_cls(
        dim=len(x_min),
        output_dim=len(y_names),
        x_min=x_min,
        x_median=x_median,
        x_max=x_max,
        with_rank=with_rank,
    )
