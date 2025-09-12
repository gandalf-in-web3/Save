"""
RNN模型及其变种
"""

from typing import Any, Dict, List, Literal

import torch
import numpy as np
from torch import nn

from ..data import StatsDataBase


class RNN(nn.Module):

    def __init__(
        self,
        dim: int,
        x_min: np.ndarray,
        x_median: np.ndarray,
        x_max: np.ndarray,
        mode: Literal["rnn", "gru", "lstm"] = "gru",
        num_layers: int = 1,
    ) -> None:
        super(RNN, self).__init__()

        self.register_buffer("x_min", torch.tensor(x_min))
        self.register_buffer("x_median", torch.tensor(x_median))
        self.register_buffer("x_max", torch.tensor(x_max))

        if mode == "rnn":
            rnn_cls = nn.RNN
        elif mode == "gru":
            rnn_cls = nn.GRU
        elif mode == "lstm":
            rnn_cls = nn.LSTM
        else:
            raise ValueError(f"{mode} is not supported")

        self.rnn = rnn_cls(
            input_size=dim, 
            hidden_size=512, 
            num_layers=num_layers,
            dropout=0.5,
            batch_first=True,
        )
        self.gru_linear = nn.Sequential(
            nn.Linear(512, 512),
        )

        self.linear1 = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.7),

            nn.Linear(1024, 512),
        )

        self.linear2 = nn.Sequential(
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

            if isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "weight_hr" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)
                        if "bias_ih" in name:
                            hs = m.hidden_size
                            p.data[hs:2*hs].fill_(1.0)

            if isinstance(m, nn.GRU):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

            if isinstance(m, nn.RNN):
                for name, p in m.named_parameters():
                    if "weight_ih" in name:
                        if m.nonlinearity == "relu":
                            nn.init.kaiming_uniform_(p, nonlinearity="relu")
                        else:
                            nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

    def forward(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # 预处理x: (b, t, n, d)
        x: torch.Tensor = data["x"]
        x = torch.where(
            torch.isnan(x),
            self.x_median.reshape(1, 1, 1, -1),
            x,
        )
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = torch.clamp(x, min=0, max=1)

        # 如果min和max中包含nan的话，那么会出现新的nan
        x[torch.isnan(x)] = 0.0

        # 推理得到输出
        b, t, n, d = x.shape

        x_rnn, _ = self.rnn(x.transpose(1, 2).reshape(b * n, t, -1))
        x_rnn = self.gru_linear(x_rnn[:, -1, :]).reshape(b, n, -1)

        x = self.linear1(x.reshape(-1, d)).reshape(b, t, n, -1)
        x = x[:, -1, :, :] + x_rnn

        y_pred = self.linear2(x.reshape(b * n, -1)).reshape(b, n, -1)

        # 对输出标准化
        mean: np.ndarray = y_pred.mean(dim=-2, keepdim=True)
        std: np.ndarray = y_pred.std(dim=-2, keepdim=True)
        y_pred = (y_pred - mean) / std
        return {"y_pred": y_pred}


def get_rnn_model(
    x_names: List[str],
    x_norm_stats_file: str,
    model_cls: nn.Module = RNN,
    mode: Literal["rnn", "gru", "lstm"] = "gru",
    num_layers: int = 1,
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
        mode=mode,
        num_layers=num_layers,
    )
