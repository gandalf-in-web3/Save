"""
模型子模块
"""

import torch
from torch import nn


class StackLinear(nn.Module):
    """
    带有残差连接的多层线性层
    """

    def __init__(self, dim: int, stack_num: int) -> None:
        super().__init__()
        self.linears = nn.ModuleList()
        for _ in range(stack_num):
            self.linears.append(nn.Linear(dim, dim))
            self.linears.append(nn.LeakyReLU())

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.linears)):
            new_feature: torch.Tensor = self.linears[i](feature)
            feature = new_feature + feature
        return feature
