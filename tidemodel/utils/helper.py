"""
辅助编写代码
"""

from typing import List

import torch
from torch import nn


class DummyClass:
    """
    作为占位符的类, 执行任何方法都不报错
    """

    def __getattr__(self, name: str) -> "DummyClass":
        return self
    
    def __call__(self, *args, **kwargs) -> None:
        pass


def get_non_finite_params(model: nn.Module) -> bool:
    """
    返回模型中梯度不在有限范围内的参数
    """

    non_finite_params: List[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        if not torch.isfinite(param.grad).all():
            non_finite_params.append((name, param.grad))
    return non_finite_params
