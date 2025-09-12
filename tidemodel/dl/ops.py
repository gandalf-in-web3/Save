"""
自定义tensor运算函数
"""

import torch
import torch.nn.functional as F


def mae_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,    
) -> torch.Tensor:
    """
    计算两个二维tensor之间的L1损失函数

    自动忽略nan
    """

    assert y.shape == y_pred.shape

    mask: torch.Tensor = ~torch.isnan(y)
    return F.l1_loss(y[mask], y_pred[mask])


def mse_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,    
) -> torch.Tensor:
    """
    计算两个二维tensor之间的L2损失函数

    自动忽略nan
    """

    assert y.shape == y_pred.shape

    mask: torch.Tensor = ~torch.isnan(y)
    return F.mse_loss(y[mask], y_pred[mask])


def cross_ic(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    mean: bool = True,
) -> torch.Tensor:
    """
    计算两个二维tensor之间的横截面ic

    形状均为(b, n)

    自动忽略nan
    """

    assert y.ndim == 2
    assert y.shape == y_pred.shape

    mask: torch.Tensor = ~torch.isnan(y)
    y = torch.nan_to_num(y, nan=0.0)

    # 计算没有被mask掉的均值
    count: torch.Tensor = torch.sum(mask, dim=1, keepdim=True)
    y_pred_mean: torch.Tensor = torch.sum(
        (y_pred * mask), dim=1, keepdim=True
    ) / (count + 1e-8)
    y_mean: torch.Tensor = torch.sum(
        (y * mask), dim=1, keepdim=True
    ) / (count + 1e-8)

    # 计算协方差和方差
    y_pred_center: torch.Tensor = (y_pred - y_pred_mean) * mask
    y_center: torch.Tensor = (y - y_mean) * mask
    cov: torch.Tensor = torch.sum(y_pred_center * y_center, dim=1)
    y_pred_var: torch.Tensor = torch.sum(y_pred_center * y_pred_center, dim=1)
    y_var: torch.Tensor = torch.sum(y_center * y_center, dim=1)

    # 计算相关系数
    ic: torch.Tensor = cov / (torch.sqrt(y_pred_var * y_var) + 1e-8)
    if mean:
        return torch.mean(ic)
    return ic


def cross_ic_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """
    计算两个二维tensor之间的横截面ic损失函数

    形状均为(b, n)

    自动忽略nan
    """

    return -cross_ic(y, y_pred)


def emd_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,
) -> torch.Tensor:
    """
    计算两个二维矩阵之间的EMD loss

    用于衡量两个分布之间的差异, 具有排序不变性
    """

    assert y.ndim == 2
    assert y.shape == y_pred.shape

    y, _ = torch.sort(y, dim=1)
    y_pred, _ = torch.sort(y_pred, dim=1)
    return mae_loss(y, y_pred)
