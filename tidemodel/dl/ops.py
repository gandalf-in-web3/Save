"""
自定义tensor运算函数
"""

from typing import Tuple

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
    dim: int,
    mean: bool = True,
) -> torch.Tensor:
    """
    计算两个tensor之间的横截面ic

    指定维度上代表计算ic的一个序列

    自动忽略nan
    """
    assert y.shape == y_pred.shape, f"{y.shape}, {y_pred.shape}"

    mask: torch.Tensor = ~torch.isnan(y)
    y = torch.nan_to_num(y, nan=0.0)

    # 计算没有被mask掉的均值
    count: torch.Tensor = torch.sum(mask, dim=dim, keepdim=True)
    y_mean: torch.Tensor = torch.sum(
        (y * mask), dim=dim, keepdim=True
    ) / (count + 1e-8)
    y_pred_mean: torch.Tensor = torch.sum(
        (y_pred * mask), dim=dim, keepdim=True
    ) / (count + 1e-8)

    # 计算协方差和方差
    y_center: torch.Tensor = (y - y_mean) * mask
    y_pred_center: torch.Tensor = (y_pred - y_pred_mean) * mask
    cov: torch.Tensor = torch.sum(y_center * y_pred_center, dim=dim)
    y_var: torch.Tensor = torch.sum(y_center * y_center, dim=dim)
    y_pred_var: torch.Tensor = torch.sum(y_pred_center * y_pred_center, dim=dim)

    # 计算相关系数
    # 将eps写在sqrt中防止梯度为nan
    ic: torch.Tensor = cov / (torch.sqrt((y_var + 1e-8) * (y_pred_var + 1e-8)))
    if mean:
        return torch.nanmean(ic)
    return ic


def cross_ic_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    计算两个tensor之间的横截面ic损失函数

    指定维度上代表计算ic的一个序列

    自动忽略nan
    """

    return -cross_ic(y, y_pred, dim=dim)


def soft_rank(
    x: torch.Tensor,
    n_bins: int = 4800,
    temp: float = 1e-4,
) -> torch.Tensor:
    """
    通过softmax近似的可导rank
    """

    assert x.ndim == 1
    if x.numel() == 0:
        return x

    # 计算等频分箱分位数点
    quantiles = torch.linspace(0, 1, n_bins + 1, device=x.device)[1: -1]
    bin_edges = torch.quantile(x.detach(), quantiles)

    # 添加最小值和最大值
    min_val = x.min()
    max_val = x.max()
    bin_edges = torch.cat([min_val.unsqueeze(0), bin_edges, max_val.unsqueeze(0)])
    
    # 计算每个点到所有边界的距离
    diffs = x.unsqueeze(1) - bin_edges.unsqueeze(0)
    
    # 使用softmax计算每个点属于每个区间的概率
    probs = torch.softmax(-diffs.abs() / temp, dim=1)

    # 计算每个点的软秩次
    bin_indices = torch.arange(n_bins + 1, device=x.device).float()
    soft_ranks = (probs * bin_indices).sum(dim=1)

    # 归一化到[0, 1]范围
    soft_ranks = soft_ranks / n_bins
    return soft_ranks


def cross_rank_ic_loss(
    y: torch.Tensor,
    y_pred: torch.Tensor,
    n_bins: int = 4800,
    temp: float = 1e-4,
) -> torch.Tensor:
    """
    计算两个tensor之间的近似横截面rank ic损失函数

    y的形状为两维度, 最后一维rank ic的一个序列

    自动忽略nan
    """

    assert y.ndim == 2
    assert y.shape == y_pred.shape

    loss: torch.Tensor = 0
    mask: torch.Tensor = ~torch.isnan(y)

    for i in range(y.shape[0]):
        _y: torch.Tensor = y[i][mask[i]]
        _y_pred: torch.Tensor = y_pred[i][mask[i]]

        _y = soft_rank(_y, n_bins=n_bins, temp=temp)
        _y_pred = soft_rank(_y_pred, n_bins=n_bins, temp=temp)
        loss += cross_ic_loss(_y, _y_pred, dim=-1)

    loss = loss / y.shape[0]
    return loss


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


def weighted_mean(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    返回tensor的加权平均数
    """

    assert x.ndim == 1
    assert x.shape == weight.shape
    return torch.sum(x * weight) / torch.sum(weight)


@torch.no_grad()
def cross_rank(
    x: torch.Tensor,
    dim: int,
    bins: int | None = None,
) -> torch.Tensor:
    """
    计算x在dim维度的截面排名并归一化

    nan会被填充为0.5
    """

    mask: torch.Tensor = ~torch.isnan(x)
    # 将nan填充为inf, 使其不影响其他值的排序
    x = torch.where(mask, x, torch.full_like(x, float("inf")))

    # 计算rank
    _, order = torch.topk(x, k=x.size(dim), dim=dim, sorted=True, largest=False)
    rank: torch.Tensor = torch.empty_like(order)
    shape: Tuple[int] = [1] * x.ndim
    shape[dim] = -1
    idx = torch.arange(x.shape[dim], device=x.device).view(*shape).expand_as(order)
    rank.scatter_(dim, order, idx)

    # 将rank归一化到0-1
    cnt: torch.Tensor = mask.sum(dim=dim, keepdim=True).clamp(min=1)
    rank = (rank.float() + 0.5) / cnt.float()
    rank = torch.where(mask, rank, torch.full_like(rank, 0.5))

    # 分箱
    if bins is not None:
        bin_idx: torch.Tensor = torch.clamp((rank * bins).floor(), min=0, max=bins - 1)
        rank = (bin_idx + 0.5) / bins
    return rank.clamp_(0.0, 1.0)


@torch.no_grad()
def apply_split(
    x: torch.Tensor,
    bins_pad: torch.Tensor,
    bins_len: torch.Tensor,
) -> torch.Tensor:
    F: int = x.shape[-1]
    assert F == bins_pad.shape[0] == bins_len.shape[0]

    mask: torch.Tensor = torch.isfinite(x)
    x_safe: torch.Tensor = torch.where(mask, x, torch.zeros_like(x))

    # 阈值扩展
    # [..., F, T]
    thr: torch.Tensor = bins_pad.view(
        (1, ) * (x_safe.dim() - 1) + bins_pad.shape
    )
    T: int = bins_pad.shape[1]
    valid: torch.Tensor = torch.arange(
        T, device=x.device
    )[None, :] < bins_len[:, None]
    valid = valid.view((1, ) * (x_safe.dim() - 1) + valid.shape)

    # 计算bin index
    idx: torch.Tensor = ((x_safe.unsqueeze(-1) > thr) & valid).sum(dim=-1)
    length: torch.Tensor = bins_len.clamp_min(1).view(
        (1, ) * (idx.dim() - 1) + (F, )
    )
    x_split: torch.Tensor = idx.float() / length
    return torch.where(mask, x_split, torch.zeros_like(x_split))


def nanmedian(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
) -> torch.Tensor:
    nan = torch.isnan(x)
    filled = torch.where(nan, torch.full_like(x, float('inf')), x)
    sorted_vals, sorted_idx = torch.sort(filled, dim=dim)

    valid = (~nan).sum(dim=dim, keepdim=True)
    kth = ((valid - 1) // 2).clamp_min(0).to(torch.long)

    vals = torch.gather(sorted_vals, dim, kth)
    idxs = torch.gather(sorted_idx, dim, kth)

    vals = torch.where(valid > 0, vals, torch.full_like(vals, float('nan')))
    idxs = torch.where(valid > 0, idxs, torch.zeros_like(idxs))

    if not keepdim:
        vals = vals.squeeze(dim)
        idxs = idxs.squeeze(dim)
    return vals, idxs


def skew_loss(
    x: torch.Tensor,
    dim: int = -1,
    target: float = 0.0,
    p: int = 2,
) -> torch.Tensor:
    """
    计算在dim维度上的偏度损失, 默认通过mean聚合
    """

    mean: torch.Tensor = x.mean(dim=dim, keepdim=True)
    diff: torch.Tensor = x - mean
    var: torch.Tensor = (diff ** 2).mean(dim=dim, keepdim=True)
    std: torch.Tensor = (var + 1e-8).sqrt()
    skew = (diff / std).pow(3).mean(dim=dim)

    if p == 1:
        skew_loss = (skew - target).abs().mean()
    else:
        skew_loss = (skew - target).pow(2).mean()
    return skew_loss
