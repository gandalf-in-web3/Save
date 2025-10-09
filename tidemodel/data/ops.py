"""
基于分钟数据计算指标
"""

import cupy as cp
import numpy as np

from .data import MinuteData


"""
Numpy相关计算函数
"""

def np_ic(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    计算两个一维序列的皮尔逊相关系数

    自动忽略nan
    """

    assert x1.ndim == 1
    assert x2.shape == x1.shape

    mask: np.ndarray = np.isfinite(x1) & np.isfinite(x2)
    return np.corrcoef(x1[mask], x2[mask])[0, 1]


def np_long_ic(
    x1: np.ndarray,
    x2: np.ndarray,
    mean: bool = True,
) -> float | np.ndarray:
    """
    计算两个多维矩阵的皮尔逊相关系数

    除最后一维外视作计算一个ic点的序列

    自动忽略nan
    """

    assert x1.shape == x2.shape

    ics: np.ndarray = np.zeros(x1.shape[-1], dtype=np.float32)
    for idx in range(x1.shape[-1]):
        ics[idx] = np_ic(x1[..., idx].reshape(-1), x2[..., idx].reshape(-1))

    if mean:
        return np.nanmean(ics)
    return ics


def np_cross_ic(
    x1: np.ndarray,
    x2: np.ndarray,
    mean: bool = True,
) -> float | np.ndarray:
    """
    计算两个多维矩阵的皮尔逊相关系数

    最后一维视作计算一个ic点的序列

    自动忽略nan
    """

    assert x1.shape == x2.shape

    ics: np.ndarray = np.zeros(x1.shape[: -1], dtype=np.float32)
    for idx in np.ndindex(x1.shape[:-1]):
        ics[idx] = np_ic(x1[idx], x2[idx])

    if mean:
        return np.nanmean(ics)
    return ics


def np_mr(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    计算x2在x1上的缺失率
    """

    assert x1.ndim == 1
    assert x2.shape == x1.shape

    x1_mask: np.ndarray = np.isfinite(x1)
    return (x1_mask & (~np.isfinite(x2))).sum() / x1_mask.sum()


def np_cross_norm(x: np.ndarray, axis: int) -> np.ndarray:
    """
    对x做截面归一化

    自动忽略nan
    """

    mean: np.ndarray = np.nanmean(x, axis=axis, keepdims=True)
    std: np.ndarray = np.nanstd(x, axis=axis, keepdims=True)
    return (x - mean) / std


# def mutual_ic(x: np.ndarray, min_n: int = 1000) -> np.ndarray:
#     """
#     计算形状为(n, d)的n个样本的相关系数矩阵

#     当共同样本数小于1000时ic记为nan
#     """

#     assert x.ndim == 2

#     X: cp.ndarray = cp.asarray(x, dtype=cp.float32)
#     M: cp.ndarray = (~cp.isnan(X)).astype(cp.float32)
#     X0: cp.ndarray = cp.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

#     N: cp.ndarray = M.T @ M
#     S: cp.ndarray = X0.T @ M
#     P: cp.ndarray = X0.T @ X0
#     S2: cp.ndarray = (X0 ** 2).T @ M

#     N_safe = cp.maximum(N, 1.0)
#     mu_i: cp.ndarray = S / N_safe
#     mu_j: cp.ndarray = mu_i.T

#     cov: cp.ndarray = P / N_safe - mu_i * mu_j
#     var_i: cp.ndarray = S2 / N_safe - mu_i ** 2
#     var_j: cp.ndarray = var_i.T

#     denom = cp.sqrt(cp.maximum(var_i, 1e-8) * cp.maximum(var_j, 1e-8))
#     r = (cov / denom).astype(cp.float32)
#     r = cp.where(N >= min_n, r, cp.nan)
#     return cp.asnumpy(r)


"""
MinuteData相关计算函数

底层均基于numpy计算
"""

def ic(
    x1: MinuteData,
    x2: MinuteData,
) -> float:
    """
    计算两个分钟频数据的整体ic
    """
    return np_ic(
        x1=x1.data.reshape(-1),
        x2=x2.data.reshape(-1),
    )


def long_ic(
    x1: MinuteData,
    x2: MinuteData,
    mean: bool = True,
) -> np.ndarray | float:
    """
    计算两个分钟频数据的时序IC
    """
    assert x1.shape[-1] == 1
    assert x2.shape[-1] == 1

    return np_long_ic(
        x1=x1.data.squeeze(-1),
        x2=x2.data.squeeze(-1),
        mean=mean,
    )


def cross_ic(
    x1: MinuteData,
    x2: MinuteData,
    mean: bool = True,
) -> np.ndarray | float:
    """
    计算两个分钟频数据的截面IC
    """
    
    assert x1.shape[-1] == 1
    assert x2.shape[-1] == 1

    return np_cross_ic(
        x1=x1.data.squeeze(-1),
        x2=x2.data.squeeze(-1),
        mean=mean,
    )


def mr(x1: MinuteData, x2: MinuteData,) -> float:
    """
    计算两个分钟频数据的缺失率
    """

    return np_mr(x1.data.reshape(-1), x2.data.reshape(-1))


def cross_norm(x: MinuteData) -> MinuteData:
    """
    对分钟频数据做截面归一化

    自动忽略nan
    """
    
    x.data = np_cross_norm(x.data, axis=2)
    return x
