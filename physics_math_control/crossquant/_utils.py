"""内部工具函数（输入检查、随机数、滑窗等）。"""
from __future__ import annotations

import numpy as np


def as_rng(seed_or_rng=None) -> np.random.Generator:
    """把 ``None`` / 整数 / ``Generator`` 统一转换成 ``numpy.random.Generator``。"""
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def as_1d(x, name: str = "x") -> np.ndarray:
    """转换为一维 float 数组，不满足则报错。"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1) if arr.size == max(arr.shape, default=0) else arr
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际形状 {arr.shape}")
    return arr


def as_2d(x, name: str = "X") -> np.ndarray:
    """转换为二维 float 数组 (T, n)；一维输入视为单列。"""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，实际形状 {arr.shape}")
    return arr


def sliding_windows(x: np.ndarray, window: int, step: int = 1) -> np.ndarray:
    """返回形状 ``(n_windows, window)`` 的滑窗视图（不复制数据）。"""
    x = np.asarray(x)
    if window > x.shape[0]:
        raise ValueError("window 大于序列长度")
    view = np.lib.stride_tricks.sliding_window_view(x, window, axis=0)
    return view[::step]


def zscore(x: np.ndarray, axis: int = 0, ddof: int = 1) -> np.ndarray:
    """标准化：减均值、除标准差。"""
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, ddof=ddof, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def ensure_spd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """把对称矩阵投影到最近的对称正定矩阵（特征值下限截断）。"""
    mat = 0.5 * (mat + mat.T)
    w, v = np.linalg.eigh(mat)
    w = np.maximum(w, eps)
    return (v * w) @ v.T


def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    """协方差矩阵 → 相关系数矩阵。"""
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def corr_to_cov(corr: np.ndarray, std: np.ndarray) -> np.ndarray:
    """相关系数矩阵 + 标准差向量 → 协方差矩阵。"""
    std = np.asarray(std, dtype=float)
    return corr * np.outer(std, std)
