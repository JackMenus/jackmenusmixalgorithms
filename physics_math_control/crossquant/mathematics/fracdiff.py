"""
分数阶差分（Fractional Differentiation）
=======================================

学科来源
--------
分数阶微积分（Liouville 1832、Grünwald 1867、Letnikov 1868）：把 d 阶差分推广到实数阶。
算子 (1−B)^d 的二项式展开给出无穷长但衰减的权重序列。Hosking (1981) 用它定义 ARFIMA 长记忆过程；
López de Prado (2018) 把它用于“在最小破坏记忆的前提下让价格序列平稳”。

量化映射
--------
- 价格 → 平稳特征：整数阶差分（收益）平稳但抹掉全部记忆；分数阶 d∈(0,1) 在通过 ADF 检验的同时保留长记忆，
  因子挖掘用它作为输入可显著提高预测力（AFML 第 5 章）；
- ARFIMA 建模长记忆波动率；
- d 本身是“记忆强度”的指标（与 Hurst 指数 H = d + 0.5 对应于平稳 ARFIMA）。

核心公式
--------
(1−B)^d X_t = Σ_{k≥0} w_k X_{t−k},   w_0 = 1,  w_k = −w_{k−1} (d − k + 1)/k
固定宽度窗口（FFD）：截断 |w_k| < τ 的尾部，使变换对所有 t 使用同样的窗口，避免负漂移。
"""
from __future__ import annotations

import numpy as np

from .._utils import as_1d
from .timeseries import adf_test


def frac_diff_weights(d: float, size: int | None = None, threshold: float = 1e-5) -> np.ndarray:
    """分数阶差分权重 w_0..w_K。给定 size 时返回固定长度；否则按 |w_k| < threshold 截断。"""
    w = [1.0]
    k = 1
    while True:
        wk = -w[-1] * (d - k + 1) / k
        if size is not None:
            if k >= size:
                break
        elif abs(wk) < threshold:
            break
        w.append(wk); k += 1
        if k > 100000:
            break
    return np.array(w)


def frac_diff_ffd(x, d: float, threshold: float = 1e-5) -> np.ndarray:
    """固定宽度窗口分数阶差分（AFML Snippet 5.3）。前 K 个值为 nan（K = 窗口长度 − 1）。"""
    x = as_1d(x, "x")
    w = frac_diff_weights(d, threshold=threshold)[::-1]  # 对齐：w_K ... w_0
    K = len(w) - 1
    out = np.full(len(x), np.nan)
    if K >= len(x):
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, K + 1)
    out[K:] = windows @ w
    return out


def frac_diff_expanding(x, d: float, threshold: float = 1e-5) -> np.ndarray:
    """扩展窗口分数阶差分（全部历史权重，早期值因权重损失有偏）。"""
    x = as_1d(x, "x")
    n = len(x)
    w = frac_diff_weights(d, size=n)
    out = np.zeros(n)
    for t in range(n):
        out[t] = w[:t + 1] @ x[t::-1]
    skip = int(np.searchsorted(np.cumsum(np.abs(w[::-1]))[::-1] < threshold * np.abs(w).sum(), True)) if n > 1 else 0
    out[:skip] = np.nan
    return out


def find_min_d(x, d_grid=None, threshold: float = 1e-5, alpha: str = "5%", regression: str = "c") -> dict:
    """搜索最小的 d，使 FFD 序列通过 ADF 检验（在 alpha 水平拒绝单位根）。

    返回 ``d_min``、``table``（每个 d 的 ADF 统计量、与原序列的相关系数）。
    """
    x = as_1d(x, "x")
    d_grid = np.linspace(0, 1, 11) if d_grid is None else np.asarray(d_grid)
    rows = []
    d_min = None
    for d in d_grid:
        y = frac_diff_ffd(x, d, threshold)
        mask = ~np.isnan(y)
        if mask.sum() < 30:
            continue
        adf = adf_test(y[mask], regression=regression)
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        rows.append(dict(d=float(d), adf_stat=adf["stat"], reject=adf["reject"][alpha], corr_with_original=float(corr)))
        if d_min is None and adf["reject"][alpha]:
            d_min = float(d)
    return dict(d_min=d_min, table=rows)
