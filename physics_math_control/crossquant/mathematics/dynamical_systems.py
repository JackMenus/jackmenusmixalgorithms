"""
非线性动力系统与混沌：Takens 嵌入、相关维数、Lyapunov 指数、递归图
===================================================================

学科来源
--------
Takens (1981) 嵌入定理：标量观测的时延向量重构吸引子；Grassberger & Procaccia (1983) 的相关维数；
Wolf 等 (1985)、Rosenstein 等 (1993) 的最大 Lyapunov 指数估计；Eckmann 等 (1987) 的递归图。
这些是 1990 年代“金融混沌”研究的工具（Hsieh 1991；Brock 等的 BDS 检验）。

量化映射
--------
- **可预测性上界**：λ_max > 0 时预测视界 ~ 1/λ_max；λ≈0 则接近随机游走；
- **非线性结构检测**：相关维数远小于嵌入维数提示低维确定性结构（对收益序列通常不成立，对波动率/订单流有时成立）；
- **递归图量化（RQA）**：递归率、确定性等指标作为 regime 特征；
- **嵌入作为特征工程**：时延向量是机器学习模型最朴素的“路径特征”（与签名方法互补）。

参考文献
--------
- Kantz & Schreiber (2004). Nonlinear Time Series Analysis. CUP.
- Rosenstein, Collins & De Luca (1993). A practical method for calculating largest Lyapunov exponents from small data sets. Physica D.
- Marwan, Romano, Thiel & Kurths (2007). Recurrence plots for the analysis of complex systems. Physics Reports.
- Hsieh (1991). Chaos and Nonlinear Dynamics: Application to Financial Markets. JF.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .._utils import as_1d


def takens_embedding(x, dim: int = 3, tau: int = 1) -> np.ndarray:
    x = as_1d(x, "x")
    n = len(x) - (dim - 1) * tau
    return np.stack([x[i * tau:i * tau + n] for i in range(dim)], axis=1)


def correlation_dimension(X, r_list=None, n_r: int = 12) -> dict:
    """Grassberger–Procaccia 相关维数：C(r) ∝ r^D。返回 ``D``、``r``、``C``。"""
    X = np.asarray(X, dtype=float)
    tree = cKDTree(X)
    n = len(X)
    d_all = tree.query(X, k=2)[0][:, 1]
    r_list = np.logspace(np.log10(np.median(d_all)), np.log10(np.std(X) * 2), n_r) if r_list is None else np.asarray(r_list)
    C = np.array([(tree.count_neighbors(tree, r) - n) / (n * (n - 1)) for r in r_list])
    mask = C > 0
    D = np.polyfit(np.log(r_list[mask]), np.log(C[mask]), 1)[0]
    return dict(D=float(D), r=r_list, C=C)


def largest_lyapunov_rosenstein(x, dim: int = 3, tau: int = 1, min_tsep: int | None = None, max_t: int = 20) -> dict:
    """Rosenstein 最大 Lyapunov 指数（每个采样步）。返回 ``lambda_max``、``divergence``（平均 log 距离曲线）。"""
    x = as_1d(x, "x")
    X = takens_embedding(x, dim, tau)
    n = len(X)
    min_tsep = max(1, int(_mean_period(x))) if min_tsep is None else min_tsep
    tree = cKDTree(X[:n - max_t])
    dist, idx = tree.query(X[:n - max_t], k=min(n - max_t, 2 * min_tsep + 2))
    nn = np.full(n - max_t, -1)
    for i in range(n - max_t):
        for j in idx[i]:
            if abs(j - i) > min_tsep:
                nn[i] = j; break
    valid = nn >= 0
    ii = np.where(valid)[0]; jj = nn[valid]
    div = np.zeros(max_t + 1)
    for k in range(max_t + 1):
        d = np.linalg.norm(X[ii + k] - X[jj + k], axis=1)
        d = d[d > 0]
        div[k] = np.mean(np.log(d))
    ks = np.arange(max_t + 1)
    fit_range = slice(0, max(3, max_t // 2))
    lam = np.polyfit(ks[fit_range], div[fit_range], 1)[0]
    return dict(lambda_max=float(lam), divergence=div, k=ks)


def _mean_period(x):
    f = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x))
    f[0] = 0
    fm = np.sum(freqs * f) / np.sum(f)
    return 1 / fm if fm > 0 else 1.0


def recurrence_matrix(X, eps: float | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    D = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    eps = 0.1 * np.std(X) * np.sqrt(X.shape[1]) if eps is None else eps
    return (D <= eps).astype(int)


def rqa(R, l_min: int = 2) -> dict:
    """递归量化分析：递归率 RR、确定性 DET（对角线长度 ≥ l_min 的递归点占比）、平均对角线长度。"""
    R = np.asarray(R)
    n = R.shape[0]
    rr = (R.sum() - np.trace(R)) / (n * (n - 1))
    lengths = []
    for k in range(1, n):
        diag = np.diagonal(R, k)
        run = 0
        for v in diag:
            if v:
                run += 1
            elif run:
                lengths.append(run); run = 0
        if run:
            lengths.append(run)
    lengths = np.array(lengths)
    long_ = lengths[lengths >= l_min]
    det = long_.sum() / lengths.sum() if lengths.sum() else 0.0
    return dict(RR=float(rr), DET=float(det), L=float(long_.mean()) if len(long_) else 0.0)


def logistic_map(r: float = 4.0, n: int = 2000, x0: float = 0.3) -> np.ndarray:
    """混沌基准：逻辑斯蒂映射，r=4 时 λ = ln 2。"""
    x = np.zeros(n); x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x
