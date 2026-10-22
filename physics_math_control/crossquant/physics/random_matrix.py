"""
随机矩阵理论（RMT）：协方差/相关矩阵去噪
=========================================

学科来源
--------
核物理（Wigner 1955 用随机矩阵描述重原子核能级），Marchenko & Pastur (1967) 给出大维样本协方差
特征值的极限分布。Laloux, Cizeau, Bouchaud & Potters (1999) 与 Plerou 等 (1999) 把它引入金融：
股票相关矩阵的绝大多数特征值落在 MP 分布的“噪声带”里，只有少数（市场、行业）是信号。

量化映射
--------
- **协方差去噪**：组合优化对协方差的噪声极其敏感（Markowitz 的“误差最大化”），
  去噪后权重更稳、样本外风险更低。
- **信号因子个数**：超过 λ₊ 的特征值个数 = 统计上显著的因子数。
- **系统性风险指标**：最大特征值占比（吸收比，Kritzman 等 2011）升高 ⇒ 市场趋于一体化/脆弱。

核心公式
--------
q = N/T（变量数/样本数），纯噪声相关矩阵特征值密度（MP 律）：
    ρ(λ) = sqrt((λ₊ − λ)(λ − λ₋)) / (2π q σ² λ),   λ± = σ²(1 ± √q)²
特征值裁剪（Laloux 等）：λ_i < λ₊ 的特征值替换为平均值（保迹）。
旋转不变估计（RIE，Ledoit & Péché 2011；Bun, Bouchaud & Potters 2017）：
    ξ_k = λ_k / |1 − q + q z_k g(z_k)|²,  z_k = λ_k − iη,  g(z) = (1/N)Σ_{j≠k} 1/(z − λ_j)
它在“保持样本特征向量”的估计量里渐近最优（Frobenius 意义）。
Ledoit–Wolf (2004) 线性收缩：Σ̂ = (1−δ) S + δ (tr S / N) I，δ 由样本闭式给出。

参考文献
--------
- Marchenko & Pastur (1967). Distribution of eigenvalues for some sets of random matrices.
- Laloux, Cizeau, Bouchaud & Potters (1999). Noise Dressing of Financial Correlation Matrices. PRL.
- Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional covariance matrices. JMVA.
- Bun, Bouchaud & Potters (2017). Cleaning large correlation matrices: tools from RMT. Physics Reports. arXiv:1610.08104
- Benaych-Georges, Bouchaud & Potters (2023). Optimal cleaning for singular values of cross-covariance matrices. arXiv:1901.05543
- Bongiorno & Challet (2025). Quantifying the information lost in optimal covariance matrix cleaning. Physica A. arXiv:2310.01963
- López de Prado (2020). Machine Learning for Asset Managers, Ch. 2.
"""
from __future__ import annotations

import numpy as np

from .._utils import as_2d, cov_to_corr


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """MP 律的特征值上下界 λ± = σ²(1 ± √q)²。"""
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def marchenko_pastur_pdf(x, q: float, sigma2: float = 1.0) -> np.ndarray:
    """MP 密度（q = N/T < 1 时无点质量）。"""
    x = np.asarray(x, dtype=float)
    lo, hi = mp_bounds(q, sigma2)
    pdf = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    pdf[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (2 * np.pi * q * sigma2 * x[inside])
    return pdf


def fit_mp_sigma2(eigs, q: float, grid=None, n_bins: int = 50) -> dict:
    """在噪声带上拟合 MP 律的 σ²（最小化直方图与密度的平方误差）。

    信号特征值会抬高样本方差，因此 σ² 通常 < 1；返回 ``sigma2``、``lambda_plus``、``n_signal``（> λ₊ 的个数）。
    """
    eigs = np.sort(np.asarray(eigs, dtype=float))
    grid = np.linspace(0.2, 1.2, 101) if grid is None else np.asarray(grid)
    best, best_err = 1.0, np.inf
    for s2 in grid:
        lo, hi = mp_bounds(q, s2)
        noise = eigs[eigs <= hi]
        if len(noise) < 5:
            continue
        hist, edges = np.histogram(noise, bins=n_bins, range=(0, hi * 1.05), density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        # 直方图按噪声特征值个数归一；密度需同比例缩放
        pdf = marchenko_pastur_pdf(centers, q, s2)
        err = np.mean((hist - pdf) ** 2)
        if err < best_err:
            best, best_err = s2, err
    lo, hi = mp_bounds(q, best)
    return dict(sigma2=float(best), lambda_minus=lo, lambda_plus=hi, n_signal=int(np.sum(eigs > hi)))


def clip_eigenvalues(corr, q: float, sigma2: float | None = None, n_signal: int | None = None) -> dict:
    """特征值裁剪去噪（Laloux 等 1999 / López de Prado 2020）。

    噪声带内（λ < λ₊）的特征值替换为它们的均值（保迹），再重新归一为相关矩阵。
    返回 ``corr_clean``、``eigs``、``lambda_plus``、``n_signal``。
    """
    corr = np.asarray(corr, dtype=float)
    w, V = np.linalg.eigh(corr)
    idx = np.argsort(w)[::-1]; w, V = w[idx], V[:, idx]
    if n_signal is None:
        fit = fit_mp_sigma2(w, q) if sigma2 is None else dict(lambda_plus=mp_bounds(q, sigma2)[1])
        lam_plus = fit["lambda_plus"]
        n_signal = int(np.sum(w > lam_plus))
    else:
        lam_plus = w[n_signal - 1] if n_signal > 0 else w[0]
    w_clean = w.copy()
    if n_signal < len(w):
        w_clean[n_signal:] = w[n_signal:].mean()
    C = (V * w_clean) @ V.T
    C = cov_to_corr(C)
    return dict(corr_clean=C, eigs=w, eigs_clean=w_clean, lambda_plus=float(lam_plus), n_signal=n_signal)


def rie_cleaning(corr, q: float, eta: float | None = None, isotonic: bool = True) -> dict:
    """旋转不变估计（RIE / Ledoit–Péché / Bun–Bouchaud–Potters）去噪相关矩阵。

    参数
    ----
    corr : 样本相关矩阵
    q : N/T
    eta : Stieltjes 变换的虚部正则化，缺省 N^{-1/2}
    isotonic : 是否对去噪特征值做单调回归（保证 ξ 随 λ 单调，抑制小特征值处的数值噪声）

    返回 ``corr_clean``、``xi``（去噪特征值，已保迹）、``eigs``。
    """
    corr = np.asarray(corr, dtype=float)
    N = corr.shape[0]
    eta = N ** -0.5 if eta is None else eta
    w, V = np.linalg.eigh(corr)
    xi = np.zeros(N)
    for k in range(N):
        z = w[k] - 1j * eta
        others = np.delete(w, k)
        g = np.mean(1.0 / (z - others))
        denom = np.abs(1 - q + q * z * g) ** 2
        xi[k] = w[k] / denom
    if isotonic:
        xi = _isotonic_increasing(xi)
    xi *= w.sum() / xi.sum()
    C = (V * xi) @ V.T
    C = cov_to_corr(C)
    return dict(corr_clean=C, xi=xi, eigs=w)


def _isotonic_increasing(y):
    """池相邻违反者算法（PAVA），把序列投影到非降序列。"""
    y = np.asarray(y, dtype=float)
    n = len(y)
    vals = list(y); weights = [1.0] * n; blocks = [[i] for i in range(n)]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1]:
            tot = weights[i] + weights[i + 1]
            vals[i] = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / tot
            weights[i] = tot; blocks[i] += blocks[i + 1]
            del vals[i + 1]; del weights[i + 1]; del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.zeros(n)
    for v, b in zip(vals, blocks):
        out[b] = v
    return out


def ledoit_wolf_shrinkage(X) -> dict:
    """Ledoit–Wolf (2004) 线性收缩到缩放单位阵。X 形状 (T, N)。返回 ``cov``、``shrinkage``。"""
    X = as_2d(X, "X")
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / T
    mu = np.trace(S) / N
    delta2 = np.linalg.norm(S - mu * np.eye(N), "fro") ** 2
    beta2 = np.mean([np.linalg.norm(np.outer(x, x) - S, "fro") ** 2 for x in Xc]) / T
    beta2 = min(beta2, delta2)
    shrink = beta2 / delta2 if delta2 > 0 else 1.0
    cov = shrink * mu * np.eye(N) + (1 - shrink) * S
    return dict(cov=cov, shrinkage=float(shrink))


def absorption_ratio(cov, n_components: int) -> float:
    """吸收比（Kritzman 等 2011）：前 n 个特征值解释的方差占比，升高表示系统性风险集中。"""
    w = np.linalg.eigvalsh(np.asarray(cov, dtype=float))[::-1]
    return float(np.sum(w[:n_components]) / np.sum(w))


def participation_ratio(eigvec) -> float:
    """逆参与率 1/Σ v_i⁴：接近 N 表示特征向量“非局域”（市场模式），接近 1 表示集中在少数股票。"""
    v = np.asarray(eigvec, dtype=float)
    return float(1.0 / np.sum(v ** 4))


def denoise_covariance(returns, method: str = "rie") -> dict:
    """一站式：收益矩阵 (T,N) → 去噪协方差。method ∈ {'clip', 'rie', 'lw'}。"""
    X = as_2d(returns, "returns")
    T, N = X.shape
    q = N / T
    cov = np.cov(X, rowvar=False)
    std = np.sqrt(np.diag(cov))
    corr = cov_to_corr(cov)
    if method == "lw":
        out = ledoit_wolf_shrinkage(X)
        return dict(cov=out["cov"], corr=cov_to_corr(out["cov"]), info=out)
    out = clip_eigenvalues(corr, q) if method == "clip" else rie_cleaning(corr, q)
    C = out["corr_clean"]
    return dict(cov=C * np.outer(std, std), corr=C, info=out)
