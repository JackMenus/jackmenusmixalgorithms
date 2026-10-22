"""
Copula：依赖结构与尾部相关
==========================

学科来源
--------
Sklar (1959) 定理：任何多元分布 = 边缘分布 + copula。Gaussian copula（Li 2000 把它引入 CDO 定价，
金融危机后因低估尾部相关而饱受批评）、t-copula（有对称尾部相关 λ > 0）、Archimedean 族。

量化映射
--------
- 多资产风险：分离“各自怎么动”（边缘）与“一起怎么动”（copula），尾部相关系数直接度量“同时崩”的概率；
- 情景生成：按 t-copula 采样保留极端共动；
- 配对/篓子选择：Kendall τ、尾部相关比线性相关更稳健。

核心公式
--------
t-copula 尾部相关：λ = 2 t_{ν+1}( −sqrt((ν+1)(1−ρ)/(1+ρ)) )；高斯 copula λ = 0
Kendall τ 与高斯/t copula 相关系数：ρ = sin(πτ/2)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def pseudo_observations(X) -> np.ndarray:
    """秩变换到 (0,1)：U = rank/(n+1)。"""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return (stats.rankdata(X, axis=0)) / (n + 1)


def gaussian_copula_fit(X) -> dict:
    U = pseudo_observations(X)
    Z = stats.norm.ppf(U)
    return dict(corr=np.corrcoef(Z, rowvar=False))


def gaussian_copula_sample(corr, n: int, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Z = rng.multivariate_normal(np.zeros(len(corr)), corr, size=n)
    return stats.norm.cdf(Z)


def t_copula_fit(X, nu_grid=None) -> dict:
    """t-copula 轮廓似然：对 ν 网格搜索，ρ 用 Kendall τ 反推（sin(πτ/2)）。"""
    U = pseudo_observations(X)
    d = U.shape[1]
    tau = np.array([[stats.kendalltau(U[:, i], U[:, j])[0] if i != j else 1.0 for j in range(d)] for i in range(d)])
    corr = np.sin(np.pi * tau / 2); np.fill_diagonal(corr, 1.0)
    w, V = np.linalg.eigh(corr); corr = (V * np.maximum(w, 1e-6)) @ V.T
    dd = np.sqrt(np.diag(corr)); corr = corr / np.outer(dd, dd)
    nu_grid = np.array([2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]) if nu_grid is None else np.asarray(nu_grid)
    best = None
    for nu in nu_grid:
        T_ = stats.t.ppf(U, nu)
        ll = np.sum(stats.multivariate_t.logpdf(T_, loc=np.zeros(d), shape=corr, df=nu)) - np.sum(stats.t.logpdf(T_, nu))
        if best is None or ll > best[1]:
            best = (nu, ll)
    return dict(corr=corr, nu=float(best[0]), loglik=float(best[1]))


def t_copula_sample(corr, nu: float, n: int, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corr = np.asarray(corr, dtype=float)
    Z = rng.multivariate_normal(np.zeros(len(corr)), corr, size=n)
    W = rng.chisquare(nu, size=n) / nu
    return stats.t.cdf(Z / np.sqrt(W)[:, None], nu)


def tail_dependence_t(rho: float, nu: float) -> float:
    return float(2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1))


def empirical_tail_dependence(X, q: float = 0.05, lower: bool = True) -> float:
    """两列样本的经验尾部相关 P(U₂ < q | U₁ < q)（或上尾）。"""
    U = pseudo_observations(X)
    a = U[:, 0] < q if lower else U[:, 0] > 1 - q
    b = U[:, 1] < q if lower else U[:, 1] > 1 - q
    return float(np.sum(a & b) / max(np.sum(a), 1))


def kendall_tau_to_rho(tau: float) -> float:
    return float(np.sin(np.pi * tau / 2))
