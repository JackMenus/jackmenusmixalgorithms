"""
最优传输（Optimal Transport）与分布距离
=======================================

学科来源
--------
Monge (1781) 的“搬土问题”，Kantorovich (1942) 的线性规划松弛，Brenier (1991) 的几何刻画，
Villani、Otto 的 Wasserstein 几何；Cuturi (2013) 的熵正则化 Sinkhorn 算法使其可大规模计算。
Wasserstein 距离把“把一个分布搬成另一个分布的最小代价”变成分布之间的真正度量。

量化映射
--------
- **分布层面的 regime 检测**：滚动窗口收益分布与基准分布的 W₁ 距离，比均值/方差变化更灵敏；
- **分布鲁棒组合（Wasserstein DRO）**：在样本分布的 Wasserstein 球内取最坏情形 ⇒ 等价于对权重的范数正则化
  （Blanchet, Chen & Zhou 2022），给出“多少正则化”对应“多大的分布不确定性”；
- **情景生成 / 压力测试**：Wasserstein 重心把多个历史情景“平均”成合成情景；
- **因子相似度 / 策略相似度**：用两组收益分布的 W 距离或 MMD 做去重；
- **订单流匹配**（OT 把供需曲线匹配成本最小化）与 **生成模型评价**（市场模拟器输出 vs 真实数据的 W 距离）。

核心公式
--------
W_p(μ,ν) = ( inf_{π∈Π(μ,ν)} ∫ |x−y|^p dπ )^{1/p}；一维：W_p^p = ∫₀¹ |F_μ^{-1}(u) − F_ν^{-1}(u)|^p du
Sinkhorn：min ⟨P, C⟩ − ε H(P)  ⇒  P = diag(u) K diag(v)，K = e^{−C/ε}，交替缩放 u = a/(Kv)，v = b/(Kᵀu)
高斯 W₂²：‖m₁−m₂‖² + tr(C₁ + C₂ − 2 (C₁^{1/2} C₂ C₁^{1/2})^{1/2})（Bures 度量）
Wasserstein DRO 均值方差（Blanchet–Chen–Zhou）：
    min_w ( sqrt(wᵀΣ̂w) + √δ ‖w‖₂ )²   s.t.  μ̂ᵀw − √δ‖w‖₂ ≥ ρ,  1ᵀw = 1
MMD²(P,Q) = E k(x,x') + E k(y,y') − 2E k(x,y)（核均值嵌入距离，Gretton 等 2012）

参考文献
--------
- Villani (2009). Optimal Transport: Old and New. Springer.
- Peyré & Cuturi (2019). Computational Optimal Transport. FnT ML.
- Cuturi (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. NeurIPS.
- Blanchet, Chen & Zhou (2022). Distributionally Robust Mean-Variance Portfolio Selection with Wasserstein Distances. Mgmt Sci.
- Hsieh & Gan (2026). Certified High-Dimensional Wasserstein Robust Portfolio Optimization. arXiv:2608.07032.
- (2025). Loss-aware distributionally robust optimization via trainable optimal transport ambiguity sets. arXiv:2509.12689.
- Gretton et al. (2012). A Kernel Two-Sample Test. JMLR.
"""
from __future__ import annotations

import numpy as np
from scipy import linalg, optimize

from .._utils import as_1d, as_rng


# --------------------------------------------------------------------------- 一维 Wasserstein
def wasserstein_1d(x, y, p: float = 1.0, wx=None, wy=None, n_quantiles: int = 2000) -> float:
    """一维样本间的 W_p 距离（分位数函数积分）。等权且样本量相同时精确；否则在 n_quantiles 网格上积分。"""
    x = np.sort(as_1d(x, "x")); y = np.sort(as_1d(y, "y"))
    if wx is None and wy is None and len(x) == len(y):
        return float(np.mean(np.abs(x - y) ** p) ** (1 / p))
    u = (np.arange(n_quantiles) + 0.5) / n_quantiles
    qx = _weighted_quantile(x, u, wx); qy = _weighted_quantile(y, u, wy)
    return float(np.mean(np.abs(qx - qy) ** p) ** (1 / p))


def _weighted_quantile(sorted_x, u, w=None):
    w = np.ones(len(sorted_x)) if w is None else np.asarray(w, dtype=float)
    w = w / w.sum()
    cdf = np.cumsum(w)
    idx = np.searchsorted(cdf, u, side="left")
    return sorted_x[np.clip(idx, 0, len(sorted_x) - 1)]


def wasserstein_barycenter_1d(samples: list, weights=None, n_quantiles: int = 1000) -> np.ndarray:
    """一维 W₂ 重心 = 分位数函数的加权平均，返回重心分布的分位数样本。"""
    weights = np.full(len(samples), 1 / len(samples)) if weights is None else np.asarray(weights, dtype=float)
    u = (np.arange(n_quantiles) + 0.5) / n_quantiles
    Q = np.stack([np.quantile(np.asarray(s, dtype=float), u) for s in samples])
    return weights @ Q


def rolling_wasserstein(returns, window: int = 60, reference=None, p: float = 1.0, step: int = 1) -> dict:
    """滚动窗口分布与基准分布（缺省为全样本）之间的 W_p 距离，作为 regime 变化指标。"""
    r = as_1d(returns, "returns")
    ref = r if reference is None else as_1d(reference, "reference")
    idx, d = [], []
    for end in range(window, len(r) + 1, step):
        idx.append(end - 1); d.append(wasserstein_1d(r[end - window:end], ref, p))
    return dict(index=np.array(idx), distance=np.array(d))


# --------------------------------------------------------------------------- Sinkhorn
def sinkhorn(a, b, C, eps: float = 0.05, n_iter: int = 2000, tol: float = 1e-9) -> dict:
    """对数域 Sinkhorn（数值稳定）。a, b 为边缘分布，C 为成本矩阵。返回 ``plan``、``cost``、``f``、``g``（对偶势）。"""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float); C = np.asarray(C, dtype=float)
    a = a / a.sum(); b = b / b.sum()
    log_a, log_b = np.log(a + 1e-300), np.log(b + 1e-300)
    f = np.zeros(len(a)); g = np.zeros(len(b))
    for it in range(n_iter):
        f_new = eps * (log_a - _logsumexp((g[None, :] - C) / eps, axis=1))
        g_new = eps * (log_b - _logsumexp((f_new[:, None] - C) / eps, axis=0))
        err = np.max(np.abs(f_new - f)) + np.max(np.abs(g_new - g))
        f, g = f_new, g_new
        if err < tol:
            break
    P = np.exp((f[:, None] + g[None, :] - C) / eps)
    return dict(plan=P, cost=float(np.sum(P * C)), f=f, g=g, n_iter=it + 1)


def _logsumexp(M, axis):
    m = np.max(M, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(M - m), axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn_divergence(x, y, eps: float = 0.05, p: float = 2.0) -> float:
    """Sinkhorn 散度 S = OT_ε(x,y) − ½OT_ε(x,x) − ½OT_ε(y,y)（去偏，Genevay 等 2018）。"""
    X = np.atleast_2d(np.asarray(x, dtype=float)).reshape(len(x), -1); Y = np.atleast_2d(np.asarray(y, dtype=float)).reshape(len(y), -1)

    def ot(A, B):
        C = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2) ** (p / 2)
        return sinkhorn(np.ones(len(A)), np.ones(len(B)), C, eps)["cost"]
    return float(ot(X, Y) - 0.5 * ot(X, X) - 0.5 * ot(Y, Y))


# --------------------------------------------------------------------------- 高斯 W2
def gaussian_w2(m1, C1, m2, C2) -> float:
    """两个高斯分布的 W₂ 距离（Bures–Wasserstein）。"""
    m1, m2 = np.atleast_1d(m1), np.atleast_1d(m2)
    C1, C2 = np.atleast_2d(C1), np.atleast_2d(C2)
    s1 = linalg.sqrtm(C1).real
    cross = linalg.sqrtm(s1 @ C2 @ s1).real
    d2 = np.sum((m1 - m2) ** 2) + np.trace(C1 + C2 - 2 * cross)
    return float(np.sqrt(max(d2, 0.0)))


# --------------------------------------------------------------------------- Wasserstein DRO
def wasserstein_dro_mean_variance(mu, Sigma, delta: float, target_return: float | None = None,
                                  long_only: bool = False) -> dict:
    """Blanchet–Chen–Zhou 型 Wasserstein 分布鲁棒均值方差组合。

    min_w ( sqrt(wᵀΣw) + √δ‖w‖₂ )²  s.t. 1ᵀw = 1，[μᵀw − √δ‖w‖₂ ≥ ρ]，[w ≥ 0]。
    δ = 0 时退化为最小方差（或给定目标收益的均值方差）。返回 ``weights``、``robust_std``、``success``。
    """
    mu = np.asarray(mu, dtype=float); Sigma = np.asarray(Sigma, dtype=float); n = len(mu)
    sd = np.sqrt(delta)

    def obj(w):
        return (np.sqrt(w @ Sigma @ w + 1e-16) + sd * np.linalg.norm(w)) ** 2
    cons = [dict(type="eq", fun=lambda w: w.sum() - 1)]
    if target_return is not None:
        cons.append(dict(type="ineq", fun=lambda w: mu @ w - sd * np.linalg.norm(w) - target_return))
    bounds = [(0, 1)] * n if long_only else None
    res = optimize.minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=bounds, constraints=cons, options=dict(maxiter=500, ftol=1e-12))
    return dict(weights=res.x, robust_std=float(np.sqrt(res.fun)), success=bool(res.success))


# --------------------------------------------------------------------------- MMD / 能量距离
def mmd(x, y, bandwidth: float | None = None) -> float:
    """无偏 MMD² 估计（高斯核；带宽缺省为合并样本的中位数距离）。"""
    X = np.asarray(x, dtype=float).reshape(len(x), -1); Y = np.asarray(y, dtype=float).reshape(len(y), -1)
    Z = np.vstack([X, Y])
    D2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
    if bandwidth is None:
        bandwidth = np.sqrt(np.median(D2[D2 > 0]) / 2)
    K = np.exp(-D2 / (2 * bandwidth ** 2))
    n, m = len(X), len(Y)
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    term_x = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    term_y = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(term_x + term_y - 2 * Kxy.mean())


def mmd_permutation_test(x, y, n_perm: int = 200, seed=None, bandwidth=None) -> dict:
    """MMD 两样本置换检验，返回 ``mmd``、``p_value``。"""
    rng = as_rng(seed)
    X = np.asarray(x, dtype=float).reshape(len(x), -1); Y = np.asarray(y, dtype=float).reshape(len(y), -1)
    stat = mmd(X, Y, bandwidth)
    Z = np.vstack([X, Y]); n = len(X)
    null = np.array([mmd(*(lambda z: (z[:n], z[n:]))(Z[rng.permutation(len(Z))]), bandwidth) for _ in range(n_perm)])
    return dict(mmd=stat, p_value=float(np.mean(null >= stat)))


def energy_distance(x, y) -> float:
    """能量距离 2E|X−Y| − E|X−X'| − E|Y−Y'|（Székely）。"""
    X = np.asarray(x, dtype=float).reshape(len(x), -1); Y = np.asarray(y, dtype=float).reshape(len(y), -1)
    d = lambda A, B: np.mean(np.sqrt(np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)))
    return float(2 * d(X, Y) - d(X, X) - d(Y, Y))
