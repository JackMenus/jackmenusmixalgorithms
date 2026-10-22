"""
Hawkes 自激励点过程：从地震余震到订单流
=======================================

学科来源
--------
Hawkes (1971) 为地震学提出：每次事件都会暂时抬高后续事件的强度（余震），
Ogata (1988) 的 ETAS 模型和 thinning 模拟算法使其可计算。它是“事件会引发事件”的最简洁描述。

量化映射
--------
- **订单流建模**：市价单/撤单/限价单到达呈显著聚集（clustering），Hawkes 的分支比 n = α/β
  度量“一个订单平均引发多少个后续订单”，n → 1 时市场接近临界（Hardiman–Bercot–Bouchaud 2013）。
- **多维 Hawkes**：买方/卖方市价单互相激励 ⇒ 短期订单流不平衡的预测；跨资产事件传染。
- **做市与挂单**：成交强度 λ(t) 不是常数，Hawkes 强度可替代 Avellaneda–Stoikov 的常数 A。
- **波动率聚集 / 跳跃传染**：价格跳跃用 Hawkes 建模能复现 ACF 的慢衰减。

核心公式（指数核）
------------------
λ(t) = μ + Σ_{t_i<t} α e^{−β(t−t_i)}
对数似然：ℓ = −μT − (α/β) Σ_i (1 − e^{−β(T−t_i)}) + Σ_i log(μ + α R_i)，
R_i = e^{−β(t_i − t_{i−1})}(1 + R_{i−1})，R_1 = 0（O(N) 递推，Ozaki 1979）。
平稳条件 α/β < 1；平稳强度 μ/(1 − α/β)。
残差检验：补偿器增量 Λ(t_i) − Λ(t_{i−1}) 应为独立 Exp(1)（时间变换定理）。

参考文献
--------
- Hawkes (1971). Spectra of some self-exciting and mutually exciting point processes. Biometrika.
- Ogata (1981). On Lewis' simulation method for point processes. IEEE IT.
- Bacry, Mastromatteo & Muzy (2015). Hawkes processes in finance. Market Microstructure and Liquidity.
- Hardiman, Bercot & Bouchaud (2013). Critical reflexivity in financial markets: a Hawkes process analysis. EPJB.
- El Karmi (2025). A Deterministic LOB Simulator with Hawkes-Driven Order Flow. arXiv:2510.08085.
- Lalor & Swishchuk (2025). Event-Based LOB Simulation under a Neural Hawkes Process. arXiv:2502.17417.
- (2025). Diffusive Limit of Hawkes Driven Order Book Dynamics With Liquidity Migration. arXiv:2511.18117.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .._utils import as_1d, as_rng


# --------------------------------------------------------------------------- 一维
def simulate_hawkes(mu: float, alpha: float, beta: float, T: float, seed=None) -> np.ndarray:
    """Ogata thinning 模拟指数核 Hawkes 过程，返回事件时间数组。"""
    rng = as_rng(seed)
    times = []
    t = 0.0
    lam_excess = 0.0  # 当前 Σ α e^{−β(t−t_i)}
    while True:
        lam_bar = mu + lam_excess
        w = rng.exponential(1.0 / lam_bar)
        t_new = t + w
        if t_new > T:
            break
        lam_excess *= np.exp(-beta * w)
        if rng.random() * lam_bar <= mu + lam_excess:
            times.append(t_new)
            lam_excess += alpha
        t = t_new
    return np.array(times)


def hawkes_loglik(times, T: float, mu: float, alpha: float, beta: float) -> float:
    """指数核 Hawkes 对数似然（递推，O(N)）。"""
    t = as_1d(times, "times")
    if len(t) == 0:
        return -mu * T
    dt = np.diff(t)
    R = np.zeros(len(t))
    for i in range(1, len(t)):
        R[i] = np.exp(-beta * dt[i - 1]) * (1 + R[i - 1])
    comp = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - t)))
    return float(np.sum(np.log(mu + alpha * R)) - comp)


def fit_hawkes(times, T: float, x0=None) -> dict:
    """极大似然估计 (μ, α, β)，返回 ``mu, alpha, beta, branching_ratio, loglik, success``。"""
    t = as_1d(times, "times")
    n = len(t)
    if x0 is None:
        rate = n / T
        x0 = np.log([0.5 * rate, 0.5, 1.0])

    def nll(theta):
        mu, alpha, beta = np.exp(theta)
        return -hawkes_loglik(t, T, mu, alpha, beta)

    res = optimize.minimize(nll, np.asarray(x0, dtype=float), method="L-BFGS-B")
    mu, alpha, beta = np.exp(res.x)
    return dict(mu=float(mu), alpha=float(alpha), beta=float(beta), branching_ratio=float(alpha / beta),
                loglik=float(-res.fun), success=bool(res.success))


def hawkes_intensity(times, grid, mu: float, alpha: float, beta: float) -> np.ndarray:
    """在时间网格上计算条件强度 λ(t)（只计入 t_i < t 的事件）。"""
    t = as_1d(times, "times"); g = np.asarray(grid, dtype=float)
    lam = np.full(len(g), mu)
    for i, s in enumerate(g):
        past = t[t < s]
        lam[i] += alpha * np.sum(np.exp(-beta * (s - past)))
    return lam


def hawkes_residuals(times, mu: float, alpha: float, beta: float) -> dict:
    """时间变换残差：τ_i = Λ(t_i) − Λ(t_{i−1})，若模型正确则 iid Exp(1)。返回残差与 KS 检验结果。"""
    t = as_1d(times, "times")
    comp = mu * t + (alpha / beta) * np.array([np.sum(1 - np.exp(-beta * (ti - t[t < ti]))) for ti in t])
    tau = np.diff(np.concatenate([[0.0], comp]))
    ks = stats.kstest(tau, "expon")
    return dict(residuals=tau, ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue))


# --------------------------------------------------------------------------- 多维
def simulate_multivariate_hawkes(mu, alpha, beta, T: float, seed=None) -> tuple[np.ndarray, np.ndarray]:
    """多维指数核 Hawkes（λ_i = μ_i + Σ_j Σ_{t^j_l<t} α_ij e^{−β_ij(t−t^j_l)}），thinning 模拟。

    返回 (times, types)。
    """
    rng = as_rng(seed)
    mu = np.asarray(mu, dtype=float); alpha = np.asarray(alpha, dtype=float); beta = np.asarray(beta, dtype=float)
    d = len(mu)
    S = np.zeros((d, d))  # S_ij = Σ_l e^{−β_ij (t − t^j_l)}
    t = 0.0
    times, types = [], []
    while True:
        lam = mu + np.sum(alpha * S, axis=1)
        lam_bar = lam.sum()
        w = rng.exponential(1.0 / lam_bar)
        t_new = t + w
        if t_new > T:
            break
        S = S * np.exp(-beta * w)
        lam = mu + np.sum(alpha * S, axis=1)
        if rng.random() * lam_bar <= lam.sum():
            m = rng.choice(d, p=lam / lam.sum())
            times.append(t_new); types.append(m)
            S[:, m] += 1.0
        t = t_new
    return np.array(times), np.array(types, dtype=int)


def multivariate_hawkes_loglik(times, types, T: float, mu, alpha, beta) -> float:
    """多维指数核 Hawkes 对数似然（按时间顺序递推，O(N d²)）。"""
    t = as_1d(times, "times"); k = np.asarray(types, dtype=int)
    mu = np.asarray(mu, dtype=float); alpha = np.asarray(alpha, dtype=float); beta = np.asarray(beta, dtype=float)
    d = len(mu)
    S = np.zeros((d, d))
    ll = 0.0
    prev = 0.0
    for ti, m in zip(t, k):
        S = S * np.exp(-beta * (ti - prev))
        lam_m = mu[m] + np.sum(alpha[m] * S[m])
        ll += np.log(max(lam_m, 1e-300))
        S[:, m] += 1.0
        prev = ti
    comp = np.sum(mu) * T
    for j in range(d):
        tj = t[k == j]
        if len(tj):
            comp += np.sum((alpha[:, j] / beta[:, j])[:, None] * (1 - np.exp(-beta[:, j][:, None] * (T - tj)[None, :])))
    return float(ll - comp)


def fit_multivariate_hawkes(times, types, T: float, d: int | None = None, shared_beta: bool = True, x0=None) -> dict:
    """多维 Hawkes 极大似然（指数核）。``shared_beta=True`` 时所有核共用一个 β（参数更少、更稳）。"""
    t = as_1d(times, "times"); k = np.asarray(types, dtype=int)
    d = int(k.max()) + 1 if d is None else d
    n_beta = 1 if shared_beta else d * d

    def unpack(theta):
        mu = np.exp(theta[:d])
        alpha = np.exp(theta[d:d + d * d]).reshape(d, d)
        beta = np.exp(theta[d + d * d:])
        beta = np.full((d, d), beta[0]) if shared_beta else beta.reshape(d, d)
        return mu, alpha, beta

    if x0 is None:
        rate = np.array([np.sum(k == j) / T for j in range(d)])
        x0 = np.concatenate([np.log(0.5 * rate + 1e-6), np.log(np.full(d * d, 0.2 / d)), np.zeros(n_beta)])

    def nll(theta):
        mu, alpha, beta = unpack(theta)
        return -multivariate_hawkes_loglik(t, k, T, mu, alpha, beta)

    res = optimize.minimize(nll, np.asarray(x0, dtype=float), method="L-BFGS-B")
    mu, alpha, beta = unpack(res.x)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(alpha / beta))))
    return dict(mu=mu, alpha=alpha, beta=beta, branching_matrix=alpha / beta, spectral_radius=spectral_radius,
                loglik=float(-res.fun), success=bool(res.success))
