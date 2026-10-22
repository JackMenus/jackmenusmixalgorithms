"""最优执行轨迹（母单如何随时间切分）。

- almgren_chriss：Almgren & Chriss (2000) 线性冲击 + 均值方差，离散二次规划求解，
  支持 alpha 漂移（因子信号）；与 sinh 闭式解在无漂移时一致。
- obizhaeva_wang：Obizhaeva & Wang (2013) 指数恢复的瞬时冲击模型闭式解。
- propagator_optimal：任意衰减核（Bouchaud 2004 / Gatheral 2010 的幂律核等）下的最优切分。
- twap / vwap / pov：基准算法。

约定：所有函数返回长度 N 的数组 ``n``，n[j] 为第 j 个时间区间的成交量（总和 = X）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ACParams:
    sigma: float      # 价格波动率（价格单位/√秒）
    eta: float        # 临时冲击系数 η（价格单位·秒/股）
    gamma: float = 0.0  # 永久冲击系数 γ（价格单位/股）
    epsilon: float = 0.0  # 固定成本（半价差 + 费用）
    risk_aversion: float = 1e-6  # λ
    drift: float = 0.0  # α：价格漂移（价格单位/秒），来自 alpha 信号


def almgren_chriss(X: float, T: float, N: int, p: ACParams, is_sell: bool = True) -> np.ndarray:
    """离散 Almgren-Chriss（含漂移）。

    最小化 U = (η̃/τ)Σ n_j² + λσ²τ Σ x_j² − α_eff·τ Σ x_j，
    其中 η̃ = η − γτ/2，x_j 为剩余持仓，α_eff = α（卖出）或 −α（买入）。
    一阶条件为三对角线性方程组。
    """
    if N < 1:
        raise ValueError("N 必须 ≥ 1")
    tau = T / N
    eta_t = p.eta - 0.5 * p.gamma * tau
    if eta_t <= 0:
        raise ValueError("η̃ = η − γτ/2 必须为正，请增大 N 或检查参数")
    a_eff = p.drift if is_sell else -p.drift
    m = N - 1
    if m == 0:
        return np.array([X], float)
    diag = 2 * eta_t / tau * 2 + 2 * p.risk_aversion * p.sigma**2 * tau
    off = -2 * eta_t / tau
    M = np.diag(np.full(m, diag)) + np.diag(np.full(m - 1, off), 1) + np.diag(np.full(m - 1, off), -1)
    b = np.full(m, a_eff * tau)
    b[0] -= off * X  # 边界 x_0 = X
    x_inner = np.linalg.solve(M, b)
    x = np.concatenate([[X], x_inner, [0.0]])
    return -np.diff(x)


def almgren_chriss_closed_form(X: float, T: float, N: int, p: ACParams) -> np.ndarray:
    """无漂移情形的 sinh 闭式解，用于验证。"""
    tau = T / N
    eta_t = p.eta - 0.5 * p.gamma * tau
    kt2 = p.risk_aversion * p.sigma**2 / eta_t
    if kt2 == 0:
        return np.full(N, X / N)
    kappa = math.acosh(0.5 * kt2 * tau**2 + 1) / tau
    t = np.arange(N + 1) * tau
    x = X * np.sinh(kappa * (T - t)) / math.sinh(kappa * T)
    return -np.diff(x)


def ac_cost_moments(n: np.ndarray, X: float, T: float, p: ACParams, is_sell: bool = True) -> Tuple[float, float]:
    """给定切分 n，返回 (期望冲击成本, 成本方差)。"""
    N = len(n)
    tau = T / N
    eta_t = p.eta - 0.5 * p.gamma * tau
    x = X - np.concatenate([[0.0], np.cumsum(n)])
    a_eff = p.drift if is_sell else -p.drift
    e = 0.5 * p.gamma * X**2 + p.epsilon * np.abs(n).sum() + eta_t / tau * (n**2).sum() - a_eff * tau * x[1:].sum()
    v = p.sigma**2 * tau * (x[1:] ** 2).sum()
    return float(e), float(v)


def efficient_frontier(X: float, T: float, N: int, p: ACParams, lambdas: Sequence[float]) -> np.ndarray:
    """返回每个 λ 对应的 (λ, E, V)。"""
    rows = []
    for lam in lambdas:
        q = ACParams(p.sigma, p.eta, p.gamma, p.epsilon, lam, p.drift)
        n = almgren_chriss(X, T, N, q)
        rows.append((lam, *ac_cost_moments(n, X, T, q)))
    return np.array(rows)


def obizhaeva_wang(X: float, T: float, N: int, rho: float) -> np.ndarray:
    """OW (2013) 最优解：首尾各一笔大单 X/(ρT+2)，中间匀速 ρX/(ρT+2)。

    返回 N 个区间的分配（首笔并入第 0 区间，尾笔并入最后区间）。
    """
    block = X / (rho * T + 2.0)
    cont = X - 2 * block
    n = np.full(N, cont / N)
    n[0] += block
    n[-1] += block
    return n


def exp_kernel(rho: float, depth: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    return lambda t: np.exp(-rho * np.asarray(t)) / depth


def power_kernel(beta: float = 0.4, t0: float = 1.0, scale: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """幂律衰减核 G(t) = scale·(1 + t/t0)^(−β)（Bouchaud 等 2004；Gatheral 2010 无套利要求 β∈(0,1)）。"""
    return lambda t: scale * (1.0 + np.asarray(t) / t0) ** (-beta)


def propagator_cost(n: np.ndarray, times: np.ndarray, kernel: Callable) -> float:
    """瞬时冲击成本 ½ nᵀ Γ n，Γ_ij = G(|t_i − t_j|)。"""
    G = kernel(np.abs(times[:, None] - times[None, :]))
    return float(0.5 * n @ G @ n)


def propagator_optimal(X: float, times: Sequence[float], kernel: Callable, nonneg: bool = True) -> np.ndarray:
    """min ½nᵀΓn s.t. Σn = X。若 nonneg=True，用主动集法迭代剔除负值（禁止反向交易）。"""
    t = np.asarray(times, float)
    G = kernel(np.abs(t[:, None] - t[None, :]))
    active = np.ones(len(t), bool)
    for _ in range(len(t)):
        Ga = G[np.ix_(active, active)]
        w = np.linalg.solve(Ga, np.ones(active.sum()))
        na = X * w / w.sum()
        if not nonneg or (na >= -1e-12).all():
            break
        idx = np.where(active)[0]
        active[idx[na < 0]] = False
    n = np.zeros(len(t))
    n[active] = na
    return np.clip(n, 0, None) if nonneg else n


def twap(X: float, N: int) -> np.ndarray:
    return np.full(N, X / N)


def vwap(X: float, volume_profile: Sequence[float]) -> np.ndarray:
    v = np.asarray(volume_profile, float)
    return X * v / v.sum()


def pov(X: float, expected_volume: Sequence[float], rate: float) -> np.ndarray:
    """按市场成交量比例参与；若到期仍未完成，剩余量放在最后一个区间。"""
    v = np.asarray(expected_volume, float) * rate
    cum = np.minimum(np.cumsum(v), X)
    n = np.diff(np.concatenate([[0.0], cum]))
    n[-1] += X - n.sum()
    return n
