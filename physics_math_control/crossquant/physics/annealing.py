"""
模拟退火与并行回火：统计物理的组合优化
=======================================

学科来源
--------
Kirkpatrick, Gelatt & Vecchi (1983) 把金属退火（缓慢降温使系统落入低能态）与 Metropolis 采样结合，
提出模拟退火；Swendsen & Wang (1986) 的副本交换（并行回火）让多个温度的链交换状态，克服能量壁垒。
量子退火（Kadowaki & Nishimori 1998）是其量子版本，D-Wave 等机器的组合优化由此而来。

量化映射
--------
- **基数约束组合**：从 N 只股票中选 K 只再做均值方差（NP 难）；
- **离散参数搜索**：策略参数网格大、目标函数非凸且有噪声；
- **指数复制 / 篓子构造**：选取少量成分跟踪指数（与 ``mathematics.sparse`` 的 OMP 互为补充）；
- **调度**：订单在多个场所/时段的离散分配。

参考文献
--------
- Kirkpatrick, Gelatt & Vecchi (1983). Optimization by Simulated Annealing. Science.
- Earl & Deem (2005). Parallel tempering: Theory, applications, and new perspectives. PCCP.
- Crama & Schyns (2003). Simulated annealing for complex portfolio selection problems. EJOR.
- Rosenberg et al. (2016). Solving the Optimal Trading Trajectory Problem Using a Quantum Annealer. IEEE JSTSP.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .._utils import as_rng


def simulated_annealing(energy: Callable, neighbor: Callable, x0, T0: float = 1.0, T_end: float = 1e-3,
                        n_steps: int = 5000, seed=None, schedule: str = "exp") -> dict:
    """通用模拟退火。energy(x) -> float（越小越好）；neighbor(x, rng) -> x'。

    返回 ``best``、``best_energy``、``trace``（每步当前能量）、``accept_rate``。
    """
    rng = as_rng(seed)
    x = x0; e = energy(x)
    best, be = x, e
    trace = np.zeros(n_steps); acc = 0
    for i in range(n_steps):
        frac = i / max(n_steps - 1, 1)
        T = T0 * (T_end / T0) ** frac if schedule == "exp" else T0 * (1 - frac) + T_end * frac
        xn = neighbor(x, rng); en = energy(xn)
        if en <= e or rng.random() < np.exp(-(en - e) / max(T, 1e-300)):
            x, e = xn, en; acc += 1
            if e < be:
                best, be = x, e
        trace[i] = e
    return dict(best=best, best_energy=float(be), trace=trace, accept_rate=acc / n_steps)


def parallel_tempering(energy: Callable, neighbor: Callable, x0, temps, n_sweeps: int = 2000, seed=None) -> dict:
    """并行回火（副本交换）。``temps`` 为升序温度列表，返回最低温副本的最优解与交换接受率。"""
    rng = as_rng(seed)
    temps = np.asarray(temps, dtype=float); R = len(temps)
    xs = [x0 for _ in range(R)]; es = np.array([energy(x0)] * R, dtype=float)
    best, be = x0, es[0]
    swaps = 0; swap_try = 0
    for _ in range(n_sweeps):
        for r in range(R):
            xn = neighbor(xs[r], rng); en = energy(xn)
            if en <= es[r] or rng.random() < np.exp(-(en - es[r]) / temps[r]):
                xs[r], es[r] = xn, en
                if en < be:
                    best, be = xn, en
        r = rng.integers(R - 1); swap_try += 1
        delta = (1 / temps[r] - 1 / temps[r + 1]) * (es[r] - es[r + 1])
        if np.log(rng.random()) < delta:
            xs[r], xs[r + 1] = xs[r + 1], xs[r]; es[r], es[r + 1] = es[r + 1], es[r]; swaps += 1
    return dict(best=best, best_energy=float(be), swap_rate=swaps / max(swap_try, 1))


# --------------------------------------------------------------------------- 应用：基数约束组合
def _mv_long_only(mu, Sigma, gamma):
    n = len(mu)
    if n == 1:
        return np.array([1.0]), -(mu[0] - gamma * Sigma[0, 0])
    res = minimize(lambda w: -(mu @ w - gamma * w @ Sigma @ w), np.full(n, 1 / n), bounds=[(0, 1)] * n,
                   constraints=[dict(type="eq", fun=lambda w: w.sum() - 1)], method="SLSQP")
    return res.x, res.fun


def cardinality_constrained_portfolio(mu, Sigma, k: int, gamma: float = 1.0, n_steps: int = 2000, seed=None,
                                      T0: float = None) -> dict:
    """用模拟退火选出 k 只资产并做多头均值方差（目标 max μᵀw − γ wᵀΣw）。

    返回 ``subset``（索引）、``weights``（全维权重）、``objective``、``trace``。
    """
    mu = np.asarray(mu, dtype=float); Sigma = np.asarray(Sigma, dtype=float); n = len(mu)
    rng = as_rng(seed)
    cache = {}

    def energy(subset):
        key = tuple(sorted(subset))
        if key not in cache:
            idx = np.array(key)
            _, f = _mv_long_only(mu[idx], Sigma[np.ix_(idx, idx)], gamma)
            cache[key] = f
        return cache[key]

    def neighbor(subset, rng):
        s = list(subset)
        out_i = rng.integers(k)
        candidates = [j for j in range(n) if j not in s]
        s[out_i] = candidates[rng.integers(len(candidates))]
        return tuple(sorted(s))

    x0 = tuple(sorted(rng.choice(n, k, replace=False)))
    T0 = T0 if T0 is not None else 0.1 * np.abs(mu).mean() + 1e-6
    out = simulated_annealing(energy, neighbor, x0, T0=T0, T_end=T0 * 1e-3, n_steps=n_steps, seed=rng.integers(1 << 31))
    idx = np.array(out["best"])
    w_sub, f = _mv_long_only(mu[idx], Sigma[np.ix_(idx, idx)], gamma)
    w = np.zeros(n); w[idx] = w_sub
    return dict(subset=idx, weights=w, objective=float(-f), trace=out["trace"], n_evaluated=len(cache))
