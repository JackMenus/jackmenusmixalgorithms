"""
最优停止：Longstaff–Schwartz 最小二乘蒙特卡洛 与 均值回复价差的最优进出场
==========================================================================

学科来源
--------
最优停止是随机控制的“只有一次决策”特例（Wald 1947 序贯分析；Snell 包络；自由边界问题）。
美式期权定价是其标志性应用；Longstaff–Schwartz (2001) 用回归近似继续价值，把高维问题变成可算。

量化映射
--------
1. **美式/百慕大期权**、可赎回债、提前终止的结构化产品定价与对冲。
2. **配对交易的最优进出场**（Leung & Li 2015）：价差服从 OU 过程，持有成本/交易成本给定，
   求“在什么水平平仓最优”的阈值 b*，以及“在什么水平开仓最优”的阈值 d*（双重停止）。
   相比固定 z-score 阈值（±2σ），最优阈值随均值回复速度、波动率、折现率与成本变化。
3. **止损/止盈规则**作为停止时间的最优化（见 ``physics.first_passage`` 的命中概率）。

核心公式
--------
Snell 包络： V_t = max( h(X_t), E[e^{−rΔ} V_{t+Δ} | X_t] )
LSM：  在每个时刻把 e^{−rΔ}V_{t+Δ} 对 X_t 的基函数回归，得到继续价值 C(X_t)，若 h(X_t) > C 则执行。
OU 最优出场（值迭代）：V(x) = max( x − c, e^{−rΔ} E[V(X_{t+Δ}) | X_t = x] )，
    X_{t+Δ} | X_t ~ N( θ + (x−θ)e^{−κΔ}, σ²(1−e^{−2κΔ})/(2κ) )

参考文献
--------
- Longstaff & Schwartz (2001). Valuing American Options by Simulation: A Simple Least-Squares Approach. RFS.
- Leung & Li (2015). Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit. IJTAF.
- Peskir & Shiryaev (2006). Optimal Stopping and Free-Boundary Problems.
- Cox, Ross & Rubinstein (1979). Option pricing: A simplified approach（二叉树，用作验证）。
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import laguerre

from .._utils import as_rng


# --------------------------------------------------------------------------- LSM
def _basis(x, degree, kind):
    x = np.asarray(x, dtype=float)
    if kind == "laguerre":
        return laguerre.lagvander(x, degree)
    cols = [x ** d for d in range(degree + 1)]
    return np.stack(cols, axis=1)


def american_option_lsm(S0: float, K: float, r: float, sigma: float, T: float, n_steps: int = 50,
                        n_paths: int = 20000, option: str = "put", degree: int = 3, basis: str = "poly",
                        antithetic: bool = True, seed=None) -> dict:
    """Longstaff–Schwartz 最小二乘蒙特卡洛定价美式期权（GBM 标的）。

    返回 ``price``、``std_err``、``exercise_boundary``（各时刻近似的执行边界，无样本处为 nan）。
    """
    rng = as_rng(seed)
    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((half, n_steps))
    if antithetic:
        Z = np.vstack([Z, -Z])
    n_paths = Z.shape[0]
    logS = np.log(S0) + np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), np.exp(logS)])
    payoff = (lambda s: np.maximum(K - s, 0.0)) if option == "put" else (lambda s: np.maximum(s - K, 0.0))
    cash = payoff(S[:, -1])
    disc = np.exp(-r * dt)
    boundary = np.full(n_steps + 1, np.nan)
    for t in range(n_steps - 1, 0, -1):
        cash *= disc
        h = payoff(S[:, t])
        itm = h > 0
        if itm.sum() < degree + 2:
            continue
        x = S[itm, t] / K
        X = _basis(x, degree, basis)
        coef, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        cont = X @ coef
        exercise = h[itm] > cont
        idx = np.where(itm)[0][exercise]
        cash[idx] = h[itm][exercise]
        if exercise.any():
            boundary[t] = (S[itm, t][exercise]).max() if option == "put" else (S[itm, t][exercise]).min()
    cash *= disc
    price = max(float(np.mean(cash)), float(payoff(np.array([S0]))[0]))
    return dict(price=price, std_err=float(np.std(cash) / np.sqrt(n_paths)), exercise_boundary=boundary)


def crr_binomial_american(S0: float, K: float, r: float, sigma: float, T: float, n_steps: int = 500,
                          option: str = "put") -> float:
    """Cox–Ross–Rubinstein 二叉树美式期权（用于验证 LSM）。"""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt)); d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n_steps + 1)
    S = S0 * u ** (n_steps - j) * d ** j
    V = np.maximum(K - S, 0) if option == "put" else np.maximum(S - K, 0)
    for n in range(n_steps - 1, -1, -1):
        j = np.arange(n + 1)
        S = S0 * u ** (n - j) * d ** j
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        ex = np.maximum(K - S, 0) if option == "put" else np.maximum(S - K, 0)
        V = np.maximum(V, ex)
    return float(V[0])


# --------------------------------------------------------------------------- OU 最优进出场
def _ou_transition_matrix(grid, theta, kappa, sigma, dt):
    m = theta + (grid - theta) * np.exp(-kappa * dt)
    v = sigma ** 2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)
    dx = grid[1] - grid[0]
    P = np.exp(-0.5 * (grid[None, :] - m[:, None]) ** 2 / v) * dx / np.sqrt(2 * np.pi * v)
    return P / P.sum(axis=1, keepdims=True)


def ou_optimal_exit(theta: float, kappa: float, sigma: float, r: float, cost: float = 0.0, dt: float = 1.0,
                    grid=None, n_grid: int = 401, tol: float = 1e-10, max_iter: int = 100000) -> dict:
    """OU 价差多头的最优平仓阈值（Leung–Li 的离散时间值迭代版本）。

    持有价差 X_t（已按 1 单位头寸计），平仓收到 X − cost；折现率 r；
    V(x) = max( x − cost, e^{−r dt} E[V(X')|x] )。返回 ``threshold``（最小的执行水平 b*）、``V``、``grid``。
    """
    sd = sigma / np.sqrt(2 * kappa)
    grid = np.linspace(theta - 6 * sd, theta + 6 * sd, n_grid) if grid is None else np.asarray(grid, dtype=float)
    P = _ou_transition_matrix(grid, theta, kappa, sigma, dt)
    disc = np.exp(-r * dt)
    stop = grid - cost
    V = stop.copy()
    for _ in range(max_iter):
        V_new = np.maximum(stop, disc * P @ V)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    exercise = stop >= disc * P @ V - 1e-12
    idx = np.where(exercise)[0]
    threshold = float(grid[idx[0]]) if len(idx) else np.nan
    return dict(threshold=threshold, V=V, grid=grid, exercise_region=exercise)


def ou_optimal_entry_exit(theta: float, kappa: float, sigma: float, r: float, cost: float = 0.0,
                          dt: float = 1.0, n_grid: int = 401) -> dict:
    """双重停止：先解出场值函数 V，再解进场 J(x) = max( V(x) − x − cost, e^{−r dt} E[J(X')|x] )。

    返回 ``exit_threshold`` b*（高于它则平仓）、``entry_threshold`` d*（低于它则开多仓）。
    """
    ex = ou_optimal_exit(theta, kappa, sigma, r, cost, dt, n_grid=n_grid)
    grid, V = ex["grid"], ex["V"]
    P = _ou_transition_matrix(grid, theta, kappa, sigma, dt)
    disc = np.exp(-r * dt)
    enter = V - grid - cost
    J = np.maximum(enter, 0.0)
    for _ in range(100000):
        J_new = np.maximum(enter, disc * P @ J)
        if np.max(np.abs(J_new - J)) < 1e-10:
            J = J_new
            break
        J = J_new
    region = enter >= disc * P @ J - 1e-12
    idx = np.where(region)[0]
    entry = float(grid[idx[-1]]) if len(idx) else np.nan
    return dict(exit_threshold=ex["threshold"], entry_threshold=entry, V=V, J=J, grid=grid)
