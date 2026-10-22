"""
HJB 方程与随机最优控制：Merton 问题与通用一维有限差分求解器
==========================================================

学科来源
--------
Hamilton–Jacobi 方程来自分析力学（Hamilton 1834），Bellman (1957) 的动态规划把它推广到控制问题，
得到 Hamilton–Jacobi–Bellman (HJB) 偏微分方程：
    V_t + sup_u { b(x,u) V_x + ½ a(x,u) V_xx + f(x,u) } = 0,   V(T,x) = g(x)
“值函数在时间上倒着传播，每一步在所有控制里挑最好的”。

量化映射
--------
- **Merton (1969) 组合问题**：财富 W 在风险资产上的比例 π 为控制，CRRA 效用 ⇒ π* = (μ−r)/(γσ²)，
  与时间、财富无关（“Merton 比例”），它是 Kelly 分数的风险厌恶推广。
- 最优执行（Almgren–Chriss/Cartea–Jaimungal）、做市（AS/GLFT）、最优停止（美式期权、配对交易出场）
  都是 HJB 的特例；通用求解器可以直接求数值解来验证闭式解或处理无闭式解的变体。

数值方法
--------
显式后向 Euler + 漂移项迎风（upwind）差分 + 扩散项中心差分，控制在网格上穷举取 max。
稳定性要求 dt ≲ dx² / max a(x,u)；边界用一阶外推（V_xx = 0）。

参考文献
--------
- Merton (1969). Lifetime Portfolio Selection under Uncertainty: The Continuous-Time Case. REStat.
- Fleming & Soner (2006). Controlled Markov Processes and Viscosity Solutions.
- Pham (2009). Continuous-time Stochastic Control and Optimization with Financial Applications.
- Forsyth & Labahn (2007). Numerical methods for controlled HJB PDEs in finance. J. Comp. Finance.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# --------------------------------------------------------------------------- Merton 闭式解
def merton_optimal_fraction(mu: float, r: float, sigma: float, gamma: float) -> float:
    """Merton 比例 π* = (μ − r)/(γσ²)。γ=1 即 Kelly（对数效用）。"""
    return (mu - r) / (gamma * sigma ** 2)


def merton_value(W, t: float, T: float, mu: float, r: float, sigma: float, gamma: float):
    """Merton 问题的值函数（CRRA，γ≠1；γ=1 用对数效用）。"""
    W = np.asarray(W, dtype=float)
    tau = T - t
    sharpe2 = (mu - r) ** 2 / sigma ** 2
    if abs(gamma - 1.0) < 1e-12:
        return np.log(W) + (r + 0.5 * sharpe2) * tau
    kappa = (1 - gamma) * (r + sharpe2 / (2 * gamma))
    return W ** (1 - gamma) / (1 - gamma) * np.exp(kappa * tau)


# --------------------------------------------------------------------------- 通用求解器
def solve_hjb_1d(x_grid, u_grid, drift: Callable, diffusion2: Callable, running: Callable,
                 terminal: Callable, T: float, n_t: int | None = None, cfl: float = 0.4) -> dict:
    """显式有限差分求解一维 HJB：V_t + max_u{ b V_x + ½ a V_xx + f } = 0。

    参数
    ----
    x_grid : (n_x,) 状态网格（等距）
    u_grid : (n_u,) 控制网格
    drift(x, u) -> b；diffusion2(x, u) -> a = σ²；running(x, u) -> f；均为向量化函数，
        输入形状 (n_x, 1) 与 (1, n_u)，输出 (n_x, n_u)
    terminal(x) -> g(x)
    T : 终端时间；n_t : 时间步数（缺省按 CFL 条件自动选择）

    返回 dict：``V (n_t+1, n_x)``（V[0] 为 t=0）、``policy (n_t, n_x)``、``t_grid``。
    """
    x = np.asarray(x_grid, dtype=float); u = np.asarray(u_grid, dtype=float)
    dx = x[1] - x[0]
    X, U = x[:, None], u[None, :]
    b = np.broadcast_to(drift(X, U), (len(x), len(u))).astype(float)
    a = np.broadcast_to(diffusion2(X, U), (len(x), len(u))).astype(float)
    f = np.broadcast_to(running(X, U), (len(x), len(u))).astype(float)
    if n_t is None:
        denom = np.max(a) / dx ** 2 + np.max(np.abs(b)) / dx + 1e-12
        n_t = int(np.ceil(T * denom / cfl)) + 1
    dt = T / n_t
    V = np.zeros((n_t + 1, len(x)))
    policy = np.zeros((n_t, len(x)))
    V[-1] = terminal(x)
    for n in range(n_t - 1, -1, -1):
        v = V[n + 1]
        # 一阶导：迎风；二阶导：中心（边界处 V_xx = 0，一阶导单侧）
        fwd = np.empty_like(v); bwd = np.empty_like(v); vxx = np.zeros_like(v)
        fwd[:-1] = (v[1:] - v[:-1]) / dx; fwd[-1] = fwd[-2]
        bwd[1:] = (v[1:] - v[:-1]) / dx; bwd[0] = bwd[1]
        vxx[1:-1] = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx ** 2
        vx = np.where(b >= 0, fwd[:, None], bwd[:, None])
        H = b * vx + 0.5 * a * vxx[:, None] + f
        best = np.argmax(H, axis=1)
        V[n] = v + dt * H[np.arange(len(x)), best]
        policy[n] = u[best]
    return dict(V=V, policy=policy, t_grid=np.linspace(0, T, n_t + 1), dt=dt)


def merton_hjb_numeric(mu: float, r: float, sigma: float, gamma: float, T: float = 1.0,
                       x_range=(-1.5, 1.5), n_x: int = 121, pi_max: float = 2.0, n_u: int = 81) -> dict:
    """用通用求解器数值求解 Merton 问题（对数财富 x = ln W 作为状态，系数为常数）。

    dx = [r + π(μ−r) − ½π²σ²] dt + πσ dB,  终端效用 U = e^{(1−γ)x}/(1−γ)（γ=1 时 U = x）。
    返回 ``policy_t0``（t=0 的最优 π(x)）、``pi_closed``、``V``、``x_grid``。
    """
    x = np.linspace(*x_range, n_x); u = np.linspace(0.0, pi_max, n_u)
    drift = lambda X, U: r + U * (mu - r) - 0.5 * U ** 2 * sigma ** 2 + 0 * X
    diff2 = lambda X, U: (U * sigma) ** 2 + 0 * X
    run = lambda X, U: 0.0 * X * U
    if abs(gamma - 1.0) < 1e-12:
        term = lambda X: X.copy()
    else:
        term = lambda X: np.exp((1 - gamma) * X) / (1 - gamma)
    sol = solve_hjb_1d(x, u, drift, diff2, run, term, T)
    return dict(policy_t0=sol["policy"][0], pi_closed=merton_optimal_fraction(mu, r, sigma, gamma),
                V=sol["V"], x_grid=x, u_grid=u, n_t=len(sol["t_grid"]) - 1)
