"""
热传导方程与 Black–Scholes PDE 的有限差分
=========================================

学科来源
--------
Fourier (1822) 的热传导方程 u_t = D u_xx 是最早的扩散方程；Bachelier (1900) 发现期权价格满足同一方程，
Black–Scholes (1973) 的 PDE 经变量代换 x = ln S、τ = T − t 可严格化为热方程，
期权定价于是继承了整套 PDE 数值方法（Crank–Nicolson 1947 原本就是为热传导设计的）。
Fokker–Planck 方程则描述概率密度本身的扩散，是 SDE 的“欧拉视角”。

量化映射
--------
- 欧式/美式期权与奇异期权定价、希腊字母（Δ、Γ 由差分直接得到）；
- 任何“状态服从扩散、价值由终端条件倒推”的问题：最优执行的 HJB、做市的值函数（见 ``control.hjb``）；
- 概率密度的时间演化（Fokker–Planck）：收益分布如何随持有期扩散、远期命中概率（见 ``first_passage``）。

核心公式
--------
BS PDE：  V_t + ½σ²S²V_SS + rS V_S − rV = 0,  V(T,S) = payoff(S)
显式格式稳定性条件：Δt ≤ ΔS² / (σ² S_max²)；隐式与 Crank–Nicolson 无条件稳定（CN 二阶精度）。
美式期权：每步后取 V = max(V, payoff)（Brennan–Schwartz 投影，近似 LCP）。

参考文献
--------
- Black & Scholes (1973). The Pricing of Options and Corporate Liabilities. JPE.
- Crank & Nicolson (1947). A practical method for numerical evaluation of solutions of PDEs of the heat-conduction type.
- Wilmott, Howison & Dewynne (1995). The Mathematics of Financial Derivatives. CUP.
- Brennan & Schwartz (1977). The valuation of American put options. JF.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import norm


def black_scholes_price(S, K: float, r: float, sigma: float, T: float, kind: str = "call", q: float = 0.0):
    """Black–Scholes 闭式价格（含连续股息率 q）。"""
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_pde_fd(S0: float, K: float, r: float, sigma: float, T: float, kind: str = "put", american: bool = False,
              S_max: float | None = None, n_s: int = 200, n_t: int | None = None, method: str = "crank_nicolson") -> dict:
    """有限差分求解 Black–Scholes PDE（S 网格，Dirichlet 边界）。

    method ∈ {'explicit', 'implicit', 'crank_nicolson'}。返回 ``price``（S0 处插值）、``S_grid``、``V``、``delta``、``gamma``。
    """
    S_max = 4 * max(S0, K) if S_max is None else S_max
    S = np.linspace(0, S_max, n_s + 1); dS = S[1] - S[0]
    if n_t is None:
        n_t = int(np.ceil(T * sigma ** 2 * S_max ** 2 / dS ** 2)) + 1 if method == "explicit" else max(200, n_s)
    dt = T / n_t
    payoff = np.maximum(K - S, 0) if kind == "put" else np.maximum(S - K, 0)
    V = payoff.copy()
    j = np.arange(1, n_s)
    a = 0.5 * dt * (sigma ** 2 * j ** 2 - r * j)
    b = -dt * (sigma ** 2 * j ** 2 + r)
    c = 0.5 * dt * (sigma ** 2 * j ** 2 + r * j)
    theta = {"explicit": 0.0, "implicit": 1.0, "crank_nicolson": 0.5}[method]
    # (I − θL)V^{n} = (I + (1−θ)L)V^{n+1}，L 为三对角 [a, b, c]
    ab = np.zeros((3, n_s - 1))
    ab[0, 1:] = -theta * c[:-1]
    ab[1, :] = 1 - theta * b
    ab[2, :-1] = -theta * a[1:]
    for n in range(n_t):
        tau = (n + 1) * dt
        lower = K * np.exp(-r * tau) if kind == "put" else 0.0
        upper = 0.0 if kind == "put" else S_max - K * np.exp(-r * tau)
        rhs = V[j] + (1 - theta) * (a * V[j - 1] + b * V[j] + c * V[j + 1])
        rhs[0] += theta * a[0] * lower
        rhs[-1] += theta * c[-1] * upper
        V_new = np.empty_like(V)
        V_new[1:-1] = solve_banded((1, 1), ab, rhs) if theta > 0 else rhs
        V_new[0], V_new[-1] = lower, upper
        if american:
            V_new = np.maximum(V_new, payoff)
        V = V_new
    price = float(np.interp(S0, S, V))
    delta = np.gradient(V, dS); gamma = np.gradient(delta, dS)
    return dict(price=price, S_grid=S, V=V, delta=float(np.interp(S0, S, delta)), gamma=float(np.interp(S0, S, gamma)), n_t=n_t)


def fokker_planck_fd(x_grid, p0, drift, diffusion2, T: float, n_t: int = 1000) -> np.ndarray:
    """一维 Fokker–Planck 方程 p_t = −(b p)_x + ½(a p)_xx 的显式有限差分（零通量边界近似）。

    返回 (n_t+1, n_x) 的密度演化。用于看“收益分布如何随时间扩散/偏斜”。
    """
    x = np.asarray(x_grid, dtype=float); dx = x[1] - x[0]; dt = T / n_t
    b = np.asarray(drift(x), dtype=float) * np.ones_like(x); a = np.asarray(diffusion2(x), dtype=float) * np.ones_like(x)
    P = np.zeros((n_t + 1, len(x))); P[0] = p0 / np.trapezoid(p0, x)
    for n in range(n_t):
        p = P[n]
        flux_adv = b * p
        dadv = np.gradient(flux_adv, dx)
        ddiff = np.gradient(np.gradient(a * p, dx), dx)
        p_new = p + dt * (-dadv + 0.5 * ddiff)
        p_new = np.maximum(p_new, 0); p_new /= np.trapezoid(p_new, x)
        P[n + 1] = p_new
    return P
