"""
随机微分方程模拟与（拟）蒙特卡洛
=================================

学科来源
--------
Itô (1944) 的随机积分与 SDE；数值格式 Euler–Maruyama（强阶 0.5）、Milstein（强阶 1.0）
（Kloeden & Platen 1992）。拟蒙特卡洛来自数论（Halton 1960、Sobol 1967 的低差异序列），
布朗桥构造（Caflisch, Morokoff & Owen 1997）把方差集中到前几维以发挥 QMC 优势。

量化映射
--------
- 定价/风险的情景生成：GBM、OU、CIR（利率/波动率）、Heston、Merton 跳跃扩散；
- 执行与做市策略的压力测试：生成大量价格路径评估策略分布；
- 方差缩减（对偶变量、控制变量、QMC）让同样算力下的估计误差下降数倍。

参考文献
--------
- Kloeden & Platen (1992). Numerical Solution of Stochastic Differential Equations. Springer.
- Glasserman (2003). Monte Carlo Methods in Financial Engineering. Springer.
- Lord, Koekkoek & van Dijk (2010). A comparison of biased simulation schemes for stochastic volatility models（全截断格式）。
- Andersen (2008). Simple and efficient simulation of the Heston stochastic volatility model（QE 格式）。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import norm

from .._utils import as_rng


# --------------------------------------------------------------------------- 通用格式
def euler_maruyama(drift: Callable, diffusion: Callable, x0: float, T: float, n_steps: int, n_paths: int = 1, seed=None) -> np.ndarray:
    """dX = b(t,X)dt + σ(t,X)dW 的 Euler–Maruyama，返回 (n_paths, n_steps+1)。"""
    rng = as_rng(seed)
    dt = T / n_steps
    X = np.zeros((n_paths, n_steps + 1)); X[:, 0] = x0
    for i in range(n_steps):
        t = i * dt
        dW = np.sqrt(dt) * rng.standard_normal(n_paths)
        X[:, i + 1] = X[:, i] + drift(t, X[:, i]) * dt + diffusion(t, X[:, i]) * dW
    return X


def milstein(drift: Callable, diffusion: Callable, d_diffusion: Callable, x0: float, T: float, n_steps: int, n_paths: int = 1, seed=None) -> np.ndarray:
    """Milstein 格式（需要 σ 对 x 的导数 σ'），强收敛阶 1。"""
    rng = as_rng(seed)
    dt = T / n_steps
    X = np.zeros((n_paths, n_steps + 1)); X[:, 0] = x0
    for i in range(n_steps):
        t = i * dt
        dW = np.sqrt(dt) * rng.standard_normal(n_paths)
        s = diffusion(t, X[:, i])
        X[:, i + 1] = X[:, i] + drift(t, X[:, i]) * dt + s * dW + 0.5 * s * d_diffusion(t, X[:, i]) * (dW ** 2 - dt)
    return X


# --------------------------------------------------------------------------- 常用模型
def simulate_gbm(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int = 1, seed=None, Z=None) -> np.ndarray:
    """几何布朗运动精确模拟；可传入标准正态矩阵 Z (n_paths, n_steps)（用于 QMC / 对偶变量）。"""
    rng = as_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps)) if Z is None else np.asarray(Z, dtype=float)
    logS = np.log(S0) + np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), np.exp(logS)])


def simulate_cir(x0: float, kappa: float, theta: float, sigma: float, T: float, n_steps: int, n_paths: int = 1, seed=None) -> np.ndarray:
    """CIR 过程 dX = κ(θ−X)dt + σ√X dW，全截断 Euler（保非负）。"""
    rng = as_rng(seed)
    dt = T / n_steps
    X = np.zeros((n_paths, n_steps + 1)); X[:, 0] = x0
    for i in range(n_steps):
        xp = np.maximum(X[:, i], 0)
        X[:, i + 1] = X[:, i] + kappa * (theta - xp) * dt + sigma * np.sqrt(xp * dt) * rng.standard_normal(n_paths)
    return np.maximum(X, 0.0)


def simulate_heston(S0: float, v0: float, mu: float, kappa: float, theta: float, xi: float, rho: float,
                    T: float, n_steps: int, n_paths: int = 1, seed=None) -> dict:
    """Heston 模型（全截断 Euler，相关布朗运动）。返回 ``S``、``v``。"""
    rng = as_rng(seed)
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps + 1)); v = np.zeros((n_paths, n_steps + 1))
    S[:, 0], v[:, 0] = S0, v0
    for i in range(n_steps):
        Z1 = rng.standard_normal(n_paths); Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n_paths)
        vp = np.maximum(v[:, i], 0)
        S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * vp) * dt + np.sqrt(vp * dt) * Z1)
        v[:, i + 1] = v[:, i] + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * Z2
    return dict(S=S, v=v)


def simulate_merton_jump(S0: float, mu: float, sigma: float, lam: float, jump_mean: float, jump_std: float,
                         T: float, n_steps: int, n_paths: int = 1, seed=None) -> np.ndarray:
    """Merton 跳跃扩散：对数跳跃 ~ N(jump_mean, jump_std²)，跳跃强度 λ（补偿使 E[S_T] = S0 e^{μT}）。"""
    rng = as_rng(seed)
    dt = T / n_steps
    k = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1
    Z = rng.standard_normal((n_paths, n_steps))
    N = rng.poisson(lam * dt, (n_paths, n_steps))
    J = N * jump_mean + np.sqrt(N) * jump_std * rng.standard_normal((n_paths, n_steps))
    logS = np.log(S0) + np.cumsum((mu - 0.5 * sigma ** 2 - lam * k) * dt + sigma * np.sqrt(dt) * Z + J, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), np.exp(logS)])


# --------------------------------------------------------------------------- 拟蒙特卡洛
def _primes(n):
    ps, c = [], 2
    while len(ps) < n:
        if all(c % p for p in ps):
            ps.append(c)
        c += 1
    return ps


def halton(n: int, d: int, scramble: bool = True, seed=None, skip: int = 20) -> np.ndarray:
    """Halton 低差异序列 (n, d)，可选随机平移（Cranley–Patterson）以获得无偏估计。"""
    rng = as_rng(seed)
    out = np.zeros((n, d))
    for j, b in enumerate(_primes(d)):
        idx = np.arange(skip, skip + n)
        f = 1.0 / b; x = np.zeros(n); i = idx.copy()
        while np.any(i > 0):
            x += f * (i % b); i //= b; f /= b
        out[:, j] = x
    if scramble:
        out = (out + rng.random(d)) % 1.0
    return out


def brownian_bridge_path(Z, T: float = 1.0) -> np.ndarray:
    """用布朗桥把标准正态 (n_paths, n_steps) 变换成布朗运动增量 (n_paths, n_steps)，
    使前几维（低差异序列质量最好的维度）决定路径的大尺度形状。"""
    Z = np.asarray(Z, dtype=float); n_paths, n = Z.shape
    dt = T / n
    W = np.zeros((n_paths, n + 1))
    W[:, n] = np.sqrt(T) * Z[:, 0]
    # 递归二分：按层级填充中点（第 0 维决定终点，后续维度依次决定各层中点）
    k = 1
    filled = {0, n}
    queue = [(0, n)]
    while queue:
        new_queue = []
        for (lo, hi) in queue:
            if hi - lo < 2:
                continue
            mid = (lo + hi) // 2
            if mid in filled:
                continue
            tl, tm, th = lo * dt, mid * dt, hi * dt
            mean = W[:, lo] + (tm - tl) / (th - tl) * (W[:, hi] - W[:, lo])
            var = (tm - tl) * (th - tm) / (th - tl)
            W[:, mid] = mean + np.sqrt(var) * Z[:, k]; k += 1
            filled.add(mid)
            new_queue += [(lo, mid), (mid, hi)]
        queue = new_queue
    return np.diff(W, axis=1) / np.sqrt(dt)   # 返回标准化增量，便于与 simulate_gbm 的 Z 接口对接


def qmc_normals(n_paths: int, n_steps: int, seed=None) -> np.ndarray:
    """Halton + 布朗桥 生成的“QMC 标准正态增量”(n_paths, n_steps)，可直接喂给 ``simulate_gbm(..., Z=)``。"""
    U = halton(n_paths, n_steps, scramble=True, seed=seed)
    Z = norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))
    return brownian_bridge_path(Z)


# --------------------------------------------------------------------------- 方差缩减
def antithetic_estimate(payoff: Callable, Z) -> dict:
    """对偶变量：对 Z 与 −Z 的 payoff 取平均。payoff(Z) -> (n_paths,)。"""
    Z = np.asarray(Z, dtype=float)
    y = 0.5 * (payoff(Z) + payoff(-Z))
    return dict(mean=float(y.mean()), std_err=float(y.std(ddof=1) / np.sqrt(len(y))))


def control_variate(Y, X, EX: float) -> dict:
    """控制变量：Y_cv = Y − β(X − E[X])，β = Cov(Y,X)/Var(X)。"""
    Y = np.asarray(Y, dtype=float); X = np.asarray(X, dtype=float)
    beta = np.cov(Y, X)[0, 1] / np.var(X, ddof=1)
    Ycv = Y - beta * (X - EX)
    return dict(mean=float(Ycv.mean()), std_err=float(Ycv.std(ddof=1) / np.sqrt(len(Y))), beta=float(beta),
                variance_reduction=float(np.var(Y) / max(np.var(Ycv), 1e-300)))
