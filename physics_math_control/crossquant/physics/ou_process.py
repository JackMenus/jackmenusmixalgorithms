"""
Ornstein–Uhlenbeck 过程：从 Langevin 方程到均值回复交易
======================================================

学科来源
--------
统计物理。Langevin (1908) 用 m dv/dt = −γv + 随机力 描述布朗粒子的速度；Uhlenbeck & Ornstein (1930)
给出其精确解：速度是高斯–马尔可夫过程，自相关指数衰减 e^{−κt}，平稳方差由涨落–耗散关系决定
（Einstein 关系 D = k_B T / γ）。这是物理里“有回复力的随机运动”的原型。

量化映射
--------
- **配对交易 / 统计套利**：价差 X_t 服从 dX = κ(θ − X)dt + σ dW。
  κ = 均值回复速度（半衰期 ln2/κ），θ = 长期均值，σ/√(2κ) = 平稳标准差（z-score 的分母）。
  “温度” σ²/(2κ) 越高，价差越容易偏离，可用它选择阈值。
- **利率、波动率、基差** 的建模（Vasicek 模型就是 OU）。
- **信号半衰期**：任何 alpha 信号的衰减速度都可以用 OU 的 κ 衡量，决定换手与持有期。

核心公式
--------
精确离散化（步长 Δ）：  X_{t+Δ} = θ + (X_t − θ) e^{−κΔ} + σ sqrt((1−e^{−2κΔ})/(2κ)) ε
等价 AR(1)：  X_{t+Δ} = a + b X_t + e，b = e^{−κΔ}，a = θ(1−b)，Var(e) = σ²(1−b²)/(2κ)
MLE 与 OLS 的点估计一致（高斯 AR(1)），κ̂ 在小样本下有向上偏差（高估回复速度），需警惕。

参考文献
--------
- Uhlenbeck & Ornstein (1930). On the Theory of the Brownian Motion. Phys. Rev.
- Vasicek (1977). An equilibrium characterization of the term structure. JFE.
- Avellaneda & Lee (2010). Statistical arbitrage in the US equities market. QF（PCA 残差的 OU 建模）。
- Leung & Li (2016). Optimal Mean Reversion Trading（含 OU 参数估计与最优阈值）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._utils import as_1d, as_rng


@dataclass
class OUParams:
    theta: float   # 长期均值
    kappa: float   # 均值回复速度（每单位时间）
    sigma: float   # 扩散系数

    @property
    def half_life(self) -> float:
        """半衰期 ln2/κ。"""
        return np.log(2) / self.kappa if self.kappa > 0 else np.inf

    @property
    def stationary_std(self) -> float:
        """平稳标准差 σ/√(2κ)。"""
        return self.sigma / np.sqrt(2 * self.kappa) if self.kappa > 0 else np.inf

    @property
    def temperature(self) -> float:
        """类比物理“温度”：平稳方差 σ²/(2κ)（涨落–耗散关系）。"""
        return self.stationary_std ** 2


def simulate_ou(theta: float, kappa: float, sigma: float, x0: float, dt: float, n: int, seed=None, n_paths: int = 1) -> np.ndarray:
    """精确离散化模拟 OU 路径，返回形状 (n+1,) 或 (n_paths, n+1)。"""
    rng = as_rng(seed)
    b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa)) if kappa > 0 else sigma * np.sqrt(dt)
    x = np.zeros((n_paths, n + 1)); x[:, 0] = x0
    eps = rng.standard_normal((n_paths, n))
    for t in range(n):
        x[:, t + 1] = theta + (x[:, t] - theta) * b + sd * eps[:, t]
    return x[0] if n_paths == 1 else x


def fit_ou(x, dt: float = 1.0) -> OUParams:
    """OU 参数估计（高斯 AR(1) 的 OLS/MLE 点估计）。

    若 AR(1) 系数 b ≥ 1（无均值回复），返回 kappa = 0 并以样本标准差近似 sigma。
    """
    x = as_1d(x, "x")
    X, Y = x[:-1], x[1:]
    n = len(X)
    Xm, Ym = X.mean(), Y.mean()
    b = np.sum((X - Xm) * (Y - Ym)) / np.sum((X - Xm) ** 2)
    a = Ym - b * Xm
    resid = Y - a - b * X
    var_e = np.sum(resid ** 2) / max(n - 2, 1)
    if b <= 0 or b >= 1:
        return OUParams(theta=float(Xm), kappa=0.0, sigma=float(np.sqrt(var_e / dt)))
    kappa = -np.log(b) / dt
    theta = a / (1 - b)
    sigma = np.sqrt(var_e * 2 * kappa / (1 - b ** 2))
    return OUParams(theta=float(theta), kappa=float(kappa), sigma=float(sigma))


def ou_log_likelihood(x, params: OUParams, dt: float = 1.0) -> float:
    """精确转移密度下的对数似然。"""
    x = as_1d(x, "x")
    b = np.exp(-params.kappa * dt)
    var = params.sigma ** 2 * (1 - b ** 2) / (2 * params.kappa)
    m = params.theta + (x[:-1] - params.theta) * b
    return float(-0.5 * np.sum(np.log(2 * np.pi * var) + (x[1:] - m) ** 2 / var))


def ou_zscore(x, params: OUParams) -> np.ndarray:
    """z = (x − θ)/平稳标准差，配对交易的开平仓信号。"""
    return (np.asarray(x, dtype=float) - params.theta) / params.stationary_std


def expected_path(params: OUParams, x0: float, t) -> np.ndarray:
    """条件期望 E[X_t | X_0 = x0] = θ + (x0 − θ)e^{−κt}。"""
    t = np.asarray(t, dtype=float)
    return params.theta + (x0 - params.theta) * np.exp(-params.kappa * t)


def hedge_ratio_ols(y, x, add_const: bool = True) -> dict:
    """OLS 对冲比率 y ≈ α + βx，返回 beta, alpha, spread（残差）。配合 ``fit_ou`` 做 Engle–Granger 两步法。"""
    y = as_1d(y, "y"); x = as_1d(x, "x")
    A = np.stack([x, np.ones_like(x)], axis=1) if add_const else x[:, None]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    beta = coef[0]; alpha = coef[1] if add_const else 0.0
    return dict(beta=float(beta), alpha=float(alpha), spread=y - beta * x - alpha)


def ou_signal_backtest(spread, params: OUParams, entry_z: float = 1.5, exit_z: float = 0.0, cost: float = 0.0) -> dict:
    """基于 z-score 的简单均值回复回测（单位头寸）：|z| > entry 开反向仓，|z| < exit 平仓。

    返回 ``pnl``（逐期）、``position``、``sharpe``（按期）、``n_trades``。
    """
    s = as_1d(spread, "spread")
    z = ou_zscore(s, params)
    pos = np.zeros(len(s)); p = 0
    trades = 0
    for t in range(len(s)):
        if p == 0 and z[t] > entry_z:
            p = -1; trades += 1
        elif p == 0 and z[t] < -entry_z:
            p = 1; trades += 1
        elif p == 1 and z[t] >= -exit_z:
            p = 0
        elif p == -1 and z[t] <= exit_z:
            p = 0
        pos[t] = p
    pnl = np.zeros(len(s))
    pnl[1:] = pos[:-1] * np.diff(s) - cost * np.abs(np.diff(pos))
    sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    return dict(pnl=pnl, position=pos, zscore=z, sharpe=float(sharpe), n_trades=trades)
