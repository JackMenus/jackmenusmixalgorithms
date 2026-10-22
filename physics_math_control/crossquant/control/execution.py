"""
最优执行：Almgren–Chriss、Bertsimas–Lo、Obizhaeva–Wang 与传播子模型
=====================================================================

学科来源
--------
- Almgren–Chriss 是典型的 **变分法 / LQ 随机控制**：在“冲击成本（走得快贵）”与“价格风险（走得慢险）”
  之间做二次权衡，解是双曲正弦形状的清算曲线。
- Obizhaeva–Wang 与传播子（propagator）模型来自 **物理的线性响应理论**：订单像外力脉冲，价格像被
  敲击后以核函数 G(τ) 衰减回弹的介质；Bouchaud 等用它解释订单流长记忆与价格扩散的共存。
- 平方根冲击律 ΔP ≈ Y σ √(Q/V) 是“经验物理定律”式的发现（Torre 1997；Tóth 等 2011 用潜在流动性
  的“V 形”给出统计物理解释）。

量化映射
--------
- 母单切片：给定 X 股、T 时长、风险厌恶 λ，输出每期交易量；
- 成本估算：实施缺口的期望/方差，供 agent 做 alpha–成本权衡（见 ``crossquant.adapters``）；
- 瞬时冲击下的最优调度：min ½ xᵀGx s.t. 1ᵀx = X ⇒ x ∝ G^{-1}1（Gatheral–Schied–Slynko）。

参考文献
--------
- Almgren & Chriss (2000). Optimal Execution of Portfolio Transactions. J. Risk.
- Bertsimas & Lo (1998). Optimal Control of Execution Costs. J. Financial Markets.
- Obizhaeva & Wang (2013). Optimal trading strategy and supply/demand dynamics. JFM.
- Alfonsi, Fruth & Schied (2010). Optimal execution strategies in limit order books with general shape functions. QF.
- Gatheral, Schied & Slynko (2012). Transient linear price impact and Fredholm integral equations. Math. Finance.
- Bouchaud, Gefen, Potters & Wyart (2004). Fluctuations and response in financial markets. QF.
- Tóth et al. (2011). Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets. PRX.
- Fu, Horst & Xia (2025). Optimal Execution among N Traders with Transient Price Impact. arXiv:2501.09638.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- Almgren–Chriss
def almgren_chriss(X: float, T: float, N: int, sigma: float, eta: float, lam: float,
                   gamma: float = 0.0, epsilon: float = 0.0) -> dict:
    """Almgren–Chriss (2000) 离散时间最优清算。

    参数
    ----
    X : 初始持仓（股）
    T : 清算时长；N : 交易期数，τ = T/N
    sigma : 每单位时间的价格波动（价格单位）
    eta : 临时冲击系数，h(v) = ε sgn(v) + η v，v = n_k/τ
    lam : 风险厌恶（对方差的权重）
    gamma : 永久冲击系数 g(v) = γ v
    epsilon : 固定成本（半价差等）

    返回 dict：``times``、``holdings x_k``（含 x_0 = X, x_N = 0）、``trades n_k``、
    ``kappa``、``expected_cost``、``variance``、``utility = E + λV``。
    注：lam → 0 退化为 TWAP（线性清算）。
    """
    tau = T / N
    eta_t = eta - 0.5 * gamma * tau
    if eta_t <= 0:
        raise ValueError("η − γτ/2 必须为正（临时冲击需大于永久冲击的半步）")
    t = np.arange(N + 1) * tau
    if lam <= 0:
        x = X * (1 - t / T)
        kappa = 0.0
    else:
        kappa_tilde2 = lam * sigma ** 2 / eta_t
        kappa = np.arccosh(0.5 * kappa_tilde2 * tau ** 2 + 1) / tau
        x = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
    n = -np.diff(x)
    expected_cost = 0.5 * gamma * X ** 2 + epsilon * np.sum(np.abs(n)) + (eta_t / tau) * np.sum(n ** 2)
    variance = sigma ** 2 * tau * np.sum(x[1:] ** 2)
    return dict(times=t, holdings=x, trades=n, kappa=float(kappa), expected_cost=float(expected_cost),
                variance=float(variance), utility=float(expected_cost + lam * variance))


def almgren_chriss_frontier(X, T, N, sigma, eta, lams, gamma=0.0, epsilon=0.0) -> dict:
    """效率前沿：不同 λ 下的 (期望成本, 方差)。"""
    E, V = [], []
    for lam in lams:
        r = almgren_chriss(X, T, N, sigma, eta, lam, gamma, epsilon)
        E.append(r["expected_cost"]); V.append(r["variance"])
    return dict(lams=np.asarray(lams), expected_cost=np.array(E), variance=np.array(V))


def almgren_chriss_continuous(X: float, T: float, t, sigma: float, eta: float, lam: float) -> np.ndarray:
    """连续时间解 x(t) = X sinh(κ(T−t))/sinh(κT)，κ = sqrt(λσ²/η)。"""
    kappa = np.sqrt(lam * sigma ** 2 / eta)
    t = np.asarray(t, dtype=float)
    return X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)


# --------------------------------------------------------------------------- Bertsimas–Lo
def bertsimas_lo(X: float, N: int) -> np.ndarray:
    """Bertsimas–Lo (1998) 基准结果：线性永久冲击 + 随机游走价格 + 无风险厌恶 ⇒ 最优为等量拆单（TWAP）。

    有可预测信号时最优交易量在 TWAP 上叠加与信号成比例的调整（原文 Section 3），
    其一般形式可由 ``finite_horizon_lqr`` 之类的 LQ 递推求得。
    """
    return np.full(N, X / N)


# --------------------------------------------------------------------------- 瞬时冲击 / 传播子
def transient_cost_matrix(times, kernel: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """G_{ij} = G(|t_i − t_j|)：瞬时冲击下的成本二次型矩阵（成本 = ½ xᵀ G x）。"""
    times = np.asarray(times, dtype=float)
    return kernel(np.abs(times[:, None] - times[None, :]))


def optimal_schedule_transient(times, kernel: Callable[[np.ndarray], np.ndarray], X: float) -> dict:
    """瞬时线性冲击下的最优调度（Gatheral–Schied–Slynko）：min ½xᵀGx, 1ᵀx = X ⇒ x = X G^{-1}1 / (1ᵀG^{-1}1)。

    kernel 需为正定核（如指数核 e^{−ρτ}、幂律核 (1+τ/τ0)^{−β}，β<1 时正定性需 τ0>0）。
    返回 ``trades``、``cost``。
    """
    G = transient_cost_matrix(times, kernel)
    G = 0.5 * (G + G.T) + 1e-12 * np.eye(len(G))
    v = np.linalg.solve(G, np.ones(len(G)))
    x = X * v / v.sum()
    return dict(trades=x, cost=float(0.5 * x @ G @ x), G=G)


def obizhaeva_wang(X: float, T: float, N: int, rho: float, depth: float = 1.0) -> dict:
    """Obizhaeva–Wang (2013) / Alfonsi–Fruth–Schied (2010) 块状订单簿、指数回弹的闭式最优策略。

    N+1 个交易时点 t_n = nτ, τ = T/N，a = e^{−ρτ}：
        x_0 = x_N = X / ((N−1)(1−a) + 2),   x_n = (1−a) x_0  (0<n<N)
    即“首尾各砸一大块，中间以恒定速度小步交易”。连续极限：首尾块 X/(ρT+2)，中间速率 ρX/(ρT+2)。
    depth 为订单簿密度 q（每单位价格的挂单量），冲击核 G(τ) = e^{−ρτ}/q。
    返回 ``times``、``trades``、``cost``（= ½xᵀGx）。
    """
    tau = T / N
    a = np.exp(-rho * tau)
    x0 = X / ((N - 1) * (1 - a) + 2)
    trades = np.full(N + 1, (1 - a) * x0)
    trades[0] = trades[-1] = x0
    times = np.arange(N + 1) * tau
    G = transient_cost_matrix(times, lambda d: np.exp(-rho * d) / depth)
    return dict(times=times, trades=trades, cost=float(0.5 * trades @ G @ trades))


def power_law_kernel(l0: float, tau0: float, beta: float) -> Callable[[np.ndarray], np.ndarray]:
    """Bouchaud 传播子的幂律核 G(τ) = l0 / (1 + τ/τ0)^β（β≈0.5 附近使价格近似扩散）。"""
    return lambda d: l0 / (1.0 + np.asarray(d, dtype=float) / tau0) ** beta


def propagator_price_path(signed_volumes, kernel: Callable[[np.ndarray], np.ndarray],
                          noise_std: float = 0.0, delta: float = 0.5, seed=None) -> np.ndarray:
    """传播子模型价格路径：p_t = Σ_{s≤t} G(t−s) sgn(v_s)|v_s|^δ + 噪声（Bouchaud 等 2004）。

    signed_volumes : 带符号成交量序列（+买 −卖）；delta : 单笔冲击的凹性指数（经验 0.3~0.6）。
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(signed_volumes, dtype=float)
    T = len(v)
    impulses = np.sign(v) * np.abs(v) ** delta
    lags = np.arange(T)
    G = kernel(lags)
    p = np.convolve(impulses, G)[:T]
    if noise_std > 0:
        p = p + np.cumsum(noise_std * rng.standard_normal(T))
    return p


# --------------------------------------------------------------------------- 平方根冲击律
def square_root_impact(Q: float, adv: float, sigma_daily: float, Y: float = 0.5) -> float:
    """平方根冲击律：ΔP/P ≈ Y · σ_daily · sqrt(Q / ADV)（返回价格比例，乘 1e4 即 bps）。

    Y 经验上在 0.3~1 之间，跨资产类别（股票、期货、期权、比特币）稳健成立。
    """
    if adv <= 0:
        return float("nan")
    return float(Y * sigma_daily * np.sqrt(abs(Q) / adv))


def implementation_shortfall(exec_prices, exec_qty, decision_price: float, side: int = 1) -> dict:
    """实施缺口（Perold 1988）：按 bps 计，正数表示成本。"""
    exec_prices = np.asarray(exec_prices, dtype=float); exec_qty = np.asarray(exec_qty, dtype=float)
    vwap = np.sum(exec_prices * exec_qty) / np.sum(exec_qty)
    is_bps = side * (vwap - decision_price) / decision_price * 1e4
    return dict(vwap=float(vwap), shortfall_bps=float(is_bps))
