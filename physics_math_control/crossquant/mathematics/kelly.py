"""
Kelly 准则、增长最优与遍历性
============================

学科来源
--------
信息论：Kelly (1956) 在贝尔实验室把 Shannon 信道容量与赌注增长率联系起来——最大化对数财富的
增长率等价于最大化信息率；Breiman (1961) 证明其渐近最优性；Thorp 把它带进 21 点与对冲基金。
Peters (2019) 的“遍历性经济学”强调：时间平均增长率 ≠ 期望收益（集合平均），Kelly 优化的是前者。

量化映射
--------
- **头寸规模**：f* = (μ − r)/σ²（连续时间），多资产 f* = Σ^{-1}(μ − r)；
- **分数 Kelly**：参数估计误差下用 c·f*（c≈0.25–0.5），牺牲少量增长换取大幅降低回撤；
- **回撤概率**：分数 c 的策略财富跌到初始的 α 倍的概率为 α^{2/c − 1}（全 Kelly 有 50% 概率减半）；
- **与 Merton 的关系**：Kelly = 对数效用的 Merton 比例，风险厌恶 γ 的 Merton 比例 = f*/γ。

核心公式
--------
二元赌局（胜率 p、赔率 b）：f* = (pb − (1−p))/b
连续：g(f) = r + f(μ−r) − ½f²σ²，f* = (μ−r)/σ²，g* = r + ½ Sharpe²
遍历性：E[W_T] = W_0 e^{μT}（集合平均）但 典型路径 W_T ≈ W_0 e^{(μ − σ²/2)T}（时间平均）

参考文献
--------
- Kelly (1956). A New Interpretation of Information Rate. Bell System Technical Journal.
- Thorp (2006). The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market. Handbook of Asset and Liability Management.
- MacLean, Thorp & Ziemba (2011). The Kelly Capital Growth Investment Criterion. World Scientific.
- Peters (2019). The ergodicity problem in economics. Nature Physics.
"""
from __future__ import annotations

import numpy as np

from .._utils import as_rng


def kelly_binary(p: float, b: float) -> float:
    """二元赌局 Kelly 比例（负值表示不下注）。"""
    return (p * b - (1 - p)) / b


def kelly_continuous(mu: float, sigma: float, r: float = 0.0) -> float:
    return (mu - r) / sigma ** 2


def kelly_multi_asset(mu, Sigma, r: float = 0.0) -> np.ndarray:
    """多资产 Kelly：f* = Σ^{-1}(μ − r)（允许杠杆与做空）。"""
    mu = np.asarray(mu, dtype=float); Sigma = np.asarray(Sigma, dtype=float)
    return np.linalg.solve(Sigma, mu - r)


def growth_rate(f, mu: float, sigma: float, r: float = 0.0):
    """连续时间对数增长率 g(f) = r + f(μ−r) − ½f²σ²。"""
    f = np.asarray(f, dtype=float)
    return r + f * (mu - r) - 0.5 * f ** 2 * sigma ** 2


def empirical_growth_rate(f: float, returns) -> float:
    """按历史收益计算 E[log(1 + f r)]（离散 Kelly 的目标函数）。"""
    r = np.asarray(returns, dtype=float)
    x = 1 + f * r
    return float(np.mean(np.log(x))) if np.all(x > 0) else -np.inf


def optimal_fraction_empirical(returns, f_grid=None) -> dict:
    r = np.asarray(returns, dtype=float)
    f_grid = np.linspace(0, 1 / max(-r.min(), 1e-6) * 0.99, 400) if f_grid is None else np.asarray(f_grid)
    g = np.array([empirical_growth_rate(f, r) for f in f_grid])
    return dict(f_star=float(f_grid[np.argmax(g)]), growth=g, f_grid=f_grid)


def drawdown_probability(alpha: float, c: float = 1.0) -> float:
    """分数 Kelly（f = c f*）财富曾经跌到初始 α 倍（0<α<1）的概率 = α^{2/c − 1}。"""
    return float(alpha ** (2 / c - 1))


def simulate_growth(f: float, mu: float, sigma: float, T: float = 10.0, n_steps: int = 2520, n_paths: int = 2000, seed=None) -> dict:
    """遍历性演示：返回 ``ensemble_mean``（集合平均 E[W_T]）、``median``（典型路径）、``time_avg_growth``（中位路径的增长率）。"""
    rng = as_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    logW = np.cumsum((f * mu - 0.5 * f ** 2 * sigma ** 2) * dt + f * sigma * np.sqrt(dt) * Z, axis=1)
    W_T = np.exp(logW[:, -1])
    return dict(ensemble_mean=float(W_T.mean()), median=float(np.median(W_T)),
                time_avg_growth=float(np.median(logW[:, -1]) / T), theory_ensemble=float(np.exp(f * mu * T)),
                theory_time_avg=float(f * mu - 0.5 * f ** 2 * sigma ** 2), wealth=W_T)
