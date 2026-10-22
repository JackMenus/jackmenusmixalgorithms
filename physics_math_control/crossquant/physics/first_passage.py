"""
首达时间与障碍命中概率：扩散方程的吸收边界问题
===============================================

学科来源
--------
物理中带吸收边界的扩散（Smoluchowski、Chandrasekhar 1943）。布朗粒子何时首次碰到墙？
反射原理（Bachelier 1900、Lévy）给出带漂移布朗运动的首达时间分布（逆高斯分布），
双障碍问题由尺度函数 s(x) = e^{−2μx/σ²} 解析求出。

量化映射
--------
- **止盈/止损参数**：给定漂移 μ（信号强度）与波动 σ，算出先触止盈 (+a) 还是先触止损 (−b) 的概率、期望持仓时间；
- **三重障碍标签**（López de Prado）：{止盈, 止损, 到期} 三类标签的理论比例 ⇒ 校准障碍宽度，让标签平衡；
- **挂单成交概率**：价格在到期前触及限价的概率 P(max X ≤ T ≥ a)；
- **OU 价差的首达时间**：无闭式解，用蒙特卡洛（或 Ricciardi–Sato 级数）。

核心公式（X_t = μt + σW_t，X_0 = 0）
----------------------------------
单障碍首达密度：  f(t) = a / (σ sqrt(2π t³)) · exp(−(a − μt)² / (2σ² t))
T 内触及 a 的概率：P = Φ((−a + μT)/(σ√T)) + e^{2μa/σ²} Φ((−a − μT)/(σ√T))
双障碍（先碰 +a 而非 −b）：P_a = (1 − e^{2μb/σ²}) / (e^{−2μa/σ²} − e^{2μb/σ²})；μ = 0 时 P_a = b/(a+b)
期望退出时间：  E[τ] = (a P_a − b(1 − P_a)) / μ；μ = 0 时 E[τ] = ab/σ²

参考文献
--------
- Redner (2001). A Guide to First-Passage Processes. CUP.
- Chandrasekhar (1943). Stochastic Problems in Physics and Astronomy. Rev. Mod. Phys.
- López de Prado (2018). AFML Ch. 3（三重障碍法）。
- Ricciardi & Sato (1988). First-passage-time density and moments of the Ornstein-Uhlenbeck process. J. Appl. Prob.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .._utils import as_rng


def bm_first_passage_pdf(t, a: float, mu: float, sigma: float) -> np.ndarray:
    """带漂移布朗运动从 0 首次到达 a>0 的时间密度（逆高斯）。"""
    t = np.asarray(t, dtype=float)
    return a / (sigma * np.sqrt(2 * np.pi * t ** 3)) * np.exp(-(a - mu * t) ** 2 / (2 * sigma ** 2 * t))


def bm_prob_hit_before_T(a: float, mu: float, sigma: float, T: float) -> float:
    """P(max_{s≤T} X_s ≥ a)，a > 0。"""
    sT = sigma * np.sqrt(T)
    return float(norm.cdf((-a + mu * T) / sT) + np.exp(2 * mu * a / sigma ** 2) * norm.cdf((-a - mu * T) / sT))


def bm_two_barrier(a: float, b: float, mu: float, sigma: float) -> dict:
    """双障碍：上障碍 +a、下障碍 −b（a, b > 0）。返回 ``p_upper``、``p_lower``、``expected_exit_time``。"""
    if abs(mu) < 1e-12:
        p_up = b / (a + b)
        et = a * b / sigma ** 2
    else:
        k = 2 * mu / sigma ** 2
        p_up = (1 - np.exp(k * b)) / (np.exp(-k * a) - np.exp(k * b))
        et = (a * p_up - b * (1 - p_up)) / mu
    return dict(p_upper=float(p_up), p_lower=float(1 - p_up), expected_exit_time=float(et))


def triple_barrier_mc(mu: float, sigma: float, tp: float, sl: float, T: float, dt: float = 1 / 390,
                      n_paths: int = 20000, seed=None, x0: float = 0.0) -> dict:
    """三重障碍蒙特卡洛（布朗运动）：返回 P(先触止盈)、P(先触止损)、P(到期未触)、平均退出时间。

    tp, sl 为相对 x0 的正数距离。dt 越小离散化偏差越小（离散监测会低估触碰概率）。
    """
    rng = as_rng(seed)
    n = int(np.ceil(T / dt))
    x = np.full(n_paths, x0)
    alive = np.ones(n_paths, dtype=bool)
    outcome = np.zeros(n_paths, dtype=int)  # 0 到期, 1 止盈, 2 止损
    exit_t = np.full(n_paths, float(T))
    for i in range(n):
        x[alive] += mu * dt + sigma * np.sqrt(dt) * rng.standard_normal(alive.sum())
        hit_up = alive & (x >= x0 + tp); hit_dn = alive & (x <= x0 - sl)
        outcome[hit_up] = 1; outcome[hit_dn] = 2
        exit_t[hit_up | hit_dn] = (i + 1) * dt
        alive &= ~(hit_up | hit_dn)
        if not alive.any():
            break
    return dict(p_take_profit=float(np.mean(outcome == 1)), p_stop_loss=float(np.mean(outcome == 2)),
                p_timeout=float(np.mean(outcome == 0)), mean_exit_time=float(exit_t.mean()), outcome=outcome)


def ou_first_passage_mc(theta: float, kappa: float, sigma: float, x0: float, level: float, dt: float = 1 / 252,
                        T_max: float = 5.0, n_paths: int = 20000, seed=None) -> dict:
    """OU 过程从 x0 首次到达 level 的时间（蒙特卡洛）。返回 ``mean``、``median``、``q90``、``p_within_T``、``times``。"""
    rng = as_rng(seed)
    n = int(np.ceil(T_max / dt))
    b = np.exp(-kappa * dt); sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa))
    x = np.full(n_paths, x0); alive = np.ones(n_paths, dtype=bool); times = np.full(n_paths, np.nan)
    up = level > x0
    for i in range(n):
        x[alive] = theta + (x[alive] - theta) * b + sd * rng.standard_normal(alive.sum())
        hit = alive & ((x >= level) if up else (x <= level))
        times[hit] = (i + 1) * dt
        alive &= ~hit
        if not alive.any():
            break
    t = times[~np.isnan(times)]
    return dict(mean=float(np.mean(t)) if len(t) else np.nan, median=float(np.median(t)) if len(t) else np.nan,
                q90=float(np.quantile(t, 0.9)) if len(t) else np.nan, p_within_T=float(len(t) / n_paths), times=times)


def barrier_calibration(mu: float, sigma: float, T: float, target_tp_prob: float = 0.5, sl_ratio: float = 1.0,
                        grid=None, **mc_kwargs) -> dict:
    """反解障碍宽度：在网格上搜索 tp（sl = sl_ratio·tp），使 P(先触止盈) 最接近目标值（用于三重障碍标签平衡）。"""
    grid = np.linspace(0.25, 3.0, 12) * sigma * np.sqrt(T) if grid is None else np.asarray(grid)
    best = None
    for tp in grid:
        out = triple_barrier_mc(mu, sigma, tp, sl_ratio * tp, T, **mc_kwargs)
        err = abs(out["p_take_profit"] - target_tp_prob)
        if best is None or err < best[0]:
            best = (err, tp, out)
    return dict(tp=float(best[1]), sl=float(sl_ratio * best[1]), probs=best[2])
