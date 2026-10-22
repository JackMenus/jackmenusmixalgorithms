"""限价单成交强度 λ(δ) = A·exp(-k·δ) 的估计。

做市模型（AS / GLFT）依赖参数 A、k。实盘中常用两种方法：
1. 直接法：统计在距中间价 δ 处挂单、单位时间内被成交的次数，做对数线性回归；
2. 间接法：统计市价单“穿透深度”分布（Avellaneda-Stoikov 2008 第 2 节）。
本模块实现直接法，同时提供基于穿透深度的估计。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class IntensityParams:
    A: float  # δ=0 处的到达强度（次/秒）
    k: float  # 衰减速度（1/价格单位）

    def rate(self, delta: float) -> float:
        return self.A * float(np.exp(-self.k * max(delta, 0.0)))


def fit_from_hit_counts(deltas: Sequence[float], hits: Sequence[float], exposure_s: Sequence[float]) -> IntensityParams:
    """直接法：deltas 为挂单距离，hits 为成交次数，exposure_s 为在该距离的暴露时长。"""
    d = np.asarray(deltas, float)
    rate = np.asarray(hits, float) / np.maximum(np.asarray(exposure_s, float), 1e-12)
    m = rate > 0
    if m.sum() < 2:
        raise ValueError("至少需要两个有成交的深度档")
    slope, intercept = np.polyfit(d[m], np.log(rate[m]), 1, w=np.sqrt(np.asarray(hits, float)[m]))
    return IntensityParams(A=float(np.exp(intercept)), k=float(-slope))


def fit_from_trade_depths(depths: Sequence[float], duration_s: float, grid: Sequence[float] | None = None) -> IntensityParams:
    """间接法：depths 为每笔市价单相对中间价的最远穿透距离。

    λ(δ) = (到达率) × P(穿透深度 ≥ δ)，用经验生存函数拟合指数。
    """
    x = np.sort(np.asarray(depths, float))
    if grid is None:
        grid = np.quantile(x, np.linspace(0.05, 0.9, 12))
    grid = np.asarray(grid, float)
    surv = np.array([(x >= g).sum() for g in grid], float) / duration_s
    return fit_from_hit_counts(grid, surv * duration_s, np.full_like(grid, duration_s))


def simulate_hits(params: IntensityParams, deltas: Sequence[float], exposure_s: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """测试/示例用：按给定参数模拟各档成交次数。"""
    d = np.asarray(deltas, float)
    lam = params.A * np.exp(-params.k * d)
    return rng.poisson(lam * exposure_s), np.full_like(d, exposure_s)
