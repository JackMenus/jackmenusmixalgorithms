"""经验延迟模型（参考实现）。路由延迟库交付后由 ``latency_routing.latency.model.VenueLatencyModel`` 替换。"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import DefaultDict, Deque, Dict, Optional, Sequence

import numpy as np


class EmpiricalLatencyModel:
    """按场所保存最近 N 个单程延迟样本，分位数直接从样本估计，样本不足时退化为对数正态先验。"""

    def __init__(self, default_us: float = 500.0, prior_us: Optional[Dict[str, float]] = None, max_samples: int = 2000, seed: int = 0):
        self.default_us = float(default_us)
        self.prior_us: Dict[str, float] = dict(prior_us or {})
        self._samples: DefaultDict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=max_samples))
        self._rng = np.random.default_rng(seed)

    def observe(self, venue: str, latency_us: float) -> None:
        if latency_us > 0:
            self._samples[venue].append(float(latency_us))

    def _prior(self, venue: str) -> float:
        return self.prior_us.get(venue, self.default_us)

    def quantile(self, venue: str, q: float) -> float:
        s = self._samples.get(venue)
        if s and len(s) >= 5:
            return float(np.quantile(np.asarray(s), q))
        # 对数正态先验：中位数 = prior，σ=0.35
        return float(self._prior(venue) * np.exp(0.35 * _norm_ppf(q)))

    def sample(self, venue: str, n: int) -> Sequence[float]:
        s = self._samples.get(venue)
        if s and len(s) >= 5:
            return self._rng.choice(np.asarray(s), size=n, replace=True).tolist()
        return (self._prior(venue) * np.exp(0.35 * self._rng.standard_normal(n))).tolist()

    def n_samples(self, venue: str) -> int:
        return len(self._samples.get(venue, ()))


def _norm_ppf(q: float) -> float:
    """标准正态分位数（Acklam 近似，避免依赖 scipy）。"""
    q = min(max(q, 1e-9), 1 - 1e-9)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    if q < plow:
        t = np.sqrt(-2 * np.log(q))
        return (((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    if q > 1 - plow:
        t = np.sqrt(-2 * np.log(1 - q))
        return -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / ((((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1)
    t = q - 0.5
    r = t * t
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * t / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
