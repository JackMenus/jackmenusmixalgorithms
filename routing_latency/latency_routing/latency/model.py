"""延迟建模。

包含：
- ``P2Quantile``：Jain & Chlamtac (1985) 的 P² 在线分位数估计，O(1) 内存，适合热路径。
- ``VenueLatencyModel``：按场所维护 EWMA 均值/方差、在线 P50/P99，以及一个
  「对数正态 + 尾部突刺」混合模型用于蒙特卡洛采样。
- ``estimate_clock_offset``：基于 NTP/Cristian 思路，用最小 RTT 样本估计本地与交易所时钟偏差。

延迟分布在实践中是右偏、多峰、有重尾的（网卡中断、GC、交易所撮合排队），
所以本模块刻意不用高斯假设，路由器应使用分位数（如 P95/P99）而不是均值做决策。
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class P2Quantile:
    """P² 算法：用 5 个标记点在线估计单个分位数，不存储样本。"""

    def __init__(self, q: float):
        if not 0.0 < q < 1.0:
            raise ValueError("q 必须在 (0,1) 内")
        self.q = q
        self._init: List[float] = []
        self.n = [0, 1, 2, 3, 4]
        self.np_ = [0.0, 2 * q, 4 * q, 2 + 2 * q, 4.0]
        self.dn = [0.0, q / 2, q, (1 + q) / 2, 1.0]
        self.h: List[float] = []

    def update(self, x: float) -> None:
        if len(self._init) < 5:
            self._init.append(x)
            if len(self._init) == 5:
                self.h = sorted(self._init)
            return
        h, n = self.h, self.n
        if x < h[0]:
            h[0] = x
            k = 0
        elif x >= h[4]:
            h[4] = x
            k = 3
        else:
            k = next(i for i in range(4) if h[i] <= x < h[i + 1])
        for i in range(k + 1, 5):
            n[i] += 1
        for i in range(5):
            self.np_[i] += self.dn[i]
        for i in (1, 2, 3):
            d = self.np_[i] - n[i]
            if (d >= 1 and n[i + 1] - n[i] > 1) or (d <= -1 and n[i - 1] - n[i] < -1):
                s = 1 if d > 0 else -1
                hp = self._parabolic(i, s)
                if h[i - 1] < hp < h[i + 1]:
                    h[i] = hp
                else:
                    h[i] = h[i] + s * (h[i + s] - h[i]) / (n[i + s] - n[i])
                n[i] += s

    def _parabolic(self, i: int, s: int) -> float:
        h, n = self.h, self.n
        return h[i] + s / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + s) * (h[i + 1] - h[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - s) * (h[i] - h[i - 1]) / (n[i] - n[i - 1])
        )

    @property
    def value(self) -> float:
        if len(self._init) < 5:
            if not self._init:
                return float("nan")
            return float(np.quantile(self._init, self.q))
        return self.h[2]


@dataclass
class _VenueStats:
    mean: float = 0.0
    var: float = 0.0
    count: int = 0
    log_mean: float = 0.0
    log_var: float = 0.0
    spikes: int = 0
    spike_sum: float = 0.0
    quantiles: Dict[float, P2Quantile] = field(default_factory=dict)


class VenueLatencyModel:
    """按交易场所维护单程（one-way）下单延迟的在线模型。

    参数
    ----
    halflife : EWMA 半衰期（样本数），越小对网络状态变化越敏感。
    spike_k  : 超过 ``exp(log_mean + spike_k * log_std)`` 的样本记为「突刺」，单独建模尾部。
    default_us : 尚无样本时使用的先验延迟。
    """

    TRACKED_Q = (0.5, 0.9, 0.99)

    def __init__(self, halflife: float = 200.0, spike_k: float = 3.0, default_us: float = 500.0):
        self.alpha = 1.0 - 0.5 ** (1.0 / halflife)
        self.spike_k = spike_k
        self.default_us = default_us
        self._s: Dict[str, _VenueStats] = defaultdict(self._new_stats)

    def _new_stats(self) -> _VenueStats:
        st = _VenueStats()
        st.quantiles = {q: P2Quantile(q) for q in self.TRACKED_Q}
        return st

    # ---- 在线更新 ----
    def observe(self, venue: str, latency_us: float) -> None:
        if latency_us <= 0:
            return
        st = self._s[venue]
        lx = math.log(latency_us)
        a = self.alpha if st.count > 10 else 1.0 / (st.count + 1)
        is_spike = st.count > 30 and lx > st.log_mean + self.spike_k * math.sqrt(max(st.log_var, 1e-12))
        if is_spike:
            st.spikes += 1
            st.spike_sum += latency_us
        else:
            d = lx - st.log_mean
            st.log_mean += a * d
            st.log_var = (1 - a) * (st.log_var + a * d * d)
        d2 = latency_us - st.mean
        st.mean += a * d2
        st.var = (1 - a) * (st.var + a * d2 * d2)
        st.count += 1
        for est in st.quantiles.values():
            est.update(latency_us)

    def observe_many(self, venue: str, xs: Iterable[float]) -> None:
        for x in xs:
            self.observe(venue, x)

    # ---- 查询 ----
    def spike_prob(self, venue: str) -> float:
        st = self._s.get(venue)
        if not st or st.count == 0:
            return 0.0
        return st.spikes / st.count

    def mean(self, venue: str) -> float:
        st = self._s.get(venue)
        return st.mean if st and st.count else self.default_us

    def quantile(self, venue: str, q: float) -> float:
        st = self._s.get(venue)
        if not st or st.count == 0:
            return self.default_us
        if q in st.quantiles:
            return st.quantiles[q].value
        # 非跟踪分位数：用对数正态近似
        z = _norm_ppf(q)
        return math.exp(st.log_mean + z * math.sqrt(max(st.log_var, 0.0)))

    def sample(self, venue: str, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """从「对数正态主体 + 突刺尾部」混合分布采样，用于蒙特卡洛路由评估。"""
        rng = rng or np.random.default_rng()
        st = self._s.get(venue)
        if not st or st.count < 5:
            return np.full(n, self.default_us, dtype=float)
        body = rng.lognormal(st.log_mean, math.sqrt(max(st.log_var, 1e-12)), n)
        p = self.spike_prob(venue)
        if p > 0:
            mask = rng.random(n) < p
            spike_mean = st.spike_sum / max(st.spikes, 1)
            body[mask] = rng.exponential(spike_mean, mask.sum())
        return body

    def venues(self) -> List[str]:
        return list(self._s.keys())

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for v, st in self._s.items():
            out[v] = {
                "count": st.count,
                "mean_us": st.mean,
                "std_us": math.sqrt(max(st.var, 0.0)),
                "p50_us": st.quantiles[0.5].value,
                "p90_us": st.quantiles[0.9].value,
                "p99_us": st.quantiles[0.99].value,
                "spike_prob": self.spike_prob(v),
            }
        return out


def estimate_clock_offset(samples: Sequence[Tuple[float, float, float, float]], best_frac: float = 0.1) -> Tuple[float, float]:
    """估计本地时钟相对交易所时钟的偏差（NTP 四时间戳法）。

    每个样本为 (t1 本地发送, t2 远端接收, t3 远端发送, t4 本地接收)。
    offset = ((t2-t1) + (t3-t4)) / 2，delay = (t4-t1) - (t3-t2)。
    只取 RTT 最小的 ``best_frac`` 比例样本（排队最少，最对称），返回 (offset, 不确定度=min_delay/2)。
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4 or len(arr) == 0:
        raise ValueError("samples 形状须为 (N,4)")
    t1, t2, t3, t4 = arr.T
    delay = (t4 - t1) - (t3 - t2)
    offset = ((t2 - t1) + (t3 - t4)) / 2.0
    k = max(1, int(len(arr) * best_frac))
    idx = np.argsort(delay)[:k]
    return float(np.median(offset[idx])), float(delay[idx].min() / 2.0)


def _norm_ppf(p: float) -> float:
    """标准正态分位数（Acklam 近似，误差 <1.2e-9），避免依赖 scipy。"""
    if not 0 < p < 1:
        raise ValueError("p 必须在 (0,1) 内")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
