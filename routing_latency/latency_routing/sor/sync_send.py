"""同步到达（arrival synchronization）调度器。

背景：跨场所同时扫单时，如果订单先到达近处交易所，快速交易者（latency arbitrageur）会观察到成交，
在我们的订单到达远处交易所之前撤掉/上移报价，导致「看得见吃不着」。
RBC THOR (2010) 以及 IEX 的 speed bump 都基于同一原理：让多场所订单在极小时间窗内同时到达。

本模块在随机延迟（含抖动、突刺）下求解每个场所的发送延时 d_i，使得

    maximize   P( max_i(d_i + L_i) - min_i(d_i + L_i) <= w )
    tie-break  minimize  E[max_i(d_i + L_i)]

做法：对每个场所取共同分位数 q 的延迟 Q_i(q)，令 d_i = T(q) - Q_i(q)，
在 q 的网格上做蒙特卡洛评估并选最优。这比只按均值对齐稳健得多，因为重尾场所会被自然提前发送。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class SyncPlan:
    send_delay_us: Dict[str, float]
    expected_arrival_us: Dict[str, float]
    success_prob: float          # 所有子单在窗口 w 内到达的概率
    expected_completion_us: float
    quantile_used: float
    naive_success_prob: float    # 不做同步（全部立即发送）时的成功概率，作对照


def plan_synchronized_send(
    venues: Sequence[str],
    latency_samples: Dict[str, np.ndarray],
    window_us: float,
    q_grid: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
) -> SyncPlan:
    """为一组场所计算发送延时。

    ``latency_samples[v]`` 为该场所单程延迟的蒙特卡洛样本（各场所样本数须相同，视为独立）。
    """
    venues = list(venues)
    if not venues:
        raise ValueError("venues 不能为空")
    S = np.stack([np.asarray(latency_samples[v], dtype=float) for v in venues])  # (K,N)

    def evaluate(delays: np.ndarray):
        arr = S + delays[:, None]
        spread = arr.max(0) - arr.min(0)
        return float(np.mean(spread <= window_us)), float(np.mean(arr.max(0)))

    naive_p, _ = evaluate(np.zeros(len(venues)))
    best = None
    for q in q_grid:
        qs = np.quantile(S, q, axis=1)
        delays = qs.max() - qs
        p, comp = evaluate(delays)
        key = (round(p, 4), -comp)
        if best is None or key > best[0]:
            best = (key, q, delays, p, comp, qs)
    _, q, delays, p, comp, qs = best
    return SyncPlan(
        send_delay_us={v: float(d) for v, d in zip(venues, delays)},
        expected_arrival_us={v: float(d + m) for v, d, m in zip(venues, delays, S.mean(1))},
        success_prob=p,
        expected_completion_us=comp,
        quantile_used=float(q),
        naive_success_prob=naive_p,
    )
