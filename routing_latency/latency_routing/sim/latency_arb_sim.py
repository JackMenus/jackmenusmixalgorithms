"""延迟套利对抗仿真。

场景：我们在 K 个场所同时扫单。场所上的报价属于一个快速做市商/套利者，
它在看到「任意一个场所发生我方成交」之后，经过 reaction_window_us 撤掉其余场所的报价。
另外每个场所的报价按泊松强度独立刷新（与我们无关的陈旧风险）。

用来对比：
- 立即全部发送（naive）
- 同步到达调度（sync_send）
两种方式的实际成交率。这是 THOR 类机制效果的最小可复现实验。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..interfaces import ChildOrder


@dataclass
class SimVenue:
    name: str
    displayed: int
    latency_samples: np.ndarray        # 我方单程延迟样本
    quote_update_rate: float = 0.0      # 次/秒，与我方无关的随机撤单


def simulate_sweep(
    children: Sequence[ChildOrder],
    venues: Dict[str, SimVenue],
    reaction_window_us: float,
    n_trials: int = 2000,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    rng = rng or np.random.default_rng(1)
    total_sent = sum(c.quantity for c in children)
    filled = np.zeros(n_trials)
    for t in range(n_trials):
        arrivals = []
        for c in children:
            v = venues[c.venue]
            lat = v.latency_samples[rng.integers(len(v.latency_samples))]
            arrivals.append((c.send_delay_us + lat, c))
        arrivals.sort(key=lambda x: x[0])
        first_hit = None
        got = 0
        for arr_t, c in arrivals:
            v = venues[c.venue]
            # 独立的报价刷新
            if v.quote_update_rate > 0 and rng.random() > np.exp(-v.quote_update_rate * 1e-6 * arr_t):
                continue
            # 对手已在反应窗口后撤单
            if first_hit is not None and arr_t - first_hit > reaction_window_us:
                continue
            q = min(c.quantity, v.displayed)
            if q > 0:
                got += q
                if first_hit is None:
                    first_hit = arr_t
        filled[t] = got
    return {
        "fill_ratio_mean": float(filled.mean() / max(total_sent, 1)),
        "fill_ratio_p10": float(np.quantile(filled, 0.1) / max(total_sent, 1)),
        "full_fill_prob": float(np.mean(filled >= total_sent)),
    }
