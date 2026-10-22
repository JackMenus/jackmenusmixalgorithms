"""报价陈旧（stale quote）与被抢单风险模型。

核心思想：我们看到的报价在订单到达交易所之前可能已经变化。若报价更新服从强度为 λ 的泊松过程，
延迟为随机变量 L，则订单到达时报价仍有效的概率为

    P(survive) = E[exp(-λ L)]

当 L 来自经验分布/蒙特卡洛样本时直接取样本均值。这是对 Budish-Cramton-Shim (2015)
「连续限价簿中的狙击 (sniping) 竞赛」在单个订单层面的近似：越慢，被快速交易者抢先撤单/吃单的概率越高。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def survival_prob(update_rate_per_s: float, latency_samples_us: Sequence[float] | float) -> float:
    """订单到达时报价仍有效的概率。"""
    lam = update_rate_per_s * 1e-6  # 次/微秒
    if np.isscalar(latency_samples_us):
        return math.exp(-lam * float(latency_samples_us))
    x = np.asarray(latency_samples_us, dtype=float)
    return float(np.mean(np.exp(-lam * x)))


def expected_fill(displayed_size: int, update_rate_per_s: float, latency_samples_us, partial_decay: float = 0.5) -> float:
    """考虑陈旧风险后的预期可成交量。

    报价「失效」不一定意味着全部消失：``partial_decay`` 表示失效后平均剩余比例的补数
    （0.5 表示失效时平均还剩一半数量）。
    """
    p = survival_prob(update_rate_per_s, latency_samples_us)
    return displayed_size * (p + (1 - p) * (1 - partial_decay))


def sniping_risk(arrival_spread_us: np.ndarray, reaction_window_us: float) -> float:
    """多场所同时扫单时，到达时间离散度超过对手反应窗口的概率。

    ``reaction_window_us``：快速交易者从看到第一个场所成交到撤掉其他场所报价所需时间
    （通常≈两个场所间最快链路的单程延迟 + 其处理时间）。
    """
    return float(np.mean(np.asarray(arrival_spread_us) > reaction_window_us))
