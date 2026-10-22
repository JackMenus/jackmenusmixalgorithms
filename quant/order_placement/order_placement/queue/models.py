"""队列位置建模与限价/市价决策。

- QueuePositionTracker：基于行情推断自身挂单在价位队列中的位置。
  成交（trade）严格先进先出；撤单（cancel）按概率模型分配到“我前面/后面”，
  使用幂函数分配 f(x)=xⁿ/(xⁿ+(1−x)ⁿ)（与开源回测框架 hftbacktest 的 PowerProbQueueModel 同思路）。
- fill_probability：队列耗尽为速率 μ 的泊松过程、价格逆向跳动为速率 ν 的指数时间时，
  在价格移动前成交的概率 (μ/(μ+ν))^(Q+1)。
- queue_value：Moallemi & Yuan (2017) 思想的简化版：队列位置的期望价值
  = P(fill)·(半价差 − 逆向选择成本) − (1 − P(fill))·追单成本。
- cont_kukanov_allocation：Cont & Kukanov (2017) 多交易所市价/限价最优拆分，
  用随机次梯度法求解样本平均近似（SAA）。与路由库的交易所选择天然衔接。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


class QueuePositionTracker:
    def __init__(self, queue_ahead: float, my_qty: float, power: float = 2.0):
        self.ahead = float(queue_ahead)
        self.my_qty = float(my_qty)
        self.power = power
        self.filled = 0.0

    def _prob_ahead(self, ahead: float, total: float) -> float:
        if total <= 0:
            return 0.0
        x = min(max(ahead / total, 0.0), 1.0)
        n = self.power
        num = x**n
        den = num + (1 - x) ** n
        return num / den if den > 0 else 0.5

    def on_trade(self, qty: float) -> float:
        """该价位发生主动成交 qty，返回本次自身成交量。"""
        eat_ahead = min(self.ahead, qty)
        self.ahead -= eat_ahead
        rest = qty - eat_ahead
        fill = min(rest, self.my_qty - self.filled)
        self.filled += fill
        return fill

    def on_level_change(self, new_level_qty: float, old_level_qty: float) -> None:
        """该价位量减少（且不是成交）时视为撤单，按概率分配到我前面。

        level_qty 不含自身挂单。
        """
        dec = old_level_qty - new_level_qty
        if dec <= 0:
            return
        behind = max(old_level_qty - self.ahead, 0.0)
        p = self._prob_ahead(self.ahead, self.ahead + behind)
        self.ahead = max(self.ahead - p * dec, 0.0)
        self.ahead = min(self.ahead, new_level_qty)

    @property
    def done(self) -> bool:
        return self.filled >= self.my_qty - 1e-12


def fill_probability(queue_ahead: float, depletion_rate: float, adverse_rate: float, unit: float = 1.0) -> float:
    """P(自身订单在价格逆向移动前成交) = (μ/(μ+ν))^(Q/unit + 1)。"""
    if depletion_rate <= 0:
        return 0.0
    n = queue_ahead / unit + 1.0
    return float((depletion_rate / (depletion_rate + adverse_rate)) ** n)


@dataclass(frozen=True)
class PlacementDecision:
    use_limit: bool
    p_fill: float
    value_limit: float
    value_market: float


def queue_value(
    spread: float,
    queue_ahead: float,
    depletion_rate: float,
    adverse_rate: float,
    adverse_selection: float,
    chase_cost: Optional[float] = None,
    unit: float = 1.0,
) -> PlacementDecision:
    """比较“挂在最优价排队”与“立即吃单”的期望价值（相对中间价，正数为有利）。

    - 吃单：value_market = −spread/2
    - 排队：成交得 +spread/2 − 逆向选择成本；未成交需追单，损失 spread/2 + 价格逆向一跳(chase_cost)
    """
    p = fill_probability(queue_ahead, depletion_rate, adverse_rate, unit)
    chase = spread if chase_cost is None else chase_cost
    v_lim = p * (0.5 * spread - adverse_selection) - (1 - p) * (0.5 * spread + chase)
    v_mkt = -0.5 * spread
    return PlacementDecision(v_lim > v_mkt, p, v_lim, v_mkt)


@dataclass(frozen=True)
class VenueSpec:
    name: str
    queue: float        # 该交易所最优价排队量 Q_k
    rebate: float       # 挂单返佣 r_k（每股）
    outflow_mean: float  # 时间窗内该价位预计流出量 E[ξ_k]
    outflow_cv: float = 0.5  # 流出量变异系数（用于抽样）


def cont_kukanov_allocation(
    S: float,
    half_spread: float,
    taker_fee: float,
    venues: Sequence[VenueSpec],
    lambda_under: float,
    lambda_over: float,
    n_samples: int = 2000,
    n_iter: int = 400,
    lr: float = 0.5,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Cont-Kukanov 最优拆分：返回 (市价单量 M, 各交易所限价单量 L_k)。

    成本 V = (h+f)M − Σ(h+r_k)·min((ξ_k−Q_k)⁺, L_k) + λ_u(S−A)⁺ + λ_o(A−S)⁺，
    A = M + Σ min((ξ_k−Q_k)⁺, L_k)。ξ_k 用对数正态抽样。
    """
    rng = np.random.default_rng(seed)
    K = len(venues)
    Q = np.array([v.queue for v in venues])
    r = np.array([v.rebate for v in venues])
    mu = np.array([v.outflow_mean for v in venues])
    cv = np.array([v.outflow_cv for v in venues])
    s2 = np.log1p(cv**2)
    xi = np.exp(np.log(np.maximum(mu, 1e-12)) - 0.5 * s2 + np.sqrt(s2) * rng.standard_normal((n_samples, K)))
    h, f = half_spread, taker_fee
    M = 0.0
    L = np.full(K, S / max(K, 1))
    for it in range(n_iter):
        avail = np.maximum(xi - Q, 0.0)
        filled = np.minimum(avail, L)
        A = M + filled.sum(axis=1)
        under = (A < S).astype(float)
        over = (A > S).astype(float)
        pen = -lambda_under * under + lambda_over * over
        gM = (h + f) + pen.mean()
        binding = (avail > L).astype(float)
        gL = (binding * (-(h + r)[None, :] + pen[:, None])).mean(axis=0)
        step = lr * S / np.sqrt(it + 1) / max(lambda_under + lambda_over, 1e-12)
        M = float(np.clip(M - step * gM, 0, S))
        L = np.clip(L - step * gL, 0, S)
    return M, L
