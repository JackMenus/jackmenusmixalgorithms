"""把执行轨迹落地为子单：被动挂单 + 落后追单 + alpha/队列感知。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..interfaces import (
    AlphaSignal,
    BookSnapshot,
    ChildOrder,
    Fill,
    LatencyProfile,
    OrderType,
    ParentOrder,
    Side,
    StrategyState,
)
from ..queue.models import queue_value


@dataclass
class ExecutorConfig:
    catch_up_ratio: float = 1.0   # 落后量超过“一个区间计划量 × 该比例”时转为主动吃单
    alpha_urgency_bps: float = 2.0  # alpha 对我方不利超过该阈值（bps）时提前执行
    depletion_rate: float = 5.0   # 队列消耗速率 μ（股/秒），可由行情在线估计
    adverse_rate: float = 0.2     # 价格逆向跳动速率 ν（次/秒）
    adverse_selection: float = 0.0  # 挂单成交后的平均逆向选择（价格单位）
    min_qty: float = 1.0


class ScheduledExecutor:
    """按给定执行轨迹 n（每区间量）执行母单。实现 OrderPlacementStrategy 协议。"""

    name = "scheduled_executor"

    def __init__(self, parent: ParentOrder, schedule: np.ndarray, config: Optional[ExecutorConfig] = None):
        self.parent = parent
        self.schedule = np.asarray(schedule, float)
        self.cum_target = np.concatenate([[0.0], np.cumsum(self.schedule)])
        self.cfg = config or ExecutorConfig()
        self.executed = 0.0
        self.working: Optional[ChildOrder] = None

    # 计划累计成交量（区间内线性插值）
    def target_executed(self, t: float) -> float:
        N = len(self.schedule)
        u = (t - self.parent.start_ts) / max(self.parent.duration, 1e-12) * N
        u = float(np.clip(u, 0, N))
        i = int(min(np.floor(u), N - 1))
        return float(self.cum_target[i] + (u - i) * self.schedule[i])

    def remaining(self) -> float:
        return max(self.parent.qty - self.executed, 0.0)

    def on_book(self, book: BookSnapshot, state: StrategyState,
                alpha: Optional[AlphaSignal] = None, latency: Optional[LatencyProfile] = None) -> List[ChildOrder]:
        rem = self.remaining()
        if rem <= 1e-12:
            return self._cancel_working(book)
        side = self.parent.side
        slice_qty = max(self.schedule.mean(), self.cfg.min_qty)
        target = self.target_executed(book.ts)
        # alpha 对我方不利（买入时看涨/卖出时看跌）→ 提前一个区间
        if alpha is not None and alpha.value * side.sign * alpha.confidence > self.cfg.alpha_urgency_bps:
            target = min(target + slice_qty, self.parent.qty)
        behind = target - self.executed
        out: List[ChildOrder] = []
        at_end = book.ts >= self.parent.end_ts
        if at_end or behind > self.cfg.catch_up_ratio * slice_qty:
            qty = rem if at_end else min(behind, rem)
            px = book.best_ask if side is Side.BUY else book.best_bid
            if self.parent.limit_price is not None:
                px = min(px, self.parent.limit_price) if side is Side.BUY else max(px, self.parent.limit_price)
            out += self._cancel_working(book)
            out.append(ChildOrder(book.symbol, side, qty, px, OrderType.IOC, book.ts, self.parent.parent_id,
                                  self.name, urgency=1.0))
            return out
        # 被动：判断挂单是否划算
        near = book.best_bid if side is Side.BUY else book.best_ask
        q_ahead = book.bids[0][1] if side is Side.BUY else book.asks[0][1]
        dec = queue_value(book.spread, q_ahead, self.cfg.depletion_rate, self.cfg.adverse_rate,
                          self.cfg.adverse_selection)
        px = near
        if not dec.use_limit and book.spread_ticks > 1:  # 价差较宽时改进一跳，抢占队首
            px = near + side.sign * book.tick_size
        if self.working is not None and abs(self.working.price - px) < 1e-12:
            return out
        out += self._cancel_working(book)
        qty = min(slice_qty, rem)
        self.working = ChildOrder(book.symbol, side, qty, px, OrderType.LIMIT, book.ts, self.parent.parent_id,
                                  self.name, urgency=max(0.0, min(1.0, behind / slice_qty)))
        out.append(self.working)
        return out

    def _cancel_working(self, book: BookSnapshot) -> List[ChildOrder]:
        if self.working is None:
            return []
        c = ChildOrder(book.symbol, self.working.side, self.working.qty, self.working.price, OrderType.CANCEL,
                       book.ts, self.parent.parent_id, self.name, order_id=self.working.order_id)
        self.working = None
        return [c]

    def on_fill(self, fill: Fill, state: StrategyState) -> None:
        self.executed += fill.qty
        sgn = fill.side.sign
        state.inventory += sgn * fill.qty
        state.cash -= sgn * fill.qty * fill.price
        if self.working is not None and fill.order_id == self.working.order_id:
            self.working.qty -= fill.qty
            if self.working.qty <= 1e-12:
                self.working = None
