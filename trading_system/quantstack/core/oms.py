"""订单管理系统（OMS）：持仓账本 + 订单状态机 + 目标持仓差分。

OMS 是系统里唯一能改变持仓与现金的地方；它不做任何「聪明」的决策，只保证账是对的。
"""
from __future__ import annotations

import copy
import itertools
from typing import Dict, Iterable, List, Optional

from ..contracts import (
    ExecutionOrder,
    Fill,
    MarketSnapshot,
    OrderSlice,
    PortfolioSnapshot,
    Position,
    RoutingDecision,
    TargetPosition,
    VenueOrder,
)


class OrderManager:
    def __init__(self, initial_cash: float = 10_000_000.0, lot_size: int = 1, min_notional: float = 0.0):
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.fees_paid = 0.0
        self.lot_size = max(int(lot_size), 1)
        self.min_notional = float(min_notional)
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, ExecutionOrder] = {}
        self.slices: Dict[str, OrderSlice] = {}
        self.venue_orders: Dict[str, VenueOrder] = {}
        self.venue_remaining: Dict[str, int] = {}
        self.fills: List[Fill] = []
        self._order_seq = itertools.count(1)
        self._slice_seq = itertools.count(1)
        self._venue_seq = itertools.count(1)
        self._fill_seq = itertools.count(1)

    # ------------------------------------------------------------------ id 生成
    def next_order_id(self) -> str:
        return f"O{next(self._order_seq):06d}"

    def next_slice_id(self) -> str:
        return f"S{next(self._slice_seq):07d}"

    def next_venue_order_id(self) -> str:
        return f"V{next(self._venue_seq):08d}"

    def next_fill_id(self) -> str:
        return f"F{next(self._fill_seq):08d}"

    # ------------------------------------------------------------------ 账本
    def position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def mark_to_market(self, prices: Dict[str, float]) -> None:
        for sym, px in prices.items():
            if sym in self.positions and px is not None and px > 0:
                self.positions[sym].last_price = float(px)

    def snapshot(self, ts_us: int) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            ts_us=ts_us,
            cash=self.cash,
            positions={k: copy.copy(v) for k, v in self.positions.items() if v.quantity != 0 or v.realized_pnl != 0},
            fees_paid=self.fees_paid,
        )

    # ------------------------------------------------------------------ 目标 -> 母单
    def targets_to_orders(
        self,
        targets: Iterable[TargetPosition],
        market: Dict[str, MarketSnapshot],
        now_us: int,
        default_window_us: int,
    ) -> List[ExecutionOrder]:
        """把目标持仓与当前持仓做差，生成母单（不做风控，风控在 RiskGate）。"""
        orders: List[ExecutionOrder] = []
        for t in targets:
            current = self.position(t.symbol).quantity
            delta = int(t.target_quantity) - current
            delta = (delta // self.lot_size) * self.lot_size
            if delta == 0:
                continue
            snap = market.get(t.symbol)
            px = snap.mid if snap is not None and snap.mid > 0 else self.position(t.symbol).last_price
            if px <= 0:
                continue
            if abs(delta) * px < self.min_notional:
                continue
            end_us = t.deadline_us if t.deadline_us else now_us + default_window_us
            order = ExecutionOrder(
                order_id=self.next_order_id(),
                symbol=t.symbol,
                side=1 if delta > 0 else -1,
                quantity=abs(delta),
                start_us=now_us,
                end_us=max(end_us, now_us + 1),
                benchmark=t.benchmark,
                urgency=float(min(max(t.urgency, 0.0), 1.0)),
                max_participation=t.max_participation,
                alpha_bps=t.alpha_bps,
                alpha_horizon_us=t.alpha_horizon_us,
                created_us=now_us,
                arrival_price=px,
            )
            orders.append(order)
        return orders

    def register_order(self, order: ExecutionOrder) -> None:
        order.status = "working"
        self.orders[order.order_id] = order

    def register_slice(self, order_slice: OrderSlice) -> None:
        self.slices[order_slice.slice_id] = order_slice

    def register_routing(self, decision: RoutingDecision) -> None:
        for vo in decision.venue_orders:
            self.venue_orders[vo.venue_order_id] = vo
            self.venue_remaining[vo.venue_order_id] = vo.quantity

    def cancel_order(self, order_id: str, reason: str = "") -> None:
        o = self.orders.get(order_id)
        if o and not o.is_complete:
            o.status = "cancelled"

    def reject_order(self, order: ExecutionOrder, reason: str = "") -> None:
        order.status = "rejected"
        self.orders[order.order_id] = order

    # ------------------------------------------------------------------ 成交
    def apply_fill(self, fill: Fill) -> None:
        if fill.venue_order_id in self.venue_remaining:
            self.venue_remaining[fill.venue_order_id] = max(self.venue_remaining[fill.venue_order_id] - fill.quantity, 0)
        parent = self.orders.get(fill.parent_id)
        if parent is not None:
            parent.apply_fill(fill.quantity, fill.price)
        pos = self.position(fill.symbol)
        signed = fill.side * fill.quantity
        if pos.quantity == 0 or (pos.quantity > 0) == (signed > 0):
            # 加仓：更新均价
            total = pos.quantity + signed
            pos.avg_price = (pos.avg_price * abs(pos.quantity) + fill.price * abs(signed)) / abs(total) if total != 0 else 0.0
            pos.quantity = total
        else:
            # 减仓/反手：先实现盈亏
            closing = min(abs(pos.quantity), abs(signed))
            pos.realized_pnl += closing * (fill.price - pos.avg_price) * (1 if pos.quantity > 0 else -1)
            remaining = abs(signed) - closing
            pos.quantity += signed
            if pos.quantity == 0:
                pos.avg_price = 0.0
            elif remaining > 0:
                pos.avg_price = fill.price
        pos.last_price = fill.price
        self.cash -= signed * fill.price + fill.fee
        self.fees_paid += fill.fee
        self.fills.append(fill)

    # ------------------------------------------------------------------ 查询
    def open_orders(self) -> List[ExecutionOrder]:
        return [o for o in self.orders.values() if not o.is_complete]

    def nav(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())
