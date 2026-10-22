"""事前风控闸门（确定性）。所有母单与子单在进入执行/路由前必须通过。

设计原则：
  * 风控永远是代码而不是 LLM；agent 只能在 ``RiskLimits.tunable_bounds`` 允许的区间内改参数。
  * 任何触发 kill switch 的事件都会产出 ``Incident``，交给 agent 做强推理复盘，但停止交易本身不等人。
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from ..contracts import ExecutionOrder, Incident, MarketSnapshot, OrderSlice, PortfolioSnapshot, RiskLimits, RiskVerdict


class PreTradeRiskGate:
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._slice_times: Deque[int] = deque()
        self.incidents: List[Incident] = []
        self._incident_seq = 0
        self.start_nav: Optional[float] = None

    # ------------------------------------------------------------------ 母单
    def check_order(self, order: ExecutionOrder, portfolio: PortfolioSnapshot, snapshot: Optional[MarketSnapshot]) -> RiskVerdict:
        L = self.limits
        if L.kill_switch:
            return RiskVerdict.reject("kill_switch 已触发，拒绝一切新订单")
        reasons: List[str] = []
        px = snapshot.mid if snapshot is not None and snapshot.mid > 0 else order.arrival_price
        if px <= 0:
            return RiskVerdict.reject("无有效参考价")
        qty = order.quantity
        allowed = qty
        if qty > L.max_order_quantity:
            allowed = min(allowed, L.max_order_quantity)
            reasons.append(f"单笔数量 {qty} 超过上限 {L.max_order_quantity}")
        if qty * px > L.max_order_notional:
            allowed = min(allowed, int(L.max_order_notional / px))
            reasons.append(f"单笔名义金额 {qty * px:,.0f} 超过上限 {L.max_order_notional:,.0f}")
        cur = portfolio.quantity(order.symbol)
        new_qty = cur + order.side * qty
        if not L.allow_short and new_qty < 0:
            allowed = min(allowed, max(cur, 0)) if order.side < 0 else allowed
            reasons.append("禁止做空")
        if abs(new_qty) * px > L.max_position_notional:
            cap = int(L.max_position_notional / px)
            room = cap - abs(cur) if (cur == 0 or (cur > 0) == (order.side > 0)) else cap + abs(cur)
            allowed = min(allowed, max(room, 0))
            reasons.append(f"持仓名义金额将达 {abs(new_qty) * px:,.0f}，超过上限 {L.max_position_notional:,.0f}")
        gross_after = portfolio.gross_exposure - abs(cur) * px + abs(new_qty) * px
        if gross_after > L.max_gross_exposure:
            reasons.append(f"总敞口将达 {gross_after:,.0f}，超过上限 {L.max_gross_exposure:,.0f}")
            allowed = min(allowed, max(int((L.max_gross_exposure - (portfolio.gross_exposure - abs(cur) * px)) / px) - abs(cur) if cur * order.side >= 0 else allowed, 0))
        net_after = portfolio.net_exposure + order.side * qty * px
        if abs(net_after) > L.max_net_exposure:
            reasons.append(f"净敞口将达 {net_after:,.0f}，超过上限 {L.max_net_exposure:,.0f}")
            allowed = 0
        if snapshot is not None and snapshot.adv and order.max_participation > L.max_participation:
            order.max_participation = L.max_participation
            reasons.append(f"参与率上限收紧到 {L.max_participation:.0%}")
        if allowed <= 0:
            return RiskVerdict(approved=False, reasons=reasons or ["风控拒绝"])
        if allowed < qty:
            return RiskVerdict(approved=True, reasons=reasons, adjusted_quantity=allowed)
        return RiskVerdict(approved=True, reasons=reasons)

    # ------------------------------------------------------------------ 子单
    def check_slice(self, order_slice: OrderSlice, snapshot: MarketSnapshot, now_us: int) -> RiskVerdict:
        L = self.limits
        if L.kill_switch:
            return RiskVerdict.reject("kill_switch 已触发")
        mid = snapshot.mid
        if order_slice.limit_price is not None and mid > 0:
            dev_bps = 1e4 * (order_slice.limit_price - mid) / mid * order_slice.side
            if dev_bps > L.price_band_bps:
                return RiskVerdict.reject(f"限价偏离中间价 {dev_bps:.0f}bps，超过价格带 {L.price_band_bps:.0f}bps（疑似乌龙指）")
        while self._slice_times and now_us - self._slice_times[0] > 60_000_000:
            self._slice_times.popleft()
        if len(self._slice_times) >= L.max_slices_per_minute:
            return RiskVerdict.reject(f"每分钟子单数达到上限 {L.max_slices_per_minute}")
        self._slice_times.append(now_us)
        return RiskVerdict.ok()

    # ------------------------------------------------------------------ 盈亏监控 / kill switch
    def check_pnl(self, portfolio: PortfolioSnapshot) -> Optional[Incident]:
        if self.start_nav is None:
            self.start_nav = portfolio.nav
            return None
        loss = self.start_nav - portfolio.nav
        if loss > self.limits.max_daily_loss and not self.limits.kill_switch:
            return self.trip(portfolio.ts_us, f"当日亏损 {loss:,.0f} 超过上限 {self.limits.max_daily_loss:,.0f}", {"loss": loss})
        return None

    def trip(self, ts_us: int, reason: str, context: Optional[Dict[str, float]] = None) -> Incident:
        self.limits.kill_switch = True
        self._incident_seq += 1
        inc = Incident(
            incident_id=f"INC{self._incident_seq:04d}",
            ts_us=ts_us,
            severity="critical",
            kind="risk_breach",
            description=reason,
            context=context or {},
        )
        self.incidents.append(inc)
        return inc

    def reset_kill_switch(self) -> None:
        self.limits.kill_switch = False

    # ------------------------------------------------------------------ agent 调参入口（带边界）
    def set_limit(self, name: str, value: float) -> Tuple[bool, str]:
        bounds = self.limits.tunable_bounds.get(name)
        if bounds is None:
            return False, f"参数 {name} 不允许由 agent 修改"
        lo, hi = bounds
        if not (lo <= value <= hi):
            return False, f"参数 {name}={value} 超出允许区间 [{lo}, {hi}]"
        old = getattr(self.limits, name)
        setattr(self.limits, name, type(old)(value) if not isinstance(old, bool) else value)
        return True, f"{name}: {old} -> {getattr(self.limits, name)}"
