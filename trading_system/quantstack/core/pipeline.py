"""交易流水线：把 alpha -> 组合 -> OMS/风控 -> 执行 -> 路由 -> 网关 -> 回报 串成一个事件循环。

同一份代码在模拟（SimClock + VenueSimulator）与实盘（WallClock + 真实网关）下运行；
agent 通过 ``plans`` 注入执行决策，通过事件日志与 TCA 读取结果。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..contracts import (
    US_PER_SEC,
    AlphaModel,
    CancelRequest,
    ExecutionOrder,
    ExecutionStrategy,
    ExecutionStrategyFactory,
    Incident,
    MarketDataFeed,
    MarketSnapshot,
    OrderEvent,
    OrderSlice,
    Panel,
    PortfolioConstructor,
    PortfolioSnapshot,
    RiskLimits,
    RiskVerdict,
    Router,
    RoutingDecision,
    SignalBatch,
    TargetPosition,
    TCAReport,
    VenueGateway,
    VenueStats,
)
from ..contracts.serialization import _to_jsonable
from ..reference.execution import choose_strategy
from .clock import Clock, SimClock
from .events import EventBus
from .oms import OrderManager
from .risk_gate import PreTradeRiskGate
from .tca import TCAEngine

PASSIVE_TYPES = ("LIMIT", "POST_ONLY", "PEG")


def _to_timestamp(as_of):
    import pandas as pd

    if isinstance(as_of, (int,)) and not isinstance(as_of, bool):
        return pd.Timestamp(int(as_of) * 1000, unit="ns")
    return pd.Timestamp(as_of)
StrategySelector = Callable[[ExecutionOrder, Optional[MarketSnapshot], Dict[str, str]], Tuple[str, Dict[str, float]]]


@dataclass
class PipelineConfig:
    step_us: int = US_PER_SEC                       # 事件循环步长（模拟）
    default_order_window_us: int = 30 * 60 * US_PER_SEC
    max_session_us: int = 60 * 60 * US_PER_SEC
    grace_us: int = 30 * US_PER_SEC                 # 母单窗口结束后的收尾时间
    passive_ttl_us: int = 15 * US_PER_SEC           # 被动子单默认存活时间（slice.ttl_us==0 时）
    flush_wait_us: int = 5 * US_PER_SEC             # 收尾时等待在途回报


@dataclass
class ActiveOrder:
    order: ExecutionOrder
    strategy: ExecutionStrategy
    strategy_name: str
    params: Dict[str, float] = field(default_factory=dict)
    started: bool = False
    done: bool = False


@dataclass
class SliceState:
    order_slice: OrderSlice
    parent_id: str
    venue_order_ids: List[str] = field(default_factory=list)
    filled: int = 0
    released: int = 0
    expires_us: Optional[int] = None
    closed: bool = False

    @property
    def outstanding(self) -> int:
        return max(self.order_slice.quantity - self.filled - self.released, 0)


@dataclass
class CycleResult:
    as_of: str
    signals: Optional[SignalBatch]
    targets: List[TargetPosition]
    orders: List[ExecutionOrder]
    rejected: List[Dict[str, object]]
    plans: Dict[str, Dict[str, object]]
    portfolio_start: PortfolioSnapshot
    portfolio_end: PortfolioSnapshot
    tca_reports: List[TCAReport]
    tca_summary: Dict[str, object]
    venue_stats: List[VenueStats]
    incidents: List[Incident]
    event_counts: Dict[str, int]
    duration_us: int

    def to_dict(self) -> Dict[str, object]:
        d = _to_jsonable(self)
        d["pnl"] = self.portfolio_end.nav - self.portfolio_start.nav
        d["nav_end"] = self.portfolio_end.nav
        return d

    def brief(self) -> Dict[str, object]:
        """给 LLM 看的精简摘要（避免把全部订单塞进上下文）。"""
        return {
            "as_of": self.as_of,
            "n_signals": len(self.signals.signals) if self.signals else 0,
            "n_targets": len(self.targets),
            "n_orders": len(self.orders),
            "n_rejected": len(self.rejected),
            "plans": {k: v.get("strategy") for k, v in self.plans.items()},
            "nav_start": round(self.portfolio_start.nav, 2),
            "nav_end": round(self.portfolio_end.nav, 2),
            "gross_exposure": round(self.portfolio_end.gross_exposure, 2),
            "net_exposure": round(self.portfolio_end.net_exposure, 2),
            "tca": self.tca_summary,
            "venue_stats": [dataclasses.asdict(v) for v in self.venue_stats],
            "incidents": [i.to_dict() for i in self.incidents],
            "event_counts": dict(self.event_counts),
            "duration_s": self.duration_us / US_PER_SEC,
        }


class TradingPipeline:
    def __init__(
        self,
        *,
        alpha_model: AlphaModel,
        portfolio: PortfolioConstructor,
        execution_factory: ExecutionStrategyFactory,
        router: Router,
        gateway: VenueGateway,
        feed: MarketDataFeed,
        clock: Optional[Clock] = None,
        oms: Optional[OrderManager] = None,
        risk_gate: Optional[PreTradeRiskGate] = None,
        tca: Optional[TCAEngine] = None,
        bus: Optional[EventBus] = None,
        config: Optional[PipelineConfig] = None,
        strategy_selector: Optional[StrategySelector] = None,
    ):
        self.alpha_model = alpha_model
        self.portfolio = portfolio
        self.execution_factory = execution_factory
        self.router = router
        self.gateway = gateway
        self.feed = feed
        self.clock = clock or SimClock(0)
        self.oms = oms or OrderManager()
        self.risk_gate = risk_gate or PreTradeRiskGate(RiskLimits())
        self.tca = tca or TCAEngine()
        self.bus = bus or EventBus()
        self.config = config or PipelineConfig()
        self.strategy_selector = strategy_selector or choose_strategy
        self.active: Dict[str, ActiveOrder] = {}
        self.slices: Dict[str, SliceState] = {}
        self._venue_to_slice: Dict[str, str] = {}
        self._silenced: set = set()
        self.halted = False
        self.last_signals: Optional[SignalBatch] = None
        self.last_targets: List[TargetPosition] = []
        self._alpha_fitted = False

    # ------------------------------------------------------------------ 工具
    def now(self) -> int:
        return self.clock.now_us()

    def snapshots(self, symbols: Sequence[str]) -> Dict[str, MarketSnapshot]:
        now = self.now()
        out = {}
        for s in symbols:
            try:
                out[s] = self.feed.snapshot(s, now)
            except KeyError:
                continue
        self.oms.mark_to_market({k: v.mid for k, v in out.items()})
        return out

    def portfolio_snapshot(self) -> PortfolioSnapshot:
        return self.oms.snapshot(self.now())

    # ------------------------------------------------------------------ 1. 信号
    def fit_alpha(self, panel: Panel, as_of=None, **kwargs) -> None:
        """只用 as_of 及之前的数据拟合 alpha 模型（避免未来信息）。"""
        train = panel if as_of is None else {k: v.loc[:_to_timestamp(as_of)] for k, v in panel.items()}
        self.alpha_model.fit(train, **kwargs)
        self._alpha_fitted = True
        self.bus.publish("alpha_fit", self.alpha_model.describe(), self.now(), "alpha")

    def generate_signals(self, panel: Panel, as_of=None) -> SignalBatch:
        if not self._alpha_fitted:
            self.fit_alpha(panel, as_of)
        batch = self.alpha_model.predict(panel, as_of=as_of)
        if not isinstance(batch, SignalBatch):
            batch = SignalBatch.from_dict(batch) if isinstance(batch, dict) else batch
        self.last_signals = batch
        self.bus.publish("signal_batch", {"model_id": batch.model_id, "n": len(batch.signals)}, self.now(), "alpha")
        return batch

    # ------------------------------------------------------------------ 2. 目标持仓
    def construct_targets(self, signals: SignalBatch) -> List[TargetPosition]:
        symbols = {s.symbol for s in signals.signals} | set(self.oms.positions)
        market = self.snapshots(sorted(symbols))
        targets = self.portfolio.construct(signals, self.portfolio_snapshot(), market, self.risk_gate.limits)
        self.last_targets = targets
        for t in targets:
            self.bus.publish("target", t, self.now(), "portfolio")
        return targets

    # ------------------------------------------------------------------ 3. 母单 + 风控
    def create_orders(self, targets: Sequence[TargetPosition]) -> Tuple[List[ExecutionOrder], List[Dict[str, object]]]:
        now = self.now()
        market = self.snapshots([t.symbol for t in targets])
        raw = self.oms.targets_to_orders(targets, market, now, self.config.default_order_window_us)
        approved: List[ExecutionOrder] = []
        rejected: List[Dict[str, object]] = []
        pf = self.portfolio_snapshot()
        for o in raw:
            verdict = self.risk_gate.check_order(o, pf, market.get(o.symbol))
            if not verdict.approved:
                self.oms.reject_order(o, "; ".join(verdict.reasons))
                rejected.append({"order_id": o.order_id, "symbol": o.symbol, "side": o.side, "quantity": o.quantity, "reasons": verdict.reasons})
                self.bus.publish("order_rejected", {"order": o, "reasons": verdict.reasons}, now, "risk")
                continue
            if verdict.adjusted_quantity and verdict.adjusted_quantity < o.quantity:
                o.quantity = verdict.adjusted_quantity
                self.bus.publish("order_resized", {"order_id": o.order_id, "reasons": verdict.reasons}, now, "risk")
            self.oms.register_order(o)
            approved.append(o)
            self.bus.publish("order", o, now, "oms")
        return approved, rejected

    # ------------------------------------------------------------------ 4. 执行规划（agent 可注入 plans）
    def start_execution(self, orders: Sequence[ExecutionOrder], plans: Optional[Dict[str, Dict[str, object]]] = None) -> List[ActiveOrder]:
        plans = plans or {}
        available = self.execution_factory.available()
        market = self.snapshots([o.symbol for o in orders])
        started: List[ActiveOrder] = []
        for o in orders:
            plan = plans.get(o.order_id) or {}
            name = plan.get("strategy") or o.strategy_hint
            params = dict(plan.get("params") or {})
            if not name or name not in available:
                name, default_params = self.strategy_selector(o, market.get(o.symbol), available)
                params = {**default_params, **params}
            if "urgency" in plan:
                o.urgency = float(plan["urgency"])
            o.strategy_hint = name
            o.params = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
            strategy = self.execution_factory.create(name, o, **params)
            ao = ActiveOrder(order=o, strategy=strategy, strategy_name=name, params=params)
            self.active[o.order_id] = ao
            started.append(ao)
            self.bus.publish("execution_plan", {"order_id": o.order_id, "strategy": name, "params": params, "urgency": o.urgency}, self.now(), "agent" if plan else "selector")
        return started

    # ------------------------------------------------------------------ 5. 事件循环
    def step(self) -> None:
        now = self.now()
        for ev in self.gateway.poll(now):
            self._handle_event(ev, now)
        if not self.halted:
            for ao in list(self.active.values()):
                if ao.done:
                    continue
                snap = self.feed.snapshot(ao.order.symbol, now)
                self.oms.mark_to_market({ao.order.symbol: snap.mid})
                if not ao.started:
                    self.tca.on_order_start(ao.order, snap, ao.strategy_name)
                    ao.started = True
                    new_slices = ao.strategy.start(ao.order, snap)
                else:
                    new_slices = ao.strategy.on_market(snap)
                self._dispatch(ao, new_slices, snap, now)
                self._cancel_requests(ao, ao.strategy.cancels(snap), now)
                self._expire_slices(ao, snap, now)
                if ao.strategy.is_complete() or now >= ao.order.end_us + self.config.grace_us:
                    self._finalize(ao, now)
            inc = self.risk_gate.check_pnl(self.portfolio_snapshot())
            if inc is not None:
                self.halt(inc.description, now)
        self.clock.advance(self.config.step_us)

    def run_until_done(self, max_us: Optional[int] = None) -> int:
        start = self.now()
        limit = max_us or self.config.max_session_us
        quiet_steps = 0
        while self.now() - start < limit:
            self.step()
            pending = any(not a.done for a in self.active.values())
            outstanding = getattr(self.gateway, "outstanding", lambda: 0)()
            if not pending and outstanding == 0:
                quiet_steps += 1
                if quiet_steps * self.config.step_us >= self.config.flush_wait_us:
                    break
            else:
                quiet_steps = 0
        return self.now() - start

    def halt(self, reason: str, now: Optional[int] = None) -> None:
        now = now if now is not None else self.now()
        self.halted = True
        for ao in self.active.values():
            if not ao.done:
                self._finalize(ao, now, reason=f"halt: {reason}")
        self.bus.publish("halt", {"reason": reason}, now, "risk")

    # ------------------------------------------------------------------ 6. 一次完整周期
    def run_cycle(self, panel: Panel, as_of=None, plans: Optional[Dict[str, Dict[str, object]]] = None, session_us: Optional[int] = None) -> CycleResult:
        start_us = self.now()
        pf_start = self.portfolio_snapshot()
        signals = self.generate_signals(panel, as_of)
        targets = self.construct_targets(signals)
        orders, rejected = self.create_orders(targets)
        actives = self.start_execution(orders, plans)
        self.run_until_done(session_us)
        pf_end = self.portfolio_snapshot()
        reports = [r for r in (self.tca.report(o.order_id) for o in orders) if r is not None]
        return CycleResult(
            as_of=str(as_of) if as_of is not None else str(signals.ts_us),
            signals=signals,
            targets=targets,
            orders=orders,
            rejected=rejected,
            plans={a.order.order_id: {"strategy": a.strategy_name, "params": a.params, "urgency": a.order.urgency} for a in actives},
            portfolio_start=pf_start,
            portfolio_end=pf_end,
            tca_reports=reports,
            tca_summary=self.tca.summary(),
            venue_stats=list(self.router.venue_stats()),
            incidents=list(self.risk_gate.incidents),
            event_counts=dict(self.bus.counts),
            duration_us=self.now() - start_us,
        )

    # ------------------------------------------------------------------ 内部：派发 / 回报 / 撤单 / 收尾
    def _dispatch(self, ao: ActiveOrder, new_slices: Sequence[OrderSlice], snap: MarketSnapshot, now: int) -> None:
        for s in new_slices or ():
            if s.parent_id != ao.order.order_id:
                s.parent_id = ao.order.order_id
            s.created_us = s.created_us or now
            # OMS 保护：在途量（其他未完成切片）+ 本切片 不得超过母单剩余量，防止执行/路由层超发导致超额成交
            in_flight = sum(st.outstanding for st in self.slices.values() if st.parent_id == ao.order.order_id and not st.closed)
            cap = ao.order.remaining - in_flight
            if cap <= 0:
                ao.strategy.on_event(OrderEvent("reject", s.slice_id, now, "", "oms:in_flight_cap", remaining_quantity=s.quantity), snap)
                continue
            if s.quantity > cap:
                reduction = s.quantity - cap
                s.quantity = cap
                ao.strategy.on_event(OrderEvent("cancel", s.slice_id, now, "", "oms:in_flight_cap_resize", remaining_quantity=reduction), snap)
            verdict = self.risk_gate.check_slice(s, snap, now)
            if not verdict.approved:
                self.bus.publish("slice_rejected", {"slice": s, "reasons": verdict.reasons}, now, "risk")
                ao.strategy.on_event(OrderEvent("reject", s.slice_id, now, "", "risk:" + "; ".join(verdict.reasons), remaining_quantity=s.quantity), snap)
                continue
            self.oms.register_slice(s)
            self.tca.on_slice(s)
            state = SliceState(order_slice=s, parent_id=ao.order.order_id)
            self.slices[s.slice_id] = state
            decision = self.router.route(s, snap)
            if not isinstance(decision, RoutingDecision):
                decision = RoutingDecision.from_dict(decision) if isinstance(decision, dict) else decision
            if not decision.venue_orders:
                state.released = s.quantity
                state.closed = True
                self.bus.publish("route_empty", {"slice_id": s.slice_id, "rationale": decision.rationale}, now, "router")
                ao.strategy.on_event(OrderEvent("reject", s.slice_id, now, "", "no_route", remaining_quantity=s.quantity), snap)
                continue
            # 路由层可以为提高成交概率而超额挂单，但总量不得超过母单剩余量（超出部分按顺序裁掉）
            allow = max(ao.order.remaining - in_flight, s.quantity)
            kept = []
            for vo in decision.venue_orders:
                if allow <= 0:
                    break
                if vo.quantity > allow:
                    vo.quantity = allow
                allow -= vo.quantity
                kept.append(vo)
            decision.venue_orders = kept
            for k, vo in enumerate(decision.venue_orders, start=1):
                vo.venue_order_id = f"{s.slice_id}-{k}"   # 约定：场所订单号保留切片前缀，执行层据此定位切片
                vo.slice_id = s.slice_id
                vo.parent_id = ao.order.order_id
                vo.symbol = vo.symbol or s.symbol
                vo.created_us = now
                state.venue_order_ids.append(vo.venue_order_id)
                self._venue_to_slice[vo.venue_order_id] = s.slice_id
            self.oms.register_routing(decision)
            routed = sum(vo.quantity for vo in decision.venue_orders)
            if routed < s.quantity:
                # 路由层只路由了部分数量（如 IOC 不超过可见量）：未路由余量立即释放，执行层下一步重新切片
                unrouted = s.quantity - routed
                state.released += unrouted
                self.bus.publish("route_partial", {"slice_id": s.slice_id, "requested": s.quantity, "routed": routed}, now, "router")
                ao.strategy.on_event(OrderEvent("cancel", s.slice_id, now, "", "unrouted", remaining_quantity=unrouted), snap)
            ttl = s.ttl_us or (self.config.passive_ttl_us if s.order_type in PASSIVE_TYPES else 0)
            state.expires_us = now + ttl if ttl else None
            for vo in decision.venue_orders:
                self.tca.on_venue_order(vo)
                self.gateway.submit(vo)
            self.bus.publish("route", {"slice_id": s.slice_id, "n_venues": len(decision.venue_orders), "expected_cost_bps": decision.expected_cost_bps, "expected_fill_ratio": decision.expected_fill_ratio, "diagnostics": decision.diagnostics}, now, "router")

    def _handle_event(self, ev: OrderEvent, now: int) -> None:
        self.router.on_event(ev)
        sid = self._venue_to_slice.get(ev.venue_order_id)
        state = self.slices.get(sid) if sid else None
        parent_id = state.parent_id if state else (ev.fill.parent_id if ev.fill else None)
        self.tca.on_event(ev, parent_id)
        ao = self.active.get(parent_id) if parent_id else None
        self.bus.publish(f"venue_{ev.kind}", ev, now, "gateway")
        if ev.kind in ("fill", "partial") and ev.fill is not None:
            self.oms.apply_fill(ev.fill)
            if state is not None:
                state.filled += ev.fill.quantity
            if ao is not None:
                snap = self.feed.snapshot(ao.order.symbol, now)
                self._dispatch(ao, ao.strategy.on_fill(ev.fill, snap), snap, now)
                if state is not None and state.filled >= state.order_slice.quantity and not state.closed:
                    self._close_slice(state, now, reason="slice_complete")  # 路由层可能超额挂单，切片完成后撤掉剩余
        elif ev.kind in ("reject", "cancel", "expire"):
            if ev.venue_order_id in self.oms.venue_remaining:
                self.oms.venue_remaining[ev.venue_order_id] = 0
            if state is not None and ev.venue_order_id not in self._silenced:
                rem = ev.remaining_quantity or 0
                state.released = min(state.released + rem, state.order_slice.quantity - state.filled)
                if state.outstanding == 0 and all(self.oms.venue_remaining.get(v, 0) <= 0 for v in state.venue_order_ids):
                    state.closed = True
                if ao is not None:
                    snap = self.feed.snapshot(ao.order.symbol, now)
                    self._dispatch(ao, ao.strategy.on_event(ev, snap), snap, now)
            self._silenced.discard(ev.venue_order_id)

    def _cancel_venue_orders(self, ids: Sequence[str], now: int, reason: str, silence: bool = True) -> int:
        n = 0
        for vid in ids:
            if self.oms.venue_remaining.get(vid, 0) <= 0:
                continue
            if silence:
                self._silenced.add(vid)
            self.gateway.cancel(CancelRequest(vid, now, reason))
            n += 1
        return n

    def _close_slice(self, state: SliceState, now: int, reason: str) -> None:
        state.closed = True
        self._cancel_venue_orders(state.venue_order_ids, now, reason)
        state.released = max(state.order_slice.quantity - state.filled, 0)

    def _cancel_requests(self, ao: ActiveOrder, requests: Sequence[CancelRequest], now: int) -> None:
        for req in requests or ():
            if req.venue_order_id in self.slices:
                # 执行层按切片撤单（挂单库约定 venue_order_id=slice_id）：展开为该切片下全部场所订单
                st = self.slices[req.venue_order_id]
                self._cancel_venue_orders(st.venue_order_ids, now, req.reason or "strategy_cancel", silence=False)
            else:
                self.gateway.cancel(CancelRequest(req.venue_order_id, now, req.reason))
            self.bus.publish("cancel_request", req, now, "execution")

    def _expire_slices(self, ao: ActiveOrder, snap: MarketSnapshot, now: int) -> None:
        for st in list(self.slices.values()):
            if st.parent_id != ao.order.order_id or st.closed or st.expires_us is None or now < st.expires_us:
                continue
            outstanding = st.outstanding
            self._close_slice(st, now, reason="ttl_expired")
            if outstanding > 0:
                ev = OrderEvent("expire", st.order_slice.slice_id, now, "", "ttl_expired", remaining_quantity=outstanding)
                self.bus.publish("slice_expired", ev, now, "pipeline")
                self._dispatch(ao, ao.strategy.on_event(ev, snap), snap, now)

    def _finalize(self, ao: ActiveOrder, now: int, reason: str = "") -> None:
        for st in self.slices.values():
            if st.parent_id == ao.order.order_id and not st.closed:
                self._close_slice(st, now, reason=reason or "order_done")
        o = ao.order
        o.status = "filled" if o.remaining == 0 else "cancelled"
        self.tca.on_order_done(o.order_id, now)
        ao.done = True
        self.bus.publish("order_done", {"order_id": o.order_id, "status": o.status, "filled": o.filled_quantity, "quantity": o.quantity, "reason": reason, "state": ao.strategy.state()}, now, "pipeline")
