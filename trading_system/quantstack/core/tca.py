"""交易成本分析（TCA）。每张母单一份报告，并汇总出 agent 复盘用的摘要与异常清单。"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Dict, List, Optional

from ..contracts import ExecutionOrder, Fill, MarketSnapshot, OrderEvent, OrderSlice, TCAReport, VenueOrder


@dataclass
class _OrderTrace:
    order: ExecutionOrder
    arrival_price: float
    start_us: int
    end_us: int = 0
    strategy: str = ""
    slices: int = 0
    venue_orders: int = 0
    fills: List[Fill] = field(default_factory=list)
    rejects: int = 0


class TCAEngine:
    def __init__(self, anomaly_threshold_bps: float = 25.0):
        self.traces: Dict[str, _OrderTrace] = {}
        self.anomaly_threshold_bps = anomaly_threshold_bps

    def on_order_start(self, order: ExecutionOrder, snapshot: MarketSnapshot, strategy: str = "") -> None:
        arrival = snapshot.mid if snapshot.mid > 0 else order.arrival_price
        order.arrival_price = arrival
        self.traces[order.order_id] = _OrderTrace(order=order, arrival_price=arrival, start_us=snapshot.ts_us, strategy=strategy)

    def on_slice(self, order_slice: OrderSlice) -> None:
        t = self.traces.get(order_slice.parent_id)
        if t:
            t.slices += 1

    def on_venue_order(self, vo: VenueOrder) -> None:
        t = self.traces.get(vo.parent_id)
        if t:
            t.venue_orders += 1

    def on_event(self, event: OrderEvent, parent_id: Optional[str] = None) -> None:
        if event.kind in ("fill", "partial") and event.fill is not None:
            t = self.traces.get(event.fill.parent_id)
            if t:
                t.fills.append(event.fill)
                t.end_us = max(t.end_us, event.ts_us)
        elif event.kind == "reject" and parent_id in self.traces:
            self.traces[parent_id].rejects += 1

    def on_order_done(self, order_id: str, ts_us: int) -> None:
        t = self.traces.get(order_id)
        if t:
            t.end_us = max(t.end_us, ts_us)

    # ------------------------------------------------------------------ 报告
    def report(self, order_id: str) -> Optional[TCAReport]:
        t = self.traces.get(order_id)
        if t is None:
            return None
        o = t.order
        filled = sum(f.quantity for f in t.fills)
        notional = sum(f.quantity * f.price for f in t.fills)
        avg = notional / filled if filled else 0.0
        arrival = t.arrival_price
        is_bps = 1e4 * o.side * (avg - arrival) / arrival if (filled and arrival > 0) else 0.0
        fees = sum(f.fee for f in t.fills)
        maker = sum(f.quantity for f in t.fills if f.liquidity == "maker")
        by_venue: DefaultDict[str, float] = defaultdict(float)
        venue_notional: DefaultDict[str, float] = defaultdict(float)
        for f in t.fills:
            by_venue[f.venue] += f.quantity
            venue_notional[f.venue] += f.quantity * f.price
        venue_breakdown = {v: q / filled for v, q in by_venue.items()} if filled else {}
        venue_slip = {
            v: 1e4 * o.side * (venue_notional[v] / q - arrival) / arrival for v, q in by_venue.items() if q > 0 and arrival > 0
        }
        return TCAReport(
            parent_id=o.order_id,
            symbol=o.symbol,
            side=o.side,
            quantity=o.quantity,
            filled_quantity=filled,
            arrival_price=arrival,
            avg_fill_price=avg,
            implementation_shortfall_bps=is_bps,
            fill_ratio=filled / o.quantity if o.quantity else 0.0,
            fees=fees,
            duration_us=max(t.end_us - t.start_us, 0),
            maker_ratio=maker / filled if filled else 0.0,
            n_slices=t.slices,
            n_venue_orders=t.venue_orders,
            venue_breakdown=venue_breakdown,
            venue_slippage_bps=venue_slip,
            strategy=t.strategy,
            extra={"rejects": float(t.rejects), "fees_bps": 1e4 * fees / notional if notional else 0.0},
        )

    def reports(self) -> List[TCAReport]:
        return [r for r in (self.report(k) for k in self.traces) if r is not None]

    def summary(self) -> Dict[str, object]:
        reps = self.reports()
        if not reps:
            return {"n_orders": 0}
        notional = [r.filled_quantity * r.avg_fill_price for r in reps]
        tot = sum(notional) or 1.0
        w_is = sum(r.implementation_shortfall_bps * n for r, n in zip(reps, notional)) / tot
        by_strategy: DefaultDict[str, List[float]] = defaultdict(list)
        for r in reps:
            by_strategy[r.strategy or "unknown"].append(r.implementation_shortfall_bps)
        anomalies = [
            {"order_id": r.parent_id, "symbol": r.symbol, "is_bps": round(r.implementation_shortfall_bps, 2), "fill_ratio": round(r.fill_ratio, 3)}
            for r in reps
            if r.implementation_shortfall_bps > self.anomaly_threshold_bps or r.fill_ratio < 0.5
        ]
        venue_slip: DefaultDict[str, List[float]] = defaultdict(list)
        for r in reps:
            for v, s in r.venue_slippage_bps.items():
                venue_slip[v].append(s)
        return {
            "n_orders": len(reps),
            "filled_notional": tot,
            "weighted_is_bps": round(w_is, 3),
            "avg_fill_ratio": round(sum(r.fill_ratio for r in reps) / len(reps), 4),
            "total_fees": round(sum(r.fees for r in reps), 2),
            "maker_ratio": round(sum(r.maker_ratio * n for r, n in zip(reps, notional)) / tot, 4),
            "is_bps_by_strategy": {k: round(sum(v) / len(v), 3) for k, v in by_strategy.items()},
            "venue_avg_slippage_bps": {k: round(sum(v) / len(v), 3) for k, v in venue_slip.items()},
            "anomalies": anomalies,
        }
