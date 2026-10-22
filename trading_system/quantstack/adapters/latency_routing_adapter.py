"""路由延迟库（latency_routing）适配器。

latency_routing 的 ``route(ParentOrder, Sequence[VenueQuote])`` 与契约的 ``route(OrderSlice, MarketSnapshot)``
仅是容器不同：这里做无损互转，并把它的 ``ChildOrder`` 变成带 id 的 ``VenueOrder``（id 由流水线统一改写为
``{slice_id}-{k}``）。``on_event`` 直接透传（他们接受 quantstack 的 OrderEvent），``venue_stats`` 字段同名。
"""
from __future__ import annotations

from typing import List, Sequence

from ..contracts import MarketSnapshot, OrderEvent, OrderSlice, RoutingDecision, VenueOrder, VenueQuote, VenueStats


class LatencyRoutingAdapter:
    def __init__(self, router, venues: Sequence[str] = ()):
        import latency_routing as lr  # noqa: F401  (需要 /mnt/project-files/routing_latency 在 sys.path)

        self._lr = lr
        self.inner = router
        self.venues = list(venues)

    @classmethod
    def create(cls, venues: Sequence[str], **cfg) -> "LatencyRoutingAdapter":
        import latency_routing as lr

        return cls(lr.make_router(list(venues), **cfg), venues)

    def _quote(self, q: VenueQuote):
        d = q.to_dict()
        d.pop("schema_version", None)
        d.pop("symbol", None)
        return self._lr.VenueQuote(**d)

    def route(self, order_slice: OrderSlice, snapshot: MarketSnapshot) -> RoutingDecision:
        parent = self._lr.ParentOrder(
            symbol=order_slice.symbol,
            side=order_slice.side,
            quantity=order_slice.quantity,
            limit_price=order_slice.limit_price,
            urgency=order_slice.urgency,
            alpha_bps=order_slice.alpha_bps,
            alpha_horizon_us=float(order_slice.alpha_horizon_us),
            order_id=order_slice.slice_id,
        )
        # 透传契约里多出的字段（若他们的 dataclass 支持）
        for extra in ("order_type", "ttl_us", "parent_id"):
            if hasattr(parent, extra):
                setattr(parent, extra, getattr(order_slice, extra))
        quotes = [self._quote(q) for q in snapshot.quotes]
        dec = self.inner.route(parent, quotes)
        venue_orders: List[VenueOrder] = []
        for k, c in enumerate(dec.children, start=1):
            venue_orders.append(
                VenueOrder(
                    venue_order_id=f"{order_slice.slice_id}-{k}",
                    slice_id=order_slice.slice_id,
                    parent_id=order_slice.parent_id,
                    venue=c.venue,
                    symbol=order_slice.symbol,
                    side=c.side,
                    quantity=int(c.quantity),
                    price=float(c.price),
                    order_type=c.order_type if c.order_type in ("LIMIT", "MARKET", "IOC", "POST_ONLY", "PEG") else "IOC",
                    send_delay_us=float(getattr(c, "send_delay_us", 0.0)),
                    expected_arrival_us=float(getattr(c, "expected_arrival_us", 0.0)),
                    created_us=snapshot.ts_us,
                    route_reason=str(getattr(c, "reason", "")),
                )
            )
        return RoutingDecision(
            slice_id=order_slice.slice_id,
            venue_orders=venue_orders,
            expected_cost_bps=float(dec.expected_cost_bps),
            expected_fill_ratio=float(dec.expected_fill_ratio),
            diagnostics={k: float(v) for k, v in dict(getattr(dec, "diagnostics", {})).items() if isinstance(v, (int, float))},
            rationale="latency_routing",
        )

    def on_event(self, event: OrderEvent) -> None:
        self.inner.on_event(event)

    def venue_stats(self) -> List[VenueStats]:
        out = []
        for s in self.inner.venue_stats():
            d = s.to_dict() if hasattr(s, "to_dict") else dict(s.__dict__)
            d.pop("schema_version", None)
            out.append(VenueStats(**{k: d[k] for k in VenueStats.__dataclass_fields__ if k in d}))
        return out
