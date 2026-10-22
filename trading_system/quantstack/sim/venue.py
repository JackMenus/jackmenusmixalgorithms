"""多场所撮合/延迟模拟器，实现 ``VenueGateway``。

关键真实感来源：
  * 订单在 ``created_us + send_delay_us + 单程延迟`` 到达，到达时按 **到达时刻** 的报价成交——延迟越大，报价越可能已经陈旧；
  * IOC 到达时若报价已不利则拒单（router 据此统计 reject 与陈旧率）；
  * 挂单进入队列，队列前方量由到达时的本方可见量决定，之后随该场所的成交流（泊松）被消耗，或被对手价穿越而成交；
  * 成交回报再经过一次单程延迟才被 ``poll`` 到；``Fill.latency_us`` 记录往返延迟供路由层延迟模型学习；
  * 每笔成交对公允价施加永久 + 临时冲击。
"""
from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..contracts import US_PER_SEC, CancelRequest, Fill, OrderEvent, VenueOrder
from .market import SyntheticMarket


@dataclass(order=True)
class _Arrival:
    ts: float
    seq: int
    order: VenueOrder = field(compare=False)
    sent_us: float = field(compare=False, default=0.0)


@dataclass(order=True)
class _CancelArrival:
    ts: float
    seq: int
    request: CancelRequest = field(compare=False)


@dataclass
class _Resting:
    order: VenueOrder
    remaining: int
    queue_ahead: float
    posted_ts: float
    transit_us: float
    last_step_ts: float


class VenueSimulator:
    def __init__(self, market: SyntheticMarket, seed: int = 1, cancel_fee: float = 0.0):
        self.market = market
        self._rng = np.random.default_rng(seed)
        self._arrivals: List[_Arrival] = []
        self._cancels: List[_CancelArrival] = []
        self._resting: Dict[str, _Resting] = {}
        self._outbox: List[Tuple[float, int, OrderEvent]] = []
        self._cancel_on_arrival: set = set()
        self._seq = itertools.count(1)
        self._fill_seq = itertools.count(1)
        self.cancel_fee = cancel_fee
        self.stats = {"arrivals": 0, "fills": 0, "rejects": 0, "cancels": 0}

    # ------------------------------------------------------------------ 协议
    def submit(self, order: VenueOrder) -> None:
        sent = order.created_us + order.send_delay_us
        arrival = sent + self.market.latency_sample(order.venue)
        heapq.heappush(self._arrivals, _Arrival(arrival, next(self._seq), order, sent))

    def cancel(self, request: CancelRequest) -> None:
        if request.venue_order_id not in self._resting and not any(a.order.venue_order_id == request.venue_order_id for a in self._arrivals):
            return
        venue = self._resting[request.venue_order_id].order.venue if request.venue_order_id in self._resting else next(
            a.order.venue for a in self._arrivals if a.order.venue_order_id == request.venue_order_id
        )
        arrival = request.ts_us + self.market.latency_sample(venue)
        heapq.heappush(self._cancels, _CancelArrival(arrival, next(self._seq), request))

    def poll(self, ts_us: int) -> List[OrderEvent]:
        # 1) 到达的新订单
        while self._arrivals and self._arrivals[0].ts <= ts_us:
            a = heapq.heappop(self._arrivals)
            self._on_arrival(a)
        # 2) 到达的撤单
        while self._cancels and self._cancels[0].ts <= ts_us:
            c = heapq.heappop(self._cancels)
            self._on_cancel(c)
        # 3) 挂单队列演化
        for vid in list(self._resting):
            self._step_resting(self._resting[vid], ts_us)
        # 4) 回报（已经过返回延迟）
        out: List[OrderEvent] = []
        self._outbox.sort(key=lambda x: (x[0], x[1]))
        keep = []
        for deliver_ts, seq, ev in self._outbox:
            (out if deliver_ts <= ts_us else keep).append((deliver_ts, seq, ev) if deliver_ts > ts_us else ev)
        self._outbox = keep
        return out

    def outstanding(self) -> int:
        return len(self._resting) + len(self._arrivals)

    # ------------------------------------------------------------------ 内部
    def _emit(self, ev_ts: float, event: OrderEvent) -> None:
        deliver = ev_ts + self.market.latency_sample(event.venue)
        self._outbox.append((deliver, next(self._seq), event))

    def _fill(self, order: VenueOrder, qty: int, price: float, ts: float, transit_us: float, liquidity: str) -> None:
        spec = self.market.venue_spec(order.venue)
        fee_per_share = spec.maker_fee if liquidity == "maker" else spec.taker_fee
        report_latency = self.market.latency_sample(order.venue)
        fill = Fill(
            fill_id=f"X{next(self._fill_seq):08d}",
            venue_order_id=order.venue_order_id,
            slice_id=order.slice_id,
            parent_id=order.parent_id,
            symbol=order.symbol,
            venue=order.venue,
            side=order.side,
            quantity=int(qty),
            price=round(price, 6),
            ts_us=int(ts),
            fee=fee_per_share * qty,
            liquidity=liquidity,
            latency_us=transit_us + report_latency,
        )
        self.market.apply_trade(order.symbol, order.side, int(qty), aggressor=(liquidity == "taker"))
        if liquidity == "taker":
            self.market.consume(order.symbol, order.venue, order.side, int(qty))
        self.stats["fills"] += 1
        ev = OrderEvent(kind="fill", venue_order_id=order.venue_order_id, ts_us=int(ts), venue=order.venue, fill=fill)
        self._outbox.append((ts + report_latency, next(self._seq), ev))

    def _on_arrival(self, a: _Arrival) -> None:
        o = a.order
        self.stats["arrivals"] += 1
        if o.venue_order_id in self._cancel_on_arrival:
            self._cancel_on_arrival.discard(o.venue_order_id)
            self.stats["cancels"] += 1
            self._emit(a.ts, OrderEvent("cancel", o.venue_order_id, int(a.ts), o.venue, "cancelled_in_flight", remaining_quantity=o.quantity))
            return
        snap = self.market.snapshot(o.symbol, int(a.ts))
        q = snap.quote(o.venue)
        if q is None:
            self.stats["rejects"] += 1
            self._emit(a.ts, OrderEvent("reject", o.venue_order_id, int(a.ts), o.venue, "unknown_venue"))
            return
        spec = self.market.venue_spec(o.venue)
        far_px, far_sz = q.far_side(o.side)
        near_px, near_sz = q.near_side(o.side)
        side = o.side
        if o.order_type in ("IOC", "MARKET"):
            if o.order_type == "IOC" and (far_px - o.price) * side > 1e-9:
                self.stats["rejects"] += 1
                self._emit(a.ts, OrderEvent("reject", o.venue_order_id, int(a.ts), o.venue, "price_moved", remaining_quantity=o.quantity))
                return
            avail = int(far_sz * (1.0 + spec.hidden_liquidity))
            fill_qty = min(o.quantity, avail)
            if fill_qty > 0:
                self._fill(o, fill_qty, far_px, a.ts, a.ts - a.sent_us, "taker")
            remaining = o.quantity - fill_qty
            if remaining > 0 and o.order_type == "MARKET":
                # 市价单继续吃下一档，每档加一个 tick，直到保护价
                px = far_px
                while remaining > 0 and (px + side * self.market.tick - o.price) * side <= 1e-9:
                    px += side * self.market.tick
                    lvl = max(int(far_sz * 0.8), 100)
                    take = min(remaining, lvl)
                    self._fill(o, take, px, a.ts, a.ts - a.sent_us, "taker")
                    remaining -= take
            if remaining > 0:
                self.stats["cancels"] += 1
                self._emit(a.ts, OrderEvent("cancel", o.venue_order_id, int(a.ts), o.venue, "ioc_unfilled" if o.order_type == "IOC" else "protection_price", remaining_quantity=remaining))
            return
        # LIMIT / POST_ONLY / PEG
        price = o.price if o.order_type != "PEG" else near_px
        if (price - far_px) * side >= -1e-9:
            if o.order_type == "POST_ONLY":
                self.stats["rejects"] += 1
                self._emit(a.ts, OrderEvent("reject", o.venue_order_id, int(a.ts), o.venue, "post_only_would_cross", remaining_quantity=o.quantity))
                return
            take = min(o.quantity, int(far_sz * (1.0 + spec.hidden_liquidity)))
            if take > 0:
                self._fill(o, take, far_px, a.ts, a.ts - a.sent_us, "taker")
            remaining = o.quantity - take
            if remaining <= 0:
                return
            queue_ahead = 0.0
        else:
            remaining = o.quantity
            queue_ahead = float(near_sz) if abs(price - near_px) < 1e-9 else (0.0 if (price - near_px) * side > 0 else float(near_sz) * 3.0)
        self._resting[o.venue_order_id] = _Resting(order=o, remaining=remaining, queue_ahead=queue_ahead, posted_ts=a.ts, transit_us=a.ts - a.sent_us, last_step_ts=a.ts)
        self._emit(a.ts, OrderEvent("ack", o.venue_order_id, int(a.ts), o.venue, remaining_quantity=remaining))

    def _on_cancel(self, c: _CancelArrival) -> None:
        vid = c.request.venue_order_id
        r = self._resting.pop(vid, None)
        if r is None:
            # 订单可能还在途（比撤单先发但后到）：标记为到达即撤
            for a in self._arrivals:
                if a.order.venue_order_id == vid:
                    self._cancel_on_arrival.add(vid)
                    return
            return  # 已成交或从未挂上：撤单落空（too late）
        self.stats["cancels"] += 1
        self._emit(c.ts, OrderEvent("cancel", r.order.venue_order_id, int(c.ts), r.order.venue, c.request.reason or "user_cancel", remaining_quantity=r.remaining))

    def _step_resting(self, r: _Resting, ts_us: int) -> None:
        dt = ts_us - r.last_step_ts
        if dt <= 0:
            return
        r.last_step_ts = ts_us
        o = r.order
        snap = self.market.snapshot(o.symbol, ts_us)
        q = snap.quote(o.venue)
        if q is None:
            return
        side = o.side
        far_px, _ = q.far_side(side)
        near_px, near_sz = q.near_side(side)
        price = o.price if o.order_type != "PEG" else near_px
        # 对手价穿越了我们的价格：全部按我们的限价成交（我们是 maker）
        if (far_px - price) * side <= 1e-9:
            self._fill(o, r.remaining, price, ts_us, r.transit_us, "maker")
            self._resting.pop(o.venue_order_id, None)
            return
        spec = self.market.venue_spec(o.venue)
        st = self.market._state[o.symbol]
        per_sec = st.adv / (23_400.0) * spec.liquidity_share * 0.5  # 该场所本方被动成交流（股/秒）
        if abs(price - near_px) < 1e-9:
            traded = self._rng.poisson(per_sec * dt / US_PER_SEC * 1.0)
            r.queue_ahead -= traded
            if r.queue_ahead < 0:
                fill_qty = min(r.remaining, int(-r.queue_ahead))
                r.queue_ahead = 0.0
                if fill_qty > 0:
                    self._fill(o, fill_qty, price, ts_us, r.transit_us, "maker")
                    r.remaining -= fill_qty
                    if r.remaining <= 0:
                        self._resting.pop(o.venue_order_id, None)
        elif (price - near_px) * side > 0:
            # 我们的价格优于本方最优价（在价差内）：成为新的最优，队列为空
            r.queue_ahead = 0.0
            traded = self._rng.poisson(per_sec * dt / US_PER_SEC * 0.7)
            fill_qty = min(r.remaining, int(traded))
            if fill_qty > 0:
                self._fill(o, fill_qty, price, ts_us, r.transit_us, "maker")
                r.remaining -= fill_qty
                if r.remaining <= 0:
                    self._resting.pop(o.venue_order_id, None)
        else:
            r.queue_ahead *= math.exp(-0.02 * dt / US_PER_SEC)  # 前方撤单缓慢减少
