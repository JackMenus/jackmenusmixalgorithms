"""成本-延迟感知的智能路由（参考实现）。

吃单（IOC/MARKET）：按「报价 + 费用 + 报价陈旧惩罚」排序分配到各场所的可见量，并计算发送延时让各场所
订单同步到达（以最慢场所的 P50 为基准），降低被延迟套利者「看见先到的子单后撤单」的风险。
挂单（LIMIT/POST_ONLY）：选择返佣最高、队列最短的场所挂全量。

路由延迟库交付后由 ``latency_routing.make_router`` 替换；本实现保证流水线在此之前即可端到端运行。
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence

from ..contracts import LatencyModel, MarketSnapshot, OrderEvent, OrderSlice, RoutingDecision, VenueOrder, VenueStats
from .latency import EmpiricalLatencyModel


class CostLatencyRouter:
    def __init__(
        self,
        venues: Sequence[str],
        latency_model: Optional[LatencyModel] = None,
        sync_enabled: bool = True,
        oversize_cap: float = 1.3,
        stale_penalty_bps: float = 1.0,
    ):
        self.venues = list(venues)
        self.latency = latency_model or EmpiricalLatencyModel()
        self.sync_enabled = sync_enabled
        self.oversize_cap = oversize_cap
        self.stale_penalty_bps = stale_penalty_bps
        self._seq = itertools.count(1)
        self._ref_mid: Dict[str, float] = {}       # venue_order_id -> 路由时中间价
        self._sent_qty: DefaultDict[str, int] = defaultdict(int)
        self._fill_qty: DefaultDict[str, int] = defaultdict(int)
        self._maker_qty: DefaultDict[str, int] = defaultdict(int)
        self._rejects: DefaultDict[str, int] = defaultdict(int)
        self._slip_num: DefaultDict[str, float] = defaultdict(float)
        self._fills: DefaultDict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------ 路由
    def route(self, order_slice: OrderSlice, snapshot: MarketSnapshot) -> RoutingDecision:
        quotes = [q for q in snapshot.quotes if q.venue in self.venues] or list(snapshot.quotes)
        if not quotes:
            return RoutingDecision(order_slice.slice_id, [], expected_fill_ratio=0.0, rationale="无可用报价")
        if order_slice.order_type in ("IOC", "MARKET"):
            return self._route_aggressive(order_slice, snapshot, quotes)
        return self._route_passive(order_slice, snapshot, quotes)

    def _route_aggressive(self, s: OrderSlice, snap: MarketSnapshot, quotes) -> RoutingDecision:
        mid = snap.mid
        side = s.side
        cands = []
        for q in quotes:
            px, size = q.far_side(side)
            if s.order_type == "IOC" and s.limit_price is not None and (px - s.limit_price) * side > 1e-12:
                continue  # 超出限价的场所不去
            p50 = self.latency.quantile(q.venue, 0.5)
            p_stale = 1.0 - math.exp(-q.quote_update_rate * p50 / 1e6)
            eff = px + side * q.taker_fee + side * mid * 1e-4 * self.stale_penalty_bps * p_stale
            cands.append((eff * side, q, px, size, p50, p_stale))
        if not cands:
            return RoutingDecision(s.slice_id, [], expected_fill_ratio=0.0, rationale="所有场所报价超出限价")
        cands.sort(key=lambda c: c[0])
        remaining = s.quantity
        alloc = []
        for eff, q, px, size, p50, p_stale in cands:
            if remaining <= 0:
                break
            take = min(remaining, max(int(size * (1.0 + p_stale * (self.oversize_cap - 1.0))), 1))
            alloc.append((q, px, take, p50, p_stale))
            remaining -= take
        if remaining > 0 and alloc:
            # 可见量不足：余量压到最优场所（MARKET）或放弃（IOC 由执行层下次再补）
            if s.order_type == "MARKET":
                q, px, take, p50, p_stale = alloc[0]
                alloc[0] = (q, px, take + remaining, p50, p_stale)
                remaining = 0
        slowest = max(a[3] for a in alloc) if self.sync_enabled else 0.0
        orders: List[VenueOrder] = []
        exp_fill = 0.0
        cost_num = 0.0
        for k, (q, px, take, p50, p_stale) in enumerate(alloc, start=1):
            delay = max(slowest - p50, 0.0) if self.sync_enabled else 0.0
            vo = VenueOrder(
                venue_order_id=f"{s.slice_id}-{k}",
                slice_id=s.slice_id,
                parent_id=s.parent_id,
                venue=q.venue,
                symbol=s.symbol,
                side=side,
                quantity=take,
                price=px if s.order_type == "IOC" else px * (1 + side * 0.002),  # MARKET 给 20bps 保护价
                order_type=s.order_type,
                send_delay_us=delay,
                expected_arrival_us=delay + p50,
                created_us=snap.ts_us,
                route_reason=f"eff_px={px + side * q.taker_fee:.4f} p_stale={p_stale:.2f}",
            )
            orders.append(vo)
            self._ref_mid[vo.venue_order_id] = mid
            self._sent_qty[q.venue] += take
            exp_fill += take * (1 - p_stale)
            cost_num += take * ((px - mid) * side + q.taker_fee)
        tot = sum(a[2] for a in alloc) or 1
        return RoutingDecision(
            slice_id=s.slice_id,
            venue_orders=orders,
            expected_cost_bps=1e4 * cost_num / tot / mid if mid > 0 else 0.0,
            expected_fill_ratio=min(exp_fill / s.quantity, 1.0),
            diagnostics={"n_venues": float(len(orders)), "sync_base_p50_us": slowest, "unfilled_visible": float(remaining)},
            rationale="按有效价（报价+费用+陈旧惩罚）分配可见量，并同步到达",
        )

    def _route_passive(self, s: OrderSlice, snap: MarketSnapshot, quotes) -> RoutingDecision:
        side = s.side
        mid = snap.mid
        bid, ask = snap.nbbo()
        limit = s.limit_price if s.limit_price is not None else (bid if side > 0 else ask)
        # POST_ONLY 不允许穿越对手价：若穿越则贴到本方最优价
        far = ask if side > 0 else bid
        if s.order_type == "POST_ONLY" and (limit - far) * side >= 0:
            limit = bid if side > 0 else ask
        # 选场所：返佣最优（maker_fee 最小），其次本方队列最短
        best = sorted(quotes, key=lambda q: (q.maker_fee, q.near_side(side)[1]))[0]
        vo = VenueOrder(
            venue_order_id=f"{s.slice_id}-1",
            slice_id=s.slice_id,
            parent_id=s.parent_id,
            venue=best.venue,
            symbol=s.symbol,
            side=side,
            quantity=s.quantity,
            price=limit,
            order_type=s.order_type if s.order_type in ("LIMIT", "POST_ONLY") else "LIMIT",
            send_delay_us=0.0,
            expected_arrival_us=self.latency.quantile(best.venue, 0.5),
            created_us=snap.ts_us,
            route_reason=f"maker_fee={best.maker_fee:.4f} queue={best.near_side(side)[1]}",
        )
        self._ref_mid[vo.venue_order_id] = mid
        self._sent_qty[best.venue] += s.quantity
        queue = best.near_side(side)[1]
        p_fill = 1.0 / (1.0 + queue / max(s.quantity, 1))  # 极粗的队列成交概率
        return RoutingDecision(
            slice_id=s.slice_id,
            venue_orders=[vo],
            expected_cost_bps=1e4 * ((limit - mid) * side + best.maker_fee) / mid if mid > 0 else 0.0,
            expected_fill_ratio=p_fill,
            diagnostics={"queue_ahead": float(queue), "n_venues": 1.0},
            rationale="挂单：选返佣最高且队列最短的场所",
        )

    # ------------------------------------------------------------------ 反馈
    def on_event(self, event: OrderEvent) -> None:
        if event.kind in ("fill", "partial") and event.fill is not None:
            f = event.fill
            self._fills[f.venue] += 1
            self._fill_qty[f.venue] += f.quantity
            if f.liquidity == "maker":
                self._maker_qty[f.venue] += f.quantity
            if f.latency_us > 0:
                self.latency.observe(f.venue, f.latency_us / 2.0)
            ref = self._ref_mid.get(f.venue_order_id)
            if ref:
                self._slip_num[f.venue] += f.quantity * 1e4 * f.side * (f.price - ref) / ref
        elif event.kind == "reject":
            self._rejects[event.venue] += 1

    def venue_stats(self) -> List[VenueStats]:
        out = []
        for v in self.venues:
            sent = self._sent_qty[v]
            filled = self._fill_qty[v]
            out.append(
                VenueStats(
                    venue=v,
                    fills=self._fills[v],
                    rejects=self._rejects[v],
                    fill_ratio=filled / sent if sent else 0.0,
                    avg_slippage_bps=self._slip_num[v] / filled if filled else 0.0,
                    latency_p50_us=self.latency.quantile(v, 0.5),
                    latency_p99_us=self.latency.quantile(v, 0.99),
                    maker_ratio=self._maker_qty[v] / filled if filled else 0.0,
                )
            )
        return out
