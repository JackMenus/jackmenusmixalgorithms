"""延迟感知智能订单路由器（Latency-Aware SOR）。

把三块组合起来：
1. ``sweep_allocate``：吃单部分按「含费有效价格」排序扫单，按陈旧风险折算可成交量并适度超额下单（oversize）。
2. ``solve_cont_kukanov``：挂单部分按 Cont-Kukanov 在各场所之间拆分。
3. ``plan_synchronized_send``：对吃单子单做同步到达调度，降低被延迟套利者抢先撤单的概率。

urgency / alpha 决定吃单与挂单的比例；这两个量分别来自挂单策略库和因子挖掘库。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..interfaces import ChildOrder, ParentOrder, RoutingDecision, VenueQuote, VenueStats
from ..latency.model import VenueLatencyModel
from .cont_kukanov import default_outflow_samples, solve_cont_kukanov, urgency_to_penalty
from .stale_quote import expected_fill, survival_prob
from .sync_send import plan_synchronized_send


@dataclass
class RouterConfig:
    mc_samples: int = 4000
    reaction_window_us: float = 50.0     # 对手从看到一处成交到撤另一处报价的时间
    oversize_cap: float = 1.0            # >1 时探测隐藏流动性，但多场所叠加可能超成交，默认关闭
    max_venues: int = 6
    passive_turnover: float = 1.2        # 默认流出量 = 队列 × turnover
    lam_over_mult: float = 2.0           # 超成交惩罚相对 (h+f) 的倍数
    sync_enabled: bool = True
    seed: Optional[int] = 7


def sweep_allocate(
    side: int,
    qty: int,
    quotes: Sequence[VenueQuote],
    latency_samples: Dict[str, np.ndarray],
    limit_price: Optional[float] = None,
    oversize_cap: float = 1.0,
) -> List[tuple[VenueQuote, int, float]]:
    """吃单扫单分配。返回 [(quote, 下单量, 预期成交量)]。

    排序键：含费有效价格（买：ask+fee，卖：bid-fee），同价时优先存活概率高（更快/更稳定）的场所。
    预期可成交量按陈旧风险折算；某场所被全部占用时，可在不超过 oversize_cap×显示量、
    且不超过母单剩余量的前提下超额发送 IOC，用来捕获隐藏/刚刷新的流动性。
    """
    def eff_px(q: VenueQuote) -> float:
        px, _ = q.far_side(side)
        return px + side * q.taker_fee

    ranked = sorted(
        quotes,
        key=lambda q: (side * eff_px(q), -survival_prob(q.quote_update_rate, latency_samples[q.venue])),
    )
    remaining = float(qty)
    out = []
    for q in ranked:
        if remaining <= 0:
            break
        px, size = q.far_side(side)
        if size <= 0 or not np.isfinite(px):
            continue
        if limit_price is not None and side * (px - limit_price) > 0:
            continue
        exp_fill = expected_fill(size, q.quote_update_rate, latency_samples[q.venue])
        if exp_fill <= 0:
            continue
        take_exp = min(exp_fill, remaining)
        # 想拿走该场所预期流动性的比例 frac，就按同比例发送显示量；
        # 显示量之外允许按 oversize_cap 探测隐藏流动性，但总量不超过母单剩余量
        frac = take_exp / exp_fill
        send = int(np.ceil(frac * size))
        if frac >= 1.0 and oversize_cap > 1.0:
            send = int(min(size * oversize_cap, size + max(remaining - take_exp, 0.0)))
        out.append((q, send, take_exp))
        remaining -= take_exp
    return out


@dataclass
class _VenueAcc:
    sent: int = 0
    filled: int = 0
    maker_filled: int = 0
    fills: int = 0
    rejects: int = 0
    slip_qty_bps: float = 0.0     # Σ qty × slippage_bps


class LatencyAwareRouter:
    """实现 ``RouterProtocol``（route / on_event / venue_stats）。

    时间基准：``ChildOrder.send_delay_us`` 相对于 ``route()`` 返回、执行层开始派发的时刻 t0；
    同一次决策的所有子单共享 t0，执行层应在 t0 + send_delay_us 时发出。
    """

    def __init__(self, latency_model: VenueLatencyModel, config: Optional[RouterConfig] = None):
        self.lat = latency_model
        self.cfg = config or RouterConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self._acc: Dict[str, _VenueAcc] = {}
        self._ref_mid: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # 反馈闭环
    def on_event(self, event) -> None:
        """处理回报事件。fill/partial：用往返延迟的一半作为单程延迟喂入延迟模型，并更新场所统计。"""
        kind = getattr(event, "kind", "")
        fill = getattr(event, "fill", None)
        venue = getattr(event, "venue", "") or (getattr(fill, "venue", "") if fill is not None else "")
        if not venue:
            return
        acc = self._acc.setdefault(venue, _VenueAcc())
        if kind == "reject":
            acc.rejects += 1
        elif kind in ("fill", "partial") and fill is not None:
            qty = int(getattr(fill, "quantity", 0))
            acc.fills += 1
            acc.filled += qty
            if getattr(fill, "liquidity", "taker") == "maker":
                acc.maker_filled += qty
            rtt = float(getattr(fill, "latency_us", 0.0) or 0.0)
            if rtt > 0:
                self.lat.observe(venue, rtt / 2.0)
            mid = self._ref_mid.get(getattr(fill, "symbol", ""))
            side = int(getattr(fill, "side", 0))
            if mid and side:
                acc.slip_qty_bps += qty * side * (float(fill.price) - mid) / mid * 1e4

    def venue_stats(self) -> List[VenueStats]:
        out = []
        for v in sorted(set(self._acc) | set(self.lat.venues())):
            a = self._acc.get(v, _VenueAcc())
            has_lat = v in self.lat.venues()
            out.append(VenueStats(
                venue=v,
                fills=a.fills,
                rejects=a.rejects,
                fill_ratio=a.filled / a.sent if a.sent else 0.0,
                avg_slippage_bps=a.slip_qty_bps / a.filled if a.filled else 0.0,
                latency_p50_us=self.lat.quantile(v, 0.5) if has_lat else 0.0,
                latency_p99_us=self.lat.quantile(v, 0.99) if has_lat else 0.0,
                maker_ratio=a.maker_filled / a.filled if a.filled else 0.0,
            ))
        return out

    def _record_sent(self, symbol: str, mid: float, children: List[ChildOrder]) -> None:
        self._ref_mid[symbol] = mid
        for c in children:
            self._acc.setdefault(c.venue, _VenueAcc()).sent += c.quantity

    def route(self, order: ParentOrder, quotes: Sequence[VenueQuote]) -> RoutingDecision:
        cfg = self.cfg
        side = order.side
        quotes = [q for q in quotes if q.bid > 0 and q.ask > 0 and q.ask >= q.bid][: cfg.max_venues]
        if not quotes:
            return RoutingDecision([], 0.0, 0.0, {"reason_no_quotes": 1.0})

        nbb = max(q.bid for q in quotes)
        nbo = min(q.ask for q in quotes)
        mid = 0.5 * (nbb + nbo)
        half_spread = max(0.5 * (nbo - nbb), 1e-9)
        lat_s = {q.venue: self.lat.sample(q.venue, cfg.mc_samples, self.rng) for q in quotes}

        # --- urgency：挂单策略给出的 urgency 与因子 alpha 叠加 ---
        cross_cost_bps = (half_spread + np.mean([q.taker_fee for q in quotes])) / mid * 1e4
        alpha_boost = 0.0
        if order.alpha_bps > 0 and cross_cost_bps > 0:
            alpha_boost = np.clip(order.alpha_bps / cross_cost_bps - 1.0, 0.0, 1.0) * 0.5
        urgency = float(np.clip(order.urgency + alpha_boost, 0.0, 1.0))

        aggressive_qty = int(round(order.quantity * urgency))
        passive_qty = order.quantity - aggressive_qty
        children: List[ChildOrder] = []
        exp_cost_px = 0.0
        exp_filled = 0.0
        diag: Dict[str, float] = {"urgency_effective": urgency, "cross_cost_bps": cross_cost_bps}

        # --- 1. 吃单 ---
        if aggressive_qty > 0:
            alloc = sweep_allocate(side, aggressive_qty, quotes, lat_s, order.limit_price, cfg.oversize_cap)
            venues = [q.venue for q, _, _ in alloc]
            delays = {v: 0.0 for v in venues}
            if cfg.sync_enabled and len(venues) > 1:
                plan = plan_synchronized_send(venues, {v: lat_s[v] for v in venues}, cfg.reaction_window_us)
                delays = plan.send_delay_us
                diag.update(sync_success_prob=plan.success_prob, sync_naive_prob=plan.naive_success_prob, sync_quantile=plan.quantile_used)
            for q, send, take_exp in alloc:
                px, _ = q.far_side(side)
                children.append(ChildOrder(
                    venue=q.venue, side=side, quantity=int(send), price=px, order_type="IOC",
                    send_delay_us=delays.get(q.venue, 0.0),
                    expected_arrival_us=delays.get(q.venue, 0.0) + float(np.median(lat_s[q.venue])),
                    parent_id=order.order_id,
                ))
                exp_cost_px += take_exp * (side * (px - mid) + q.taker_fee)
                exp_filled += take_exp
            # 吃不到的部分转入挂单
            passive_qty += int(max(aggressive_qty - exp_filled, 0))

        # --- 2. 挂单（Cont-Kukanov）---
        if passive_qty > 0:
            Qs = np.array([q.near_side(side)[1] for q in quotes], dtype=float)
            # 只有处在 NBBO 上的场所挂单才有意义；不在最优价的场所排队视为无穷大
            best_near = nbb if side > 0 else nbo
            on_top = np.array([abs(q.near_side(side)[0] - best_near) < 1e-12 for q in quotes])
            Qs_eff = np.where(on_top, Qs, Qs + 1e12)
            xi = default_outflow_samples(np.where(on_top, Qs, 0.0) + 1.0, np.full(len(quotes), cfg.passive_turnover), n=cfg.mc_samples // 2, rng=self.rng)
            fee = float(np.mean([q.taker_fee for q in quotes]))
            rebates = np.array([-q.maker_fee for q in quotes])
            alpha_px = order.alpha_bps * 1e-4 * mid
            lam_u = urgency_to_penalty(urgency, half_spread, fee, alpha_px)
            ck = solve_cont_kukanov(
                passive_qty, Qs_eff, xi, half_spread, fee, rebates, lam_u, cfg.lam_over_mult * (half_spread + fee),
            )
            diag.update(ck_market_qty=ck.market_qty, ck_shortfall_prob=ck.shortfall_prob,
                        passive_target_qty=float(passive_qty), passive_posted_qty=float(np.round(ck.limit_qty).sum()))
            # 注意：CK 会在成交不确定时「超额挂单」（posted > target），这是最优解的一部分，
            # 下游挂单/执行层必须在母单完成时撤销剩余挂单（cancel-on-complete）。
            for q, L in zip(quotes, ck.limit_qty):
                n = int(round(L))
                if n <= 0:
                    continue
                px, _ = q.near_side(side)
                children.append(ChildOrder(venue=q.venue, side=side, quantity=n, price=px, order_type="POST_ONLY",
                                           expected_arrival_us=float(np.median(lat_s[q.venue])), parent_id=order.order_id))
            m = int(round(ck.market_qty))
            if m > 0:
                best = min(quotes, key=lambda q: side * (q.far_side(side)[0] + side * q.taker_fee))
                children.append(ChildOrder(venue=best.venue, side=side, quantity=m, price=best.far_side(side)[0], order_type="IOC",
                                           expected_arrival_us=float(np.median(lat_s[best.venue])), parent_id=order.order_id))
            # CK 的期望成本按「相对 mid」：挂单成交为 -h - r，市价为 h + f
            exp_cost_px += ck.expected_cost * passive_qty
            exp_filled += ck.expected_fill

        self._record_sent(order.symbol, mid, children)
        total = max(order.quantity, 1)
        return RoutingDecision(
            children=children,
            expected_cost_bps=float(exp_cost_px / total / mid * 1e4),
            expected_fill_ratio=float(min(exp_filled / total, 1.0)),
            diagnostics=diag,
        )


def make_router(venues: Sequence[str] = (), latency_prior_us: Optional[Dict[str, float]] = None,
                default_latency_us: float = 500.0, halflife: float = 200.0, **cfg) -> LatencyAwareRouter:
    """工厂函数。``venues`` 仅用于预置延迟先验；``cfg`` 透传给 ``RouterConfig``。

    ``latency_prior_us``：{venue: 单程延迟}，在没有实测样本前作为该场所的伪样本，
    之后会被 ``on_event`` 喂入的实测延迟逐步覆盖。
    """
    lat = VenueLatencyModel(halflife=halflife, default_us=default_latency_us)
    prior = latency_prior_us or {}
    for v in venues:
        if v in prior:
            lat.observe_many(v, [float(prior[v])] * 5)
    return LatencyAwareRouter(lat, RouterConfig(**cfg))
