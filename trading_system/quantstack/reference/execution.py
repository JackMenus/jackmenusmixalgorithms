"""执行策略参考实现：TWAP / Almgren-Chriss（实施缺口最优轨迹）/ POV。

挂单策略库交付后由 ``order_placement.make_execution_factory`` 替换（它提供 twap/vwap/pov/almgren_chriss/
obizhaeva_wang/adaptive_passive/market_making/is_rl）。本文件的价值在于：
  * 演示 ``ExecutionStrategy`` 协议的事件驱动写法与簿记（在途量 / 已成交 / 释放量）；
  * 给 agent 的执行规划提供一个确定性的默认选择器 ``choose_strategy``。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..contracts import US_PER_SEC, CancelRequest, ExecutionOrder, Fill, MarketSnapshot, OrderEvent, OrderSlice


class _BaseStrategy:
    name = "base"

    def __init__(self, order: ExecutionOrder, **params):
        self.order = order
        self.params = params
        self.filled = 0
        self._n = 0
        self._slice_qty: Dict[str, int] = {}
        self._slice_filled: Dict[str, int] = {}
        self._slice_released: Dict[str, int] = {}
        self.started_us: int = order.start_us
        self.last_snapshot: Optional[MarketSnapshot] = None

    # ------------------------------------------------------------------ 簿记
    @property
    def outstanding(self) -> int:
        return sum(max(q - self._slice_filled.get(k, 0) - self._slice_released.get(k, 0), 0) for k, q in self._slice_qty.items())

    @property
    def unsent(self) -> int:
        return max(self.order.quantity - self.filled - self.outstanding, 0)

    def _make_slice(self, qty: int, order_type: str, limit_price: Optional[float], urgency: float, now_us: int, ttl_us: int = 0, tag: str = "") -> OrderSlice:
        self._n += 1
        sid = f"{self.order.order_id}-{self._n}"
        s = OrderSlice(
            slice_id=sid,
            parent_id=self.order.order_id,
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=int(qty),
            order_type=order_type,
            limit_price=None if limit_price is None else round(limit_price, 6),
            urgency=urgency,
            alpha_bps=self.order.alpha_bps,
            alpha_horizon_us=self.order.alpha_horizon_us or 1_000_000,
            ttl_us=ttl_us,
            created_us=now_us,
            tag=tag,
        )
        self._slice_qty[sid] = int(qty)
        return s

    def _locate_slice(self, venue_order_id: str) -> Optional[str]:
        if venue_order_id in self._slice_qty:
            return venue_order_id
        for sid in self._slice_qty:
            if venue_order_id.startswith(sid + "-"):
                return sid
        return None

    # ------------------------------------------------------------------ 协议
    def start(self, order: ExecutionOrder, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.started_us = snapshot.ts_us
        self.last_snapshot = snapshot
        return self.on_market(snapshot)

    def on_market(self, snapshot: MarketSnapshot) -> List[OrderSlice]:  # pragma: no cover - 子类实现
        raise NotImplementedError

    def on_fill(self, fill: Fill, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.filled += fill.quantity
        sid = self._locate_slice(fill.venue_order_id) or fill.slice_id
        if sid in self._slice_qty:
            self._slice_filled[sid] = self._slice_filled.get(sid, 0) + fill.quantity
        return []

    def on_event(self, event: OrderEvent, snapshot: MarketSnapshot) -> List[OrderSlice]:
        if event.kind in ("cancel", "reject", "expire"):
            sid = self._locate_slice(event.venue_order_id)
            if sid is not None:
                released = event.remaining_quantity or 0
                cap = self._slice_qty[sid] - self._slice_filled.get(sid, 0) - self._slice_released.get(sid, 0)
                self._slice_released[sid] = self._slice_released.get(sid, 0) + max(min(released, cap), 0)
        return []

    def cancels(self, snapshot: MarketSnapshot) -> List[CancelRequest]:
        return []

    def is_complete(self) -> bool:
        return self.filled >= self.order.quantity

    def state(self) -> Dict[str, float]:
        return {
            "filled": float(self.filled),
            "outstanding": float(self.outstanding),
            "unsent": float(self.unsent),
            "progress": self.filled / self.order.quantity if self.order.quantity else 1.0,
            "n_slices": float(self._n),
        }

    # ------------------------------------------------------------------ 公共工具
    def _time_frac(self, now_us: int) -> float:
        span = max(self.order.end_us - self.started_us, 1)
        return min(max((now_us - self.started_us) / span, 0.0), 1.0)

    def _slice_for_shortfall(self, shortfall: int, snapshot: MarketSnapshot, aggressive: bool, now_us: int, ttl_us: int, tag: str) -> List[OrderSlice]:
        if shortfall <= 0:
            return []
        min_slice = max(100, self.order.quantity // 50)
        if shortfall < min_slice and self.unsent > min_slice:
            return []   # 太小的切片不值得一次路由；等累积到最小切片
        shortfall = min(shortfall, self.unsent)
        if shortfall <= 0:
            return []
        side = self.order.side
        if aggressive:
            touch = snapshot.touch_price(side)
            cap = max(100, int(snapshot.displayed_size(side) * 1.2))
            qty = min(shortfall, cap)
            return [self._make_slice(qty, "IOC", touch, max(self.order.urgency, 0.7), now_us, 0, tag)]
        bid, ask = snapshot.nbbo()
        near = bid if side > 0 else ask
        return [self._make_slice(shortfall, "POST_ONLY", near, min(self.order.urgency, 0.4), now_us, ttl_us, tag)]


class TWAPStrategy(_BaseStrategy):
    """等时间切片。落后进度时用 IOC 追赶，否则在本方最优价被动挂单。"""

    name = "twap"

    def __init__(self, order: ExecutionOrder, n_buckets: Optional[int] = None, passive: Optional[bool] = None, **params):
        super().__init__(order, **params)
        span_s = max((order.end_us - order.start_us) / US_PER_SEC, 1.0)
        self.n_buckets = int(n_buckets or max(4, min(60, span_s // 30)))
        self.passive = (order.urgency < 0.6) if passive is None else passive

    def on_market(self, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.last_snapshot = snapshot
        now = snapshot.ts_us
        frac = self._time_frac(now)
        bucket = min(int(frac * self.n_buckets) + 1, self.n_buckets)
        target_cum = int(round(self.order.quantity * bucket / self.n_buckets))
        if frac >= 0.9:
            target_cum = self.order.quantity   # 收尾：最后 10% 时间全部转为主动完成
        shortfall = target_cum - self.filled - self.outstanding
        behind = (target_cum - self.filled) > self.order.quantity / self.n_buckets * 1.5
        aggressive = (not self.passive) or behind or frac >= 0.9
        bucket_us = max(int((self.order.end_us - self.started_us) / self.n_buckets), US_PER_SEC)
        return self._slice_for_shortfall(shortfall, snapshot, aggressive, now, ttl_us=bucket_us, tag="catchup" if behind else ("passive" if not aggressive else "sched"))


class AlmgrenChrissStrategy(_BaseStrategy):
    """Almgren-Chriss (2000) 最优执行轨迹：x(t) = X·sinh(κ(T−t))/sinh(κT)，κ = sqrt(λσ²/η)。

    σ 为每秒价格波动（$/√s），η 为临时冲击系数（$/股 per 股/秒）。``urgency`` 以 10^(2(u−0.5)) 放缩风险厌恶 λ，
    紧急度 1 即 λ×10（前置执行），紧急度 0 即 λ/10（接近 TWAP）。
    """

    name = "almgren_chriss"

    def __init__(self, order: ExecutionOrder, risk_aversion: float = 1e-5, eta: Optional[float] = None, **params):
        super().__init__(order, **params)
        self.lam = risk_aversion * 10 ** (2 * (order.urgency - 0.5))
        self.eta = eta
        self.kappa = 0.0
        self.T = max((order.end_us - order.start_us) / US_PER_SEC, 1.0)

    def start(self, order: ExecutionOrder, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.started_us = snapshot.ts_us
        self.T = max((order.end_us - self.started_us) / US_PER_SEC, 1.0)
        mid = snapshot.mid
        daily_vol = snapshot.volatility or 0.02
        sigma = mid * daily_vol / math.sqrt(23_400.0)
        adv = snapshot.adv or 1e6
        if self.eta is None:
            half_spread = max(mid * snapshot.spread_bps / 2e4, 0.005)
            v10 = 0.1 * adv / 23_400.0
            self.eta = half_spread / max(v10, 1e-6)
        self.kappa = math.sqrt(self.lam * sigma * sigma / self.eta) if self.eta > 0 else 0.0
        return self.on_market(snapshot)

    def remaining_target(self, t_s: float) -> float:
        X = self.order.quantity
        if self.kappa * self.T < 1e-6:
            return X * max(1 - t_s / self.T, 0.0)
        return X * math.sinh(self.kappa * max(self.T - t_s, 0.0)) / math.sinh(self.kappa * self.T)

    def on_market(self, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.last_snapshot = snapshot
        now = snapshot.ts_us
        t = (now - self.started_us) / US_PER_SEC
        step_ahead = min(t + 30.0, self.T)  # 向前看 30 秒的轨迹目标
        desired_cum = self.order.quantity - self.remaining_target(step_ahead)
        if t >= 0.9 * self.T:
            desired_cum = self.order.quantity
        shortfall = int(round(desired_cum)) - self.filled - self.outstanding
        behind = (desired_cum - self.filled) > 0.08 * self.order.quantity
        aggressive = behind or self.order.urgency >= 0.75 or t >= 0.9 * self.T
        return self._slice_for_shortfall(shortfall, snapshot, aggressive, now, ttl_us=20 * US_PER_SEC, tag="ac_catchup" if behind else "ac")

    def state(self) -> Dict[str, float]:
        st = super().state()
        st.update({"kappa": self.kappa, "lambda": self.lam, "eta": self.eta or 0.0})
        return st


class POVStrategy(_BaseStrategy):
    """成交量参与率（Percentage of Volume）：跟随市场成交量，以固定参与率吃单。"""

    name = "pov"

    def __init__(self, order: ExecutionOrder, rate: Optional[float] = None, **params):
        super().__init__(order, **params)
        self.rate = float(rate or min(max(order.max_participation, 0.01), 0.3))
        self._v0: Optional[float] = None

    def on_market(self, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.last_snapshot = snapshot
        now = snapshot.ts_us
        if self._v0 is None:
            self._v0 = snapshot.session_volume
        traded = max(snapshot.session_volume - self._v0, 0.0)
        target = min(self.rate / (1 - self.rate) * traded + 100, self.order.quantity)
        if self._time_frac(now) >= 0.9:
            target = self.order.quantity
        shortfall = int(round(target)) - self.filled - self.outstanding
        return self._slice_for_shortfall(shortfall, snapshot, True, now, ttl_us=0, tag="pov")

    def state(self) -> Dict[str, float]:
        st = super().state()
        st["rate"] = self.rate
        return st


class ReferenceExecutionFactory:
    _registry = {"twap": TWAPStrategy, "almgren_chriss": AlmgrenChrissStrategy, "pov": POVStrategy}

    def available(self) -> Dict[str, str]:
        return {
            "twap": "等时间切片；低紧急度时被动挂单、落后时 IOC 追赶。参数 n_buckets, passive",
            "almgren_chriss": "Almgren-Chriss 最优轨迹（风险厌恶 λ 由 urgency 放缩）。参数 risk_aversion, eta",
            "pov": "成交量参与率跟随。参数 rate",
        }

    def create(self, name: str, order: ExecutionOrder, **params):
        cls = self._registry.get(name)
        if cls is None:
            raise KeyError(f"未知执行策略 {name!r}，可用：{list(self._registry)}")
        return cls(order, **params)


def choose_strategy(order: ExecutionOrder, snapshot: Optional[MarketSnapshot], available: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    """确定性的默认执行策略选择（agent 缺席或作为 agent 的基线）。

    规则：母单相对 ADV 很大 -> 参与率/最优轨迹；alpha 半衰期短 / 紧急 -> 前置（POV 或高 λ 的 AC）；
    否则 TWAP 被动以赚取价差与返佣。
    """
    adv = snapshot.adv if snapshot and snapshot.adv else 1e6
    pct_adv = order.quantity / adv
    span_s = max((order.end_us - order.start_us) / US_PER_SEC, 1)
    horizon_s = (order.alpha_horizon_us or 0) / US_PER_SEC
    if order.urgency >= 0.75 and "pov" in available:
        return "pov", {"rate": float(min(max(order.max_participation, 0.05), 0.25))}
    if (pct_adv > 0.02 or (0 < horizon_s < 2 * span_s)) and "almgren_chriss" in available:
        return "almgren_chriss", {"risk_aversion": 1e-5}
    if "twap" in available:
        return "twap", {}
    name = next(iter(available))
    return name, {}
