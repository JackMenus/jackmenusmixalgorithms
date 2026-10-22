"""quantstack 集成层适配器：把本库算法包装为 ``quantstack.contracts.ExecutionStrategy``。

使用方式（集成层）::

    import sys; sys.path += ["/mnt/project-files/trading_system", "/mnt/project-files/quant/order_placement"]
    from order_placement import make_execution_factory
    factory = make_execution_factory(trading_seconds_per_day=23400)
    strat = factory.create("almgren_chriss", exec_order)
    slices = strat.start(exec_order, snapshot)

统一执行机制（所有策略共享，差异只在“计划累计成交曲线”）：
1. 每个策略给出 t 时刻的计划累计成交量 target(t)；
2. 若落后 target 超过 ``catch_up_slices`` 个切片 → 发 IOC 吃单追赶（urgency=1）；
3. 否则在 NBBO 近端挂一张被动单（POST_ONLY，ttl=refresh_us），到期由路由层撤单或策略主动撤；
4. 到达 end_us 仍有剩余 → 一次性 IOC 扫尾；
5. 母单 limit_price 作为所有子单的价格保护。

约定：
- ``alpha_bps`` 按集成层约定“以 side 方向为正”：>0 表示价格预计朝 side 方向移动
  （买单看涨 / 卖单看跌），对执行而言是“等待更贵”，因此会加速执行。
- 撤单：``cancels`` 返回的 ``CancelRequest.venue_order_id`` 填 **slice_id**，
  ``reason`` 以 ``"slice:"`` 开头，路由层需把它展开为该切片下的全部场所订单。
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from quantstack.contracts import (
    CancelRequest,
    ExecutionOrder,
    Fill,
    MarketSnapshot,
    OrderEvent,
    OrderSlice,
)

from ..execution.schedules import ACParams, almgren_chriss, obizhaeva_wang, twap, vwap
from ..market_making.models import glft_coefficients
from ..queue.models import queue_value

US = 1_000_000
_slice_ids = itertools.count(1)

DEFAULTS: Dict[str, float] = {
    "trading_seconds_per_day": 23400.0,  # A 股/美股 6.5h；7×24 加密市场设 86400
    "bucket_s": 60.0,                    # 计划曲线的区间长度
    "refresh_s": 5.0,                    # 被动单 ttl / 刷新周期
    "catch_up_slices": 1.0,              # 落后多少个切片后转主动
    "impact_frac": 0.1,                  # 以 ADV 速率交易时临时冲击 ≈ impact_frac·日波动
    "alpha_urgency_bps": 2.0,            # |alpha| 超过该值才影响节奏
    "min_slice": 1.0,
    "min_passive_frac": 0.5,             # 被动单最小为半个切片
    "queue_depletion_frac": 0.5,         # 队列消耗速率 ≈ 该比例 × 平均每秒成交量
}


def _f(params: Dict, key: str) -> float:
    return float(params.get(key, DEFAULTS[key]))


class QuantstackExecution:
    """ExecutionStrategy 协议的通用实现；``plan`` 决定计划累计成交曲线。"""

    name = "base"

    def __init__(self, order: ExecutionOrder, params: Optional[Dict] = None):
        self.order = order
        self.p = {**DEFAULTS, **(order.params or {}), **(params or {})}
        self.filled = 0
        self.notional = 0.0
        self.working: Dict[str, Tuple[int, float, str, int]] = {}  # slice_id -> (剩余量, 价格, 类型, 创建时间)
        self._to_cancel: List[str] = []
        self._cancelling: set = set()  # 已发撤单、等待回报的切片：仍计入在途量，避免重复下单
        self.arrival = 0.0
        self.start_volume = 0.0
        self.cum_target: Optional[np.ndarray] = None
        self.done_flag = False
        self.last_ts = order.start_us

    # ---------------- 计划曲线 ----------------
    @property
    def duration_s(self) -> float:
        return max((self.order.end_us - self.order.start_us) / US, 1e-6)

    def n_buckets(self) -> int:
        return int(max(1, min(500, round(self.duration_s / _f(self.p, "bucket_s")))))

    def plan(self, snapshot: MarketSnapshot) -> np.ndarray:
        """返回每个区间的计划量（总和 = quantity）。子类覆盖。"""
        return twap(self.order.quantity, self.n_buckets())

    def target(self, snapshot: MarketSnapshot) -> float:
        n = len(self.cum_target) - 1
        u = (snapshot.ts_us - self.order.start_us) / US / self.duration_s * n
        u = float(np.clip(u, 0, n))
        i = int(min(math.floor(u), n - 1))
        return float(self.cum_target[i] + (u - i) * (self.cum_target[i + 1] - self.cum_target[i]))

    def slice_size(self) -> float:
        return max(self.order.quantity / (len(self.cum_target) - 1), _f(self.p, "min_slice"))

    # ---------------- 市场参数估计 ----------------
    def sigma_price_per_sqrt_s(self, snap: MarketSnapshot) -> float:
        vol = snap.volatility if snap.volatility else 0.02
        return vol * snap.mid / math.sqrt(_f(self.p, "trading_seconds_per_day"))

    def volume_rate(self, snap: MarketSnapshot) -> float:
        adv = snap.adv if snap.adv else 100.0 * self.order.quantity
        return adv / _f(self.p, "trading_seconds_per_day")

    def eta(self, snap: MarketSnapshot) -> float:
        vol = snap.volatility if snap.volatility else 0.02
        return _f(self.p, "impact_frac") * vol * snap.mid / max(self.volume_rate(snap), 1e-9)

    def drift(self, snap: MarketSnapshot) -> float:
        """alpha → 价格漂移（价格单位/秒，按 side 方向为正）。"""
        o = self.order
        if not o.alpha_horizon_us or abs(o.alpha_bps) < _f(self.p, "alpha_urgency_bps"):
            return 0.0
        return snap.mid * o.alpha_bps * 1e-4 / (o.alpha_horizon_us / US)

    # ---------------- 协议方法 ----------------
    def start(self, order: ExecutionOrder, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.order = order
        self.arrival = order.arrival_price or snapshot.mid
        self.start_volume = snapshot.session_volume
        n = np.asarray(self.plan(snapshot), float)
        n = n * order.quantity / max(n.sum(), 1e-12)
        self.cum_target = np.concatenate([[0.0], np.cumsum(n)])
        return self.on_market(snapshot)

    def remaining(self) -> int:
        return max(self.order.quantity - self.filled, 0)

    def inflight(self, kinds=("IOC", "MARKET", "LIMIT", "POST_ONLY", "PEG")) -> int:
        return sum(q for q, _, t, _ in self.working.values() if t in kinds)

    def _mk(self, snap, qty, otype, price, urgency, tag, ttl_us=0) -> OrderSlice:
        o = self.order
        sid = f"{o.order_id}-s{next(_slice_ids)}"
        if o.limit_price is not None and price is not None:
            price = min(price, o.limit_price) if o.side > 0 else max(price, o.limit_price)
        s = OrderSlice(slice_id=sid, parent_id=o.order_id, symbol=o.symbol, side=o.side, quantity=int(qty),
                       order_type=otype, limit_price=price, urgency=urgency, alpha_bps=o.alpha_bps,
                       alpha_horizon_us=o.alpha_horizon_us or US, ttl_us=int(ttl_us), created_us=snap.ts_us,
                       tag=tag)
        self.working[sid] = (int(qty), price if price is not None else float("nan"), otype, snap.ts_us)
        return s

    def _cancel_passive(self) -> None:
        for sid, (_, _, t, _) in list(self.working.items()):
            if t in ("LIMIT", "POST_ONLY", "PEG") and sid not in self._to_cancel and sid not in self._cancelling:
                self._to_cancel.append(sid)

    def _drop(self, sid: str) -> None:
        self.working.pop(sid, None)
        self._cancelling.discard(sid)
        if sid in self._to_cancel:
            self._to_cancel.remove(sid)

    def _expire_stale(self, snap: MarketSnapshot) -> None:
        """路由层若未回 expire 事件，按 ttl 自行视为失效（避免重复计入在途量）。"""
        ttl = _f(self.p, "refresh_s") * US
        for sid, (q, px, t, created) in list(self.working.items()):
            if t in ("IOC", "MARKET") and snap.ts_us - created > 2 * ttl:
                self._drop(sid)
            elif t in ("LIMIT", "POST_ONLY", "PEG") and snap.ts_us - created > ttl * 3:
                self._drop(sid)

    def urgency_target(self, snap: MarketSnapshot) -> float:
        tgt = self.target(snap)
        if self.order.alpha_bps > _f(self.p, "alpha_urgency_bps"):  # 等待更贵 → 提前一个切片
            tgt += self.slice_size()
        return min(tgt, self.order.quantity)

    def on_market(self, snapshot: MarketSnapshot) -> List[OrderSlice]:
        self.last_ts = snapshot.ts_us
        rem = self.remaining()
        if rem <= 0 or self.done_flag:
            self._cancel_passive()
            return []
        self._expire_stale(snapshot)
        o, snap = self.order, snapshot
        bid, ask = snap.nbbo()
        far = ask if o.side > 0 else bid
        near = bid if o.side > 0 else ask
        out: List[OrderSlice] = []
        if snap.ts_us >= o.end_us:  # 扫尾
            self._cancel_passive()
            q = rem - self.inflight(("IOC", "MARKET"))
            if q > 0:
                out.append(self._mk(snap, q, "IOC", far, 1.0, "final"))
            return out
        behind = self.urgency_target(snap) - self.filled - self.inflight(("IOC", "MARKET"))
        sl = self.slice_size()
        if behind > _f(self.p, "catch_up_slices") * sl:
            q = int(min(behind, rem - self.inflight(("IOC", "MARKET"))))
            disp = snap.displayed_size(o.side)
            if disp > 0:
                q = min(q, disp)
            if q > 0:
                self._cancel_passive()
                out.append(self._mk(snap, q, "IOC", far, 1.0, "catchup"))
            return out
        return out + self.passive(snap, near, sl, rem)

    def passive_price(self, snap: MarketSnapshot, near: float, sl: float) -> float:
        """默认挂近端；价差宽且排队不划算时改进一跳。"""
        o = self.order
        tick = snap.extra.get("tick_size", 0.01)
        q = next(iter(sorted(snap.quotes, key=lambda x: -x.bid if o.side > 0 else x.ask)), None)
        q_ahead = (q.bid_size if o.side > 0 else q.ask_size) if q else 0
        bid, ask = snap.nbbo()
        dec = queue_value(ask - bid, q_ahead, _f(self.p, "queue_depletion_frac") * self.volume_rate(snap),
                          adverse_rate=0.2, adverse_selection=0.0)
        if not dec.use_limit and (ask - bid) > 1.5 * tick:
            return near + o.side * tick
        return near

    def passive(self, snap, near, sl, rem) -> List[OrderSlice]:
        px = self.passive_price(snap, near, sl)
        live = [(sid, v) for sid, v in self.working.items()
                if v[2] in ("LIMIT", "POST_ONLY", "PEG") and sid not in self._cancelling and sid not in self._to_cancel]
        if live and all(abs(v[1] - px) < 1e-9 for _, v in live):
            return []
        if live:
            self._cancel_passive()  # 价格变化：先撤后挂，旧单在撤单回报前仍计入在途
        # 被动单只允许超前计划一个切片：额度 = 计划(含 alpha 提前) + 1 切片 − 已成交 − 全部在途
        allow = min(self.urgency_target(snap) + sl, self.order.quantity) - self.filled - self.inflight()
        q = int(min(math.ceil(sl), allow, rem))
        # 额度不足半个切片时先等待，避免大量碎单（碎单会增加撤改量与逐笔冲击）
        if q < min(max(_f(self.p, "min_slice"), _f(self.p, "min_passive_frac") * sl), rem):
            return []
        return [self._mk(snap, q, "POST_ONLY", px, 0.2, "passive", _f(self.p, "refresh_s") * US)]

    def on_fill(self, fill: Fill, snapshot: MarketSnapshot) -> List[OrderSlice]:
        if fill.parent_id != self.order.order_id:
            return []
        self.filled += fill.quantity
        self.notional += fill.quantity * fill.price
        if fill.slice_id in self.working:
            q, px, t, c = self.working[fill.slice_id]
            q -= fill.quantity
            if q <= 0:
                self._drop(fill.slice_id)
            else:
                self.working[fill.slice_id] = (q, px, t, c)
        if self.remaining() <= 0:
            self._cancel_passive()
        return []

    def on_event(self, event: OrderEvent, snapshot: MarketSnapshot) -> List[OrderSlice]:
        """reject/cancel/expire：从在途表移除；IOC 未成交部分在下次 on_market 自动补。

        OrderEvent 只带 venue_order_id；集成层约定其中包含 slice_id 前缀（slice_id 或 slice_id-xxx）。
        """
        if event.kind in ("reject", "cancel", "expire"):
            for sid in list(self.working):
                if event.venue_order_id == sid or event.venue_order_id.startswith(sid + "-") or \
                        event.reason.endswith(sid):
                    q, px, t, c = self.working[sid]
                    if 0 < event.remaining_quantity < q:
                        # 部分释放（未路由余量 unrouted、OMS 缩量 oms:in_flight_cap_resize、
                        # 或多场所切片中某一场所单撤销/过期）：只扣减 remaining_quantity，其余仍在途
                        self.working[sid] = (q - event.remaining_quantity, px, t, c)
                    else:
                        self._drop(sid)
            if event.kind == "reject" and "limit" in event.reason.lower():
                self.done_flag = True
        return []

    def cancels(self, snapshot: MarketSnapshot) -> List[CancelRequest]:
        out = [CancelRequest(venue_order_id=sid, ts_us=snapshot.ts_us, reason=f"slice:{sid}")
               for sid in self._to_cancel]
        self._cancelling.update(self._to_cancel)
        self._to_cancel = []
        return out

    def is_complete(self) -> bool:
        return self.remaining() <= 0 or self.done_flag

    def state(self) -> Dict[str, float]:
        o = self.order
        elapsed = (self.last_ts - o.start_us) / max(o.end_us - o.start_us, 1)
        avg = self.notional / self.filled if self.filled else 0.0
        is_bps = o.side * (avg - self.arrival) / self.arrival * 1e4 if self.filled and self.arrival else 0.0
        return {
            "filled": float(self.filled),
            "remaining": float(self.remaining()),
            "progress": self.filled / max(o.quantity, 1),
            "time_elapsed": float(np.clip(elapsed, 0, 1)),
            "schedule_target": float(self.target_at(self.last_ts) / max(o.quantity, 1)) if self.cum_target is not None else 0.0,
            "inflight": float(self.inflight()),
            "avg_price": avg,
            "shortfall_bps": is_bps,
        }

    def target_at(self, ts_us: int) -> float:
        return self.target(MarketSnapshot(symbol=self.order.symbol, ts_us=ts_us))


# ---------------------------------------------------------------- 具体策略
class TWAPExec(QuantstackExecution):
    name = "twap"


class VWAPExec(QuantstackExecution):
    """params['volume_profile'] 可传入区间成交量曲线（list）；缺省为 U 型日内曲线。"""

    name = "vwap"

    def plan(self, snap):
        prof = self.p.get("volume_profile")
        n = self.n_buckets()
        if prof is None:
            x = np.linspace(-1, 1, n)
            prof = 1.0 + 1.5 * x**2
        return vwap(self.order.quantity, np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(prof)), prof))


class POVExec(QuantstackExecution):
    """按实时市场成交量的 max_participation 比例跟随（target 动态，不预先固定）。"""

    name = "pov"

    def target(self, snap):
        traded = max(snap.session_volume - self.start_volume, 0.0) if snap.session_volume else None
        if traded is None or traded == 0:  # 无实时成交量时用 ADV 速率估算
            traded = self.volume_rate(snap) * max((snap.ts_us - self.order.start_us) / US, 0)
        return min(self.order.max_participation * traded, self.order.quantity)


class AlmgrenChrissExec(QuantstackExecution):
    """urgency∈[0,1] 映射为 κT∈[0,8]（0≈TWAP，越大越前置），可用 params['risk_aversion'] 直接指定 λ。"""

    name = "almgren_chriss"

    def plan(self, snap):
        n = self.n_buckets()
        T = self.duration_s
        sigma, eta = self.sigma_price_per_sqrt_s(snap), self.eta(snap)
        if "risk_aversion" in self.p:
            lam = float(self.p["risk_aversion"])
        else:
            kappa = 8.0 * self.order.urgency / T
            lam = kappa**2 * eta / max(sigma**2, 1e-18)
        # 漂移按 side 方向为正 → 对“卖出”而言漂移为 −side·μ… 统一转为 is_sell=False 的买入视角
        mu = self.drift(snap)  # >0：价格朝不利方向走
        return almgren_chriss(self.order.quantity, T, n, ACParams(sigma, eta, 0.0, 0.0, lam, drift=mu),
                              is_sell=False)

    def urgency_target(self, snap):  # 漂移已体现在轨迹中，不再额外提前
        return self.target(snap)


class ObizhaevaWangExec(QuantstackExecution):
    name = "obizhaeva_wang"

    def plan(self, snap):
        return obizhaeva_wang(self.order.quantity, self.duration_s, self.n_buckets(),
                              rho=float(self.p.get("resilience_per_s", 1.0 / 60)))


class AdaptivePassiveExec(QuantstackExecution):
    """尽量被动：落后 3 个切片才追单，排队不划算时才改进一跳；适合低紧急度、宽价差标的。"""

    name = "adaptive_passive"

    def __init__(self, order, params=None):
        super().__init__(order, {"catch_up_slices": 3.0, **(params or {})})


class MarketMakingExec(QuantstackExecution):
    """GLFT 库存驱动的单边被动挂单：把剩余量视为待平库存 q，
    挂单距离 δ = c1 − (q−½)·c2（库存越大越靠近/越激进），alpha 再平移 μ/(γσ²)·c2。"""

    name = "market_making"

    def passive_price(self, snap, near, sl):
        o = self.order
        sigma = self.sigma_price_per_sqrt_s(snap)
        k = float(self.p.get("k", 1.0 / max(snap.mid * 1e-4, 1e-9)))  # 默认 1bp 衰减尺度
        A = float(self.p.get("A", 1.0))
        gamma = float(self.p.get("gamma", 0.1 + o.urgency))
        lots = max(self.remaining() / max(sl, 1), 0.0)
        c1, c2 = glft_coefficients(gamma, max(sigma, 1e-12), A, k, 1.0)
        mu = self.drift(snap)
        delta = c1 - (min(lots, 10) - 0.5) * c2 - (mu / (gamma * sigma**2) * c2 if sigma > 0 else 0.0)
        bid, ask = snap.nbbo()
        mid = snap.mid
        px = mid - o.side * delta
        tick = snap.extra.get("tick_size", 0.01)
        px = math.floor(px / tick) * tick if o.side > 0 else math.ceil(px / tick) * tick
        # 不穿越对手价，不劣于近端两跳以外
        if o.side > 0:
            return float(min(max(px, bid - 2 * tick), ask - tick))
        return float(max(min(px, ask + 2 * tick), bid + tick))


class ISRLExec(QuantstackExecution):
    """表格 Q-learning（Nevmyvaka 等 2006）：每个区间按策略选择 TWAP 量的倍数。

    同一组 (区间数, 漂移档) 的智能体在进程内缓存，首次创建约 1–2 秒训练。
    """

    name = "is_rl"
    _cache: Dict[Tuple, object] = {}

    def __init__(self, order, params=None):
        super().__init__(order, params)
        self.bucket_idx = -1
        self.plan_n: Optional[np.ndarray] = None

    def _agent(self, n: int):
        from ..rl.execution_env import ExecEnvConfig, ExecutionEnv, QLearningExecutor
        key = (n,)
        if key not in self._cache:
            cfg = ExecEnvConfig(N=n, drift_vol=0.05, sigma=0.02)
            self._cache[key] = (cfg, QLearningExecutor(cfg).train(ExecutionEnv(cfg, seed=0),
                                                                   episodes=int(self.p.get("rl_episodes", 20000))))
        return self._cache[key]

    def n_buckets(self):
        return int(min(super().n_buckets(), 20))

    def plan(self, snap):
        self.plan_n = twap(self.order.quantity, self.n_buckets())
        return self.plan_n

    def on_market(self, snapshot):
        n = len(self.cum_target) - 1
        i = int(np.clip((snapshot.ts_us - self.order.start_us) / US / self.duration_s * n, 0, n - 1))
        if i != self.bucket_idx:
            from ..rl.execution_env import ACTIONS
            self.bucket_idx = i
            cfg, agent = self._agent(n)
            inv_frac = self.remaining() / max(self.order.quantity, 1)
            ib = min(int(inv_frac * cfg.inv_bins), cfg.inv_bins - 1) if inv_frac > 0 else 0
            # 环境为卖出视角：漂移档 2 表示“继续持有有利”，即对本单而言价格朝有利方向
            a_side = self.order.alpha_bps
            db = 1 if abs(a_side) < _f(self.p, "alpha_urgency_bps") else (0 if a_side > 0 else 2)
            mult = float(ACTIONS[agent.act((i, ib, db), greedy=True)]) if i < n - 1 else None
            base = self.order.quantity / n
            done_before = self.filled
            want = self.remaining() if mult is None else min(base * mult, self.remaining())
            self.cum_target[i + 1:] = np.minimum(done_before + want + base * np.arange(n - i), self.order.quantity)
            self.cum_target[i] = min(self.cum_target[i], done_before)
        return super().on_market(snapshot)

    def urgency_target(self, snap):
        return self.target(snap)


STRATEGIES: Dict[str, Tuple[type, str]] = {
    "twap": (TWAPExec, "时间均匀切分；被动挂单 + 落后追单。适合无明显 alpha、流动性均匀的标的。"),
    "vwap": (VWAPExec, "按日内成交量曲线切分（params.volume_profile，缺省 U 型）。基准为 VWAP 时首选。"),
    "pov": (POVExec, "按实时成交量的 max_participation 比例跟随，隐蔽性好、完成时间不确定。"),
    "almgren_chriss": (AlmgrenChrissExec, "冲击-风险均值方差最优轨迹；urgency 越高越前置；alpha_bps 进入漂移项。到达价基准首选。"),
    "obizhaeva_wang": (ObizhaevaWangExec, "订单簿指数恢复模型：首尾大单 + 中间匀速；params.resilience_per_s 为恢复速度。"),
    "adaptive_passive": (AdaptivePassiveExec, "尽量挂单赚价差，落后 3 个切片才追单；低紧急度、宽价差标的。"),
    "market_making": (MarketMakingExec, "GLFT 库存驱动的单边被动报价，剩余量越大越激进；params: gamma, A, k。"),
    "is_rl": (ISRLExec, "表格 Q-learning 每区间选择 TWAP 倍数，利用 alpha 漂移调节节奏（首次创建需训练 1–2 秒）。"),
}


class ExecutionFactory:
    def __init__(self, **cfg):
        self.cfg = cfg

    def create(self, name: str, order: ExecutionOrder, **params) -> QuantstackExecution:
        key = (name or order.strategy_hint or "twap").lower()
        if key not in STRATEGIES:
            raise KeyError(f"未知执行策略 {name!r}，可选：{sorted(STRATEGIES)}")
        return STRATEGIES[key][0](order, {**self.cfg, **params})

    def available(self) -> Dict[str, str]:
        return {k: v[1] for k, v in STRATEGIES.items()}


class SqrtLawImpact:
    """ImpactModel：E[成本 bps] = ½价差 + Y·σ_日·√(Q/ADV)（平方根冲击律）。

    执行时长越短，额外乘 √(参考时长/实际时长) 的加急惩罚（参考时长默认 30 分钟，上限 3 倍）。
    """

    def __init__(self, y: float = 0.7, ref_duration_s: float = 1800.0):
        self.y, self.ref = y, ref_duration_s

    def expected_cost_bps(self, quantity: int, snapshot: MarketSnapshot, duration_us: int) -> float:
        vol = snapshot.volatility or 0.02
        adv = snapshot.adv or max(quantity * 100.0, 1.0)
        rush = min(math.sqrt(self.ref / max(duration_us / US, 1.0)), 3.0) if duration_us > 0 else 3.0
        return float(0.5 * snapshot.spread_bps + self.y * vol * math.sqrt(quantity / adv) * 1e4 * max(rush, 1.0))


def make_execution_factory(**cfg) -> ExecutionFactory:
    return ExecutionFactory(**cfg)
