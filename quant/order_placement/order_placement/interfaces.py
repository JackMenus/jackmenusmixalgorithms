"""统一数据结构与协议（跨库集成契约）。

本模块是挂单策略库对外的唯一“契约层”：
- 因子挖掘库 → 本库：通过 ``AlphaSignal`` 提供短期预期收益；
- 本库 → 路由延迟优化库：通过 ``ChildOrder`` 输出待发送子单；
- 路由延迟优化库 → 本库：通过 ``LatencyProfile`` 回传链路延迟画像，
  通过 ``Fill`` 回传成交回报。

所有字段均为基础类型，便于序列化（``to_dict``）与跨进程传输。
价格单位：报价货币；数量单位：股/张/币（由调用方保证一致）；
时间单位：秒（float，Unix 时间戳或仿真时钟均可）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable
import itertools

_id_counter = itertools.count(1)


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    IOC = "IOC"
    POST_ONLY = "POST_ONLY"
    CANCEL = "CANCEL"


@dataclass(frozen=True)
class BookSnapshot:
    """L2 订单簿快照。bids 按价格降序，asks 按价格升序，元素为 (price, qty)。"""

    symbol: str
    ts: float
    bids: Tuple[Tuple[float, float], ...]
    asks: Tuple[Tuple[float, float], ...]
    tick_size: float = 0.01

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_ticks(self) -> float:
        return self.spread / self.tick_size

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlphaSignal:
    """来自因子挖掘库的短期 alpha。

    value: 在 horizon_s 秒内的预期收益，单位为 **基点 (bps)**，正数看涨。
    confidence: [0, 1]，用于缩放信号强度。
    """

    symbol: str
    ts: float
    value: float
    horizon_s: float
    confidence: float = 1.0
    source: str = "factor_mining"

    def drift_per_second(self, price: float) -> float:
        """换算成价格漂移 μ（价格单位/秒），供做市与执行模型使用。"""
        if self.horizon_s <= 0:
            return 0.0
        return price * self.value * 1e-4 * self.confidence / self.horizon_s

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class LatencyProfile:
    """来自路由延迟优化库的链路画像（可按交易所/通道区分）。"""

    venue: str = "default"
    one_way_ms: float = 1.0
    jitter_ms: float = 0.2
    cancel_ack_ms: float = 1.0

    @property
    def round_trip_s(self) -> float:
        return 2.0 * (self.one_way_ms + 2.0 * self.jitter_ms) / 1000.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChildOrder:
    """输出给路由层的子单。price=None 表示市价。"""

    symbol: str
    side: Side
    qty: float
    price: Optional[float]
    order_type: OrderType = OrderType.LIMIT
    ts: float = 0.0
    parent_id: Optional[str] = None
    strategy: str = ""
    urgency: float = 0.0  # 0=被动，1=立即；路由层可据此选择通道/交易所
    venue_hint: Optional[str] = None
    order_id: str = field(default_factory=lambda: f"co-{next(_id_counter)}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        return d


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    price: float
    qty: float
    ts: float
    venue: str = "default"
    is_maker: bool = True


@dataclass
class ParentOrder:
    """大单（母单），由最优执行模块切分。"""

    symbol: str
    side: Side
    qty: float
    start_ts: float
    end_ts: float
    limit_price: Optional[float] = None
    parent_id: str = field(default_factory=lambda: f"po-{next(_id_counter)}")

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


@dataclass
class StrategyState:
    """策略运行状态（持仓、现金、剩余时间等）。"""

    inventory: float = 0.0
    cash: float = 0.0
    t: float = 0.0
    horizon_end: float = 1.0
    remaining_parent_qty: float = 0.0

    @property
    def time_left(self) -> float:
        return max(self.horizon_end - self.t, 0.0)


@runtime_checkable
class OrderPlacementStrategy(Protocol):
    """所有挂单/执行策略实现的统一协议。集成 agent 只依赖此协议。"""

    name: str

    def on_book(
        self,
        book: BookSnapshot,
        state: StrategyState,
        alpha: Optional[AlphaSignal] = None,
        latency: Optional[LatencyProfile] = None,
    ) -> List[ChildOrder]:
        ...

    def on_fill(self, fill: Fill, state: StrategyState) -> None:
        ...


@runtime_checkable
class Router(Protocol):
    """路由层需实现的最小协议（由路由延迟优化库提供）。"""

    def route(self, orders: Sequence[ChildOrder]) -> List[str]:
        ...

    def latency_profile(self, venue: Optional[str] = None) -> LatencyProfile:
        ...


@runtime_checkable
class AlphaProvider(Protocol):
    """因子层需实现的最小协议（由因子挖掘库提供）。"""

    def latest(self, symbol: str, ts: float) -> Optional[AlphaSignal]:
        ...
