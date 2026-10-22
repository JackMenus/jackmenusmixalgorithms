"""订单生命周期契约：母单 -> 子单切片 -> 场所订单 -> 成交/事件。

三层命名刻意区分，避免「ParentOrder/ChildOrder」在不同库里含义不同：

    ExecutionOrder   组合层/OMS 交给 **挂单策略库** 的母单（一个标的、一个方向、一个时间窗）
    OrderSlice       挂单策略库每次决策切出的子单（价格/数量/类型/紧急度），交给 **路由延迟库**
                     —— 字段与 ``latency_routing.interfaces.ParentOrder`` 一一对应
    VenueOrder       路由延迟库决定场所与发送时刻后的最终订单，交给网关/交易所
                     —— 字段与 ``latency_routing.interfaces.ChildOrder`` 一一对应
    Fill/OrderEvent  交易所回报，沿原路回传给路由层（场所统计）、执行层（剩余量）、OMS（持仓）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .serialization import Contract

ORDER_TYPES = ("LIMIT", "MARKET", "IOC", "POST_ONLY", "PEG")
BENCHMARKS = ("arrival", "vwap", "close", "twap")


@dataclass
class ExecutionOrder(Contract):
    """交给挂单策略库的母单。"""

    order_id: str
    symbol: str
    side: int                              # +1 买 / -1 卖
    quantity: int
    start_us: int
    end_us: int
    benchmark: str = "arrival"
    urgency: float = 0.5
    limit_price: Optional[float] = None    # 母单级别的价格保护（可选）
    max_participation: float = 0.1
    alpha_bps: float = 0.0
    alpha_horizon_us: int = 0
    strategy_hint: str = ""                # agent 建议的执行策略名（twap / almgren_chriss / pov / ...）
    params: Dict[str, float] = field(default_factory=dict)
    created_us: int = 0
    status: str = "new"                    # new / working / filled / cancelled / rejected
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    arrival_price: float = 0.0

    @property
    def remaining(self) -> int:
        return max(self.quantity - self.filled_quantity, 0)

    @property
    def is_complete(self) -> bool:
        return self.remaining == 0 or self.status in ("filled", "cancelled", "rejected")

    def apply_fill(self, quantity: int, price: float) -> None:
        total = self.filled_quantity + quantity
        if total > 0:
            self.avg_fill_price = (self.avg_fill_price * self.filled_quantity + price * quantity) / total
        self.filled_quantity = total
        self.status = "filled" if self.remaining == 0 else "working"


@dataclass
class OrderSlice(Contract):
    """挂单策略库输出、路由延迟库输入的子单切片。"""

    slice_id: str
    parent_id: str
    symbol: str
    side: int
    quantity: int
    order_type: str = "IOC"                # LIMIT / MARKET / IOC / POST_ONLY / PEG
    limit_price: Optional[float] = None
    urgency: float = 0.5                   # 0=完全被动 1=立即成交
    alpha_bps: float = 0.0                 # 按 side 方向为正表示有利
    alpha_horizon_us: int = 1_000_000
    ttl_us: int = 0                        # 0 表示不自动过期
    created_us: int = 0
    display_quantity: Optional[int] = None  # 冰山单可见量
    tag: str = ""                          # 策略内部标签（如 "passive"/"catchup"）

    def __post_init__(self) -> None:
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"order_type 必须是 {ORDER_TYPES}，收到 {self.order_type!r}")
        if self.side not in (1, -1):
            raise ValueError("side 必须是 +1 或 -1")
        if self.quantity <= 0:
            raise ValueError("quantity 必须为正")


@dataclass
class VenueOrder(Contract):
    """路由层输出、发送给具体场所的订单。"""

    venue_order_id: str
    slice_id: str
    parent_id: str
    venue: str
    symbol: str
    side: int
    quantity: int
    price: float
    order_type: str = "IOC"
    send_delay_us: float = 0.0             # 相对决策时刻的发送延时（用于同步到达）
    expected_arrival_us: float = 0.0
    created_us: int = 0
    route_reason: str = ""


@dataclass
class RoutingDecision(Contract):
    """一次路由决策的完整输出，包含可解释信息，便于 agent 记录与复盘。"""

    slice_id: str
    venue_orders: List[VenueOrder]
    expected_cost_bps: float = 0.0
    expected_fill_ratio: float = 1.0
    diagnostics: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Fill(Contract):
    fill_id: str
    venue_order_id: str
    slice_id: str
    parent_id: str
    symbol: str
    venue: str
    side: int
    quantity: int
    price: float
    ts_us: int
    fee: float = 0.0                       # 正数为付费，负数为返佣
    liquidity: str = "taker"               # maker / taker
    latency_us: float = 0.0                # 发送到成交回报的往返延迟

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class OrderEvent(Contract):
    """场所/网关回报事件。``kind`` ∈ ack / reject / cancel / expire / fill / partial。"""

    kind: str
    venue_order_id: str
    ts_us: int
    venue: str = ""
    reason: str = ""
    fill: Optional[Fill] = None
    remaining_quantity: int = 0


@dataclass
class CancelRequest(Contract):
    venue_order_id: str
    ts_us: int
    reason: str = ""
