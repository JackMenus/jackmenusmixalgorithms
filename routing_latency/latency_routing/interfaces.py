"""对外数据结构与接口协议。

这是本库与「因子挖掘库」「挂单策略库」以及集成 Agent 之间的唯一契约层。
所有结构都是纯 dataclass，可以用 ``to_dict()`` 序列化为 JSON，方便跨进程/跨语言传递。

时间单位约定：全部使用 **微秒 (us)**，价格使用浮点数，数量使用整数股/张。
方向约定：``side = +1`` 表示买，``side = -1`` 表示卖。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Protocol, Sequence, runtime_checkable


@dataclass
class VenueQuote:
    """某一交易场所当前的最优报价快照（我方视角，已扣除行情延迟前的时间戳）。"""

    venue: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    ts_us: int = 0                 # 行情在交易所侧产生的时间戳
    # 每股费用：正数为收费，负数为返佣（maker rebate）
    taker_fee: float = 0.0
    maker_fee: float = 0.0
    # 该场所队首到成交的平均撤单/更新强度（次/秒），用于估算报价陈旧风险
    quote_update_rate: float = 50.0

    def far_side(self, side: int) -> tuple[float, int]:
        """吃单方向的价格和数量：买单看 ask，卖单看 bid。"""
        return (self.ask, self.ask_size) if side > 0 else (self.bid, self.bid_size)

    def near_side(self, side: int) -> tuple[float, int]:
        """挂单方向的价格和数量：买单挂 bid，卖单挂 ask。"""
        return (self.bid, self.bid_size) if side > 0 else (self.ask, self.ask_size)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParentOrder:
    """上游（挂单策略库 / Agent）交给路由层的母单。"""

    symbol: str
    side: int                      # +1 买 / -1 卖
    quantity: int
    limit_price: Optional[float] = None
    urgency: float = 0.5           # 0=完全被动 1=立即成交；由挂单策略或因子信号给出
    alpha_bps: float = 0.0         # 因子库预测的短期 alpha（基点，按 side 方向为正表示有利）
    alpha_horizon_us: float = 1e6  # alpha 的有效期
    order_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChildOrder:
    """路由层输出的子单。"""

    venue: str
    side: int
    quantity: int
    price: float
    order_type: str = "IOC"        # IOC / LIMIT / POST_ONLY
    send_delay_us: float = 0.0     # 相对决策时刻的发送延时（用于同步到达）
    expected_arrival_us: float = 0.0
    parent_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoutingDecision:
    """一次路由决策的完整输出，包含可解释信息，便于 Agent 记录与复盘。"""

    children: List[ChildOrder]
    expected_cost_bps: float
    expected_fill_ratio: float
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "children": [c.to_dict() for c in self.children],
            "expected_cost_bps": self.expected_cost_bps,
            "expected_fill_ratio": self.expected_fill_ratio,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class VenueStats:
    """场所质量统计（字段与集成层 quantstack.contracts.tca.VenueStats 一致）。"""

    venue: str
    fills: int = 0
    rejects: int = 0
    fill_ratio: float = 0.0        # 成交量 / 路由发出量
    avg_slippage_bps: float = 0.0  # 成交价相对路由时中间价，按方向不利为正
    latency_p50_us: float = 0.0    # 单程延迟估计
    latency_p99_us: float = 0.0
    maker_ratio: float = 0.0       # maker 成交量占比

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class LatencyModelProtocol(Protocol):
    """延迟模型协议：任何实现了这三个方法的对象都能被路由器使用。"""

    def observe(self, venue: str, latency_us: float) -> None: ...

    def quantile(self, venue: str, q: float) -> float: ...

    def sample(self, venue: str, n: int) -> "Sequence[float]": ...


@runtime_checkable
class RouterProtocol(Protocol):
    """路由器协议，集成 Agent 只依赖这个接口。"""

    def route(self, order: ParentOrder, quotes: Sequence[VenueQuote]) -> RoutingDecision: ...

    def on_event(self, event) -> None:
        """接收回报事件（鸭子类型，兼容 quantstack.contracts.orders.OrderEvent）：
        需要属性 kind / venue，fill 事件还需 fill.{quantity, price, side, liquidity, latency_us, symbol}。"""
        ...

    def venue_stats(self) -> List[VenueStats]: ...
