"""行情契约。

约定（与路由延迟库保持一致）：
  * 时间戳 ``ts_us``：自 epoch 起的 **微秒** 整数；日频数据用 pandas 的 DatetimeIndex。
  * 方向 ``side``：+1 买，-1 卖。
  * 价格为 float，数量为整数股/张。
  * 日频面板 ``Panel = dict[str, pd.DataFrame]``：index 为时间、columns 为资产、字段名小写
    （open/high/low/close/volume/vwap/returns），与因子挖掘库 ``factormine.data.Panel`` 完全相同。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:  # pandas 仅在因子层/面板数据场景需要，契约层不强依赖
    import pandas as pd

    Panel = Dict[str, "pd.DataFrame"]
except Exception:  # pragma: no cover
    Panel = Dict[str, object]  # type: ignore[misc]

from .serialization import Contract

BUY = 1
SELL = -1
US_PER_SEC = 1_000_000
US_PER_MS = 1_000


@dataclass
class Bar(Contract):
    """一根 K 线（日频或分钟频）。"""

    symbol: str
    ts_us: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


@dataclass
class VenueQuote(Contract):
    """某交易场所对某标的的最优报价快照。

    字段与 ``latency_routing.interfaces.VenueQuote`` 对齐（本结构多一个 ``symbol``），
    ``adapters.latency_routing_adapter.to_routing_quote`` 负责互转。
    费用约定：每股费用，正数为收费，负数为返佣（maker rebate）。
    """

    venue: str
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    ts_us: int = 0
    taker_fee: float = 0.0
    maker_fee: float = 0.0
    quote_update_rate: float = 50.0  # 报价更新强度（次/秒），用于估算报价陈旧风险

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def far_side(self, side: int) -> Tuple[float, int]:
        """吃单方向的价格和数量：买单看 ask，卖单看 bid。"""
        return (self.ask, self.ask_size) if side > 0 else (self.bid, self.bid_size)

    def near_side(self, side: int) -> Tuple[float, int]:
        """挂单方向的价格和数量：买单挂 bid，卖单挂 ask。"""
        return (self.bid, self.bid_size) if side > 0 else (self.ask, self.ask_size)


@dataclass
class MarketSnapshot(Contract):
    """某标的在某时刻的跨场所行情快照，是执行层与路由层共同的输入。"""

    symbol: str
    ts_us: int
    quotes: List[VenueQuote] = field(default_factory=list)
    last_price: Optional[float] = None
    adv: Optional[float] = None          # 平均日成交量（股），用于参与率/冲击估计
    volatility: Optional[float] = None   # 日收益波动率（小数），用于 Almgren-Chriss 等
    session_volume: float = 0.0          # 当日累计成交量（股）
    extra: Dict[str, float] = field(default_factory=dict)

    def nbbo(self) -> Tuple[float, float]:
        """跨场所最优买卖价（National Best Bid and Offer）。"""
        if not self.quotes:
            p = self.last_price or 0.0
            return p, p
        return max(q.bid for q in self.quotes), min(q.ask for q in self.quotes)

    @property
    def mid(self) -> float:
        bid, ask = self.nbbo()
        return 0.5 * (bid + ask)

    @property
    def spread_bps(self) -> float:
        bid, ask = self.nbbo()
        mid = 0.5 * (bid + ask)
        return 1e4 * (ask - bid) / mid if mid > 0 else 0.0

    def quote(self, venue: str) -> Optional[VenueQuote]:
        for q in self.quotes:
            if q.venue == venue:
                return q
        return None

    def touch_price(self, side: int) -> float:
        """吃单价：买单取最优 ask，卖单取最优 bid。"""
        bid, ask = self.nbbo()
        return ask if side > 0 else bid

    def displayed_size(self, side: int) -> int:
        """吃单方向上所有场所的可见挂单量之和。"""
        return int(sum(q.far_side(side)[1] for q in self.quotes))
