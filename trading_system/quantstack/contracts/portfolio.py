"""组合/持仓契约：组合层 -> OMS/执行层 的接口对象。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .serialization import Contract


@dataclass
class Position(Contract):
    symbol: str
    quantity: int = 0                      # 正多负空
    avg_price: float = 0.0
    last_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.avg_price) * self.quantity


@dataclass
class PortfolioSnapshot(Contract):
    ts_us: int
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    fees_paid: float = 0.0

    @property
    def nav(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    def quantity(self, symbol: str) -> int:
        p = self.positions.get(symbol)
        return p.quantity if p else 0


@dataclass
class TargetPosition(Contract):
    """组合层给出的目标持仓。OMS 用它与当前持仓做差得到母单。

    * ``urgency`` 0..1：越接近 1 越需要尽快完成（alpha 衰减快 / 风险事件）。
    * ``benchmark``：执行基准，arrival（到达价，默认）/ vwap / close。
    * ``alpha_bps`` / ``alpha_horizon_us``：透传给执行层和路由层做成本-alpha 权衡。
    """

    symbol: str
    target_quantity: int
    target_weight: float = 0.0
    urgency: float = 0.5
    max_participation: float = 0.1         # 最大成交量参与率
    deadline_us: Optional[int] = None
    benchmark: str = "arrival"
    alpha_bps: float = 0.0
    alpha_horizon_us: int = 0
    reason: str = ""
