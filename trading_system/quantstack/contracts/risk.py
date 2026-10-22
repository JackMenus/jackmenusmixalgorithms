"""风控契约。风控是 **确定性代码**，LLM agent 只能在限额之内调参，不能绕过。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .serialization import Contract


@dataclass
class RiskLimits(Contract):
    max_order_quantity: int = 1_000_000
    max_order_notional: float = 5_000_000.0
    max_position_notional: float = 10_000_000.0
    max_gross_exposure: float = 1e9
    max_net_exposure: float = 1e9
    max_daily_loss: float = 1e9            # 当日亏损（正数）超过即触发 kill switch
    price_band_bps: float = 300.0          # 子单限价偏离中间价的最大幅度
    max_participation: float = 0.3
    max_slices_per_minute: int = 3000
    allow_short: bool = True
    kill_switch: bool = False

    # agent 可调参数的允许区间（下限, 上限），超出即拒绝
    tunable_bounds: Dict[str, List[float]] = field(default_factory=lambda: {
        "price_band_bps": [20.0, 500.0],
        "max_participation": [0.01, 0.3],
        "max_order_notional": [1e4, 5e6],
    })


@dataclass
class RiskVerdict(Contract):
    approved: bool
    reasons: List[str] = field(default_factory=list)
    adjusted_quantity: int = 0             # 若被缩量，给出允许的数量；0 表示未调整

    @classmethod
    def ok(cls) -> "RiskVerdict":
        return cls(approved=True)

    @classmethod
    def reject(cls, *reasons: str) -> "RiskVerdict":
        return cls(approved=False, reasons=list(reasons))


@dataclass
class Incident(Contract):
    """需要 agent 介入（强推理）的异常事件。"""

    incident_id: str
    ts_us: int
    severity: str                          # info / warning / critical
    kind: str                              # risk_breach / execution_anomaly / data_quality / venue_degradation
    description: str
    context: Dict[str, float] = field(default_factory=dict)
