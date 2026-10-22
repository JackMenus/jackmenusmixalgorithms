"""交易成本分析（TCA）契约：执行结果的统一度量，反馈给因子/执行/路由三层与 agent。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .serialization import Contract


@dataclass
class TCAReport(Contract):
    parent_id: str
    symbol: str
    side: int
    quantity: int
    filled_quantity: int
    arrival_price: float
    avg_fill_price: float
    implementation_shortfall_bps: float    # (avg_fill - arrival)/arrival * side * 1e4（正数=成本）
    fill_ratio: float
    fees: float = 0.0
    duration_us: int = 0
    maker_ratio: float = 0.0
    n_slices: int = 0
    n_venue_orders: int = 0
    venue_breakdown: Dict[str, float] = field(default_factory=dict)   # 场所 -> 成交量占比
    venue_slippage_bps: Dict[str, float] = field(default_factory=dict)
    strategy: str = ""
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class VenueStats(Contract):
    """路由层维护、agent 可读的场所质量统计。"""

    venue: str
    fills: int = 0
    rejects: int = 0
    fill_ratio: float = 0.0
    avg_slippage_bps: float = 0.0
    latency_p50_us: float = 0.0
    latency_p99_us: float = 0.0
    maker_ratio: float = 0.0
