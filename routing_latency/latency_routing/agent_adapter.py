"""给集成 Agent 使用的适配层。

集成 Agent（把因子挖掘 → 挂单策略 → 路由串起来的那一层）只需要：

    svc = RoutingService()
    svc.observe_latency("SSE", 180.0)            # 由回报/心跳持续喂入延迟
    out = svc.route_json({...})                   # dict in / dict out，JSON 安全

``TOOL_SCHEMA`` 是可直接注册为 LLM 工具（Claude tool use / MCP）的 JSON Schema。
数值计算全部在本库内确定性完成，LLM 只负责编排与参数选择，不在热路径上。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .interfaces import ParentOrder, RoutingDecision, VenueQuote
from .latency.model import VenueLatencyModel
from .sor.router import LatencyAwareRouter, RouterConfig

TOOL_SCHEMA: Dict[str, Any] = {
    "name": "route_order",
    "description": "延迟感知智能订单路由：把母单拆分成多场所子单（吃单+挂单），并给出同步发送延时与预期成本。",
    "input_schema": {
        "type": "object",
        "required": ["order", "quotes"],
        "properties": {
            "order": {
                "type": "object",
                "required": ["symbol", "side", "quantity"],
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "integer", "enum": [1, -1]},
                    "quantity": {"type": "integer", "minimum": 1},
                    "limit_price": {"type": ["number", "null"]},
                    "urgency": {"type": "number", "minimum": 0, "maximum": 1, "description": "来自挂单策略库"},
                    "alpha_bps": {"type": "number", "description": "来自因子挖掘库的短期 alpha（基点）"},
                    "alpha_horizon_us": {"type": "number"},
                    "order_id": {"type": "string"},
                },
            },
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["venue", "bid", "ask", "bid_size", "ask_size"],
                    "properties": {
                        "venue": {"type": "string"},
                        "bid": {"type": "number"}, "ask": {"type": "number"},
                        "bid_size": {"type": "integer"}, "ask_size": {"type": "integer"},
                        "taker_fee": {"type": "number"}, "maker_fee": {"type": "number"},
                        "quote_update_rate": {"type": "number"},
                    },
                },
            },
        },
    },
}


class RoutingService:
    def __init__(self, config: Optional[RouterConfig] = None, latency_model: Optional[VenueLatencyModel] = None):
        self.latency = latency_model or VenueLatencyModel()
        self.router = LatencyAwareRouter(self.latency, config)

    def observe_latency(self, venue: str, latency_us: float) -> None:
        self.latency.observe(venue, latency_us)

    def route(self, order: ParentOrder, quotes: List[VenueQuote]) -> RoutingDecision:
        return self.router.route(order, quotes)

    def route_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        order = ParentOrder(**payload["order"])
        quotes = [VenueQuote(**q) for q in payload["quotes"]]
        return self.route(order, quotes).to_dict()

    def on_event(self, event) -> None:
        self.router.on_event(event)

    def venue_stats(self):
        return self.router.venue_stats()

    def latency_report(self) -> Dict[str, Dict[str, float]]:
        return self.latency.snapshot()
