"""暴露给模型的工具：只读查询 + 有边界的调参 + 熔断。

工具只是流水线对象的薄包装；真正的权限边界在 ``PreTradeRiskGate.set_limit``（参数区间）与
``TradingPipeline.halt``（只能停不能开）。模型无法通过工具下单。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..contracts.serialization import _to_jsonable


def _j(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, default=str)


class ToolBox:
    """工具的纯 Python 实现（离线后端 / 单测直接调用）。``build_tools`` 把它们包装成 SDK 工具。"""

    def __init__(self, pipeline, impact_model=None):
        self.pipeline = pipeline
        self.impact_model = impact_model
        self.audit: List[Dict[str, Any]] = []

    # ---- 只读
    def get_portfolio(self) -> str:
        pf = self.pipeline.portfolio_snapshot()
        return _j({
            "ts_us": pf.ts_us, "cash": round(pf.cash, 2), "nav": round(pf.nav, 2), "gross_exposure": round(pf.gross_exposure, 2),
            "net_exposure": round(pf.net_exposure, 2), "realized_pnl": round(pf.realized_pnl, 2), "unrealized_pnl": round(pf.unrealized_pnl, 2),
            "positions": {k: {"quantity": p.quantity, "avg_price": round(p.avg_price, 4), "last_price": round(p.last_price, 4)} for k, p in pf.positions.items()},
        })

    def get_open_orders(self) -> str:
        return _j([{"order_id": o.order_id, "symbol": o.symbol, "side": o.side, "quantity": o.quantity, "filled": o.filled_quantity, "status": o.status, "strategy": o.strategy_hint, "urgency": o.urgency}
                   for o in self.pipeline.oms.open_orders()])

    def get_execution_strategies(self) -> str:
        return _j(self.pipeline.execution_factory.available())

    def get_tca_summary(self) -> str:
        return _j(self.pipeline.tca.summary())

    def get_tca_report(self, order_id: str) -> str:
        r = self.pipeline.tca.report(order_id)
        return _j(r.to_dict() if r else {"error": f"未知母单 {order_id}"})

    def get_venue_stats(self) -> str:
        return _j([v.to_dict() for v in self.pipeline.router.venue_stats()])

    def get_recent_events(self, kind: str = "", n: int = 20) -> str:
        n = max(1, min(int(n), 200))
        return _j(self.pipeline.bus.records(n, kind or None))

    def get_risk_limits(self) -> str:
        return _j(self.pipeline.risk_gate.limits.to_dict())

    def get_market_snapshot(self, symbol: str) -> str:
        try:
            snap = self.pipeline.feed.snapshot(symbol, self.pipeline.now())
        except KeyError:
            return _j({"error": f"未知标的 {symbol}"})
        bid, ask = snap.nbbo()
        return _j({"symbol": symbol, "bid": bid, "ask": ask, "mid": snap.mid, "spread_bps": round(snap.spread_bps, 2), "adv": snap.adv, "volatility": snap.volatility,
                   "displayed_buy": snap.displayed_size(1), "displayed_sell": snap.displayed_size(-1), "quotes": [q.to_dict() for q in snap.quotes]})

    def estimate_impact(self, symbol: str, quantity: int, duration_s: float = 1800.0) -> str:
        try:
            snap = self.pipeline.feed.snapshot(symbol, self.pipeline.now())
        except KeyError:
            return _j({"error": f"未知标的 {symbol}"})
        if self.impact_model is not None:
            bps = float(self.impact_model.expected_cost_bps(int(quantity), snap, int(duration_s * 1_000_000)))
            model = type(self.impact_model).__name__
        else:
            # 平方根律兜底：cost ≈ 半价差 + 0.5·σ_daily·sqrt(Q/ADV)
            adv = snap.adv or 1e6
            bps = snap.spread_bps / 2 + 0.5 * (snap.volatility or 0.02) * ((int(quantity) / adv) ** 0.5) * 1e4
            model = "sqrt_law_fallback"
        return _j({"symbol": symbol, "quantity": int(quantity), "duration_s": duration_s, "expected_cost_bps": round(bps, 3), "pct_adv": round(int(quantity) / (snap.adv or 1e6), 4), "model": model})

    # ---- 有边界的写操作
    def set_risk_limit(self, name: str, value: float) -> str:
        ok, msg = self.pipeline.risk_gate.set_limit(name, float(value))
        self.audit.append({"tool": "set_risk_limit", "name": name, "value": value, "ok": ok, "msg": msg})
        self.pipeline.bus.publish("agent_param_change", {"name": name, "value": value, "ok": ok, "msg": msg}, self.pipeline.now(), "agent")
        return _j({"ok": ok, "message": msg})

    def halt_trading(self, reason: str) -> str:
        self.pipeline.halt(f"agent: {reason}")
        self.audit.append({"tool": "halt_trading", "reason": reason})
        return _j({"ok": True, "message": "已熔断：全部在途母单收尾，拒绝新订单。恢复需要人工确认。"})


def build_tools(toolbox: ToolBox, include_write: bool = True) -> list:
    """把 ToolBox 方法包装为 Anthropic SDK 的 ``@beta_tool`` 工具列表。"""
    from anthropic import beta_tool

    tb = toolbox

    @beta_tool
    def get_portfolio() -> str:
        """读取当前组合快照：现金、净值、总/净敞口、已实现与未实现盈亏、各持仓。"""
        return tb.get_portfolio()

    @beta_tool
    def get_open_orders() -> str:
        """列出未完成母单：标的、方向、数量、已成交、状态、当前执行策略与紧急度。"""
        return tb.get_open_orders()

    @beta_tool
    def get_execution_strategies() -> str:
        """列出挂单策略库当前可用的执行策略名及其说明/参数。执行计划里的 strategy 必须来自这里。"""
        return tb.get_execution_strategies()

    @beta_tool
    def get_tca_summary() -> str:
        """读取交易成本分析摘要：加权实施缺口、成交率、费用、maker 占比、分策略/分场所滑点、异常母单。"""
        return tb.get_tca_summary()

    @beta_tool
    def get_tca_report(order_id: str) -> str:
        """读取某张母单的详细 TCA 报告。

        Args:
            order_id: 母单编号，如 O000003。
        """
        return tb.get_tca_report(order_id)

    @beta_tool
    def get_venue_stats() -> str:
        """读取路由层维护的场所质量统计：成交率、拒单数、滑点、延迟 P50/P99、maker 占比。"""
        return tb.get_venue_stats()

    @beta_tool
    def get_recent_events(kind: str = "", n: int = 20) -> str:
        """读取事件日志尾部。

        Args:
            kind: 事件类型过滤（如 venue_reject / route / slice_expired / halt），空字符串表示全部。
            n: 返回条数（1-200）。
        """
        return tb.get_recent_events(kind, n)

    @beta_tool
    def get_risk_limits() -> str:
        """读取当前风控限额与 agent 可调参数的允许区间。"""
        return tb.get_risk_limits()

    @beta_tool
    def get_market_snapshot(symbol: str) -> str:
        """读取某标的的跨场所实时行情快照（NBBO、价差、各场所报价与可见量、ADV、波动率）。

        Args:
            symbol: 标的代码。
        """
        return tb.get_market_snapshot(symbol)

    @beta_tool
    def estimate_impact(symbol: str, quantity: int, duration_s: float = 1800.0) -> str:
        """估计在给定时长内执行 quantity 股的预期成本（基点），用于与 alpha 做权衡。

        Args:
            symbol: 标的代码。
            quantity: 股数。
            duration_s: 执行时长（秒）。
        """
        return tb.estimate_impact(symbol, quantity, duration_s)

    tools = [get_portfolio, get_open_orders, get_execution_strategies, get_tca_summary, get_tca_report, get_venue_stats, get_recent_events, get_risk_limits, get_market_snapshot, estimate_impact]
    if include_write:

        @beta_tool
        def set_risk_limit(name: str, value: float) -> str:
            """在允许区间内修改一个风控可调参数（price_band_bps / max_participation / max_order_notional）。越界会被拒绝。

            Args:
                name: 参数名。
                value: 新值。
            """
            return tb.set_risk_limit(name, value)

        @beta_tool
        def halt_trading(reason: str) -> str:
            """触发熔断：停止全部交易并收尾在途订单。只在发现严重风险时使用；恢复交易需要人工。

            Args:
                reason: 熔断原因。
            """
            return tb.halt_trading(reason)

        tools += [set_risk_limit, halt_trading]
    return tools
