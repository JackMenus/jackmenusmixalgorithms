"""LLM 后端：ClaudeBackend（Anthropic SDK，真实模型）与 ScriptedBackend（离线、确定性）。

两者返回同构的 ``AgentResponse``，编排器不关心背后是谁；ScriptedBackend 让整条流水线在没有 API 凭据的
环境里也能端到端运行、可测试，并作为真实模型决策的「基线对照」。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Type

from pydantic import BaseModel

from . import decisions as D
from .model_policy import ComplexitySignals, ModelSpec, TaskClass

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AgentRequest:
    task: TaskClass
    system: str
    user: str
    context: Dict[str, Any] = field(default_factory=dict)   # 结构化上下文：离线后端直接读；在线后端已渲染进 user
    tools: Sequence[Any] = ()
    output_model: Optional[Type[BaseModel]] = None
    signals: Optional[ComplexitySignals] = None
    max_attempts: int = 2
    max_tool_iterations: int = 12


@dataclass
class AgentResponse:
    task: TaskClass
    text: str = ""
    parsed: Any = None
    model: str = ""
    effort: str = ""
    attempts: int = 1
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    refused: bool = False
    error: str = ""
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.refused and not self.error and (self.parsed is not None or bool(self.text))

    def to_record(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "model": self.model,
            "effort": self.effort,
            "attempts": self.attempts,
            "ok": self.ok,
            "refused": self.refused,
            "error": self.error,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "n_tool_calls": len(self.tool_calls),
            "latency_s": round(self.latency_s, 2),
            "parsed": self.parsed.model_dump() if isinstance(self.parsed, BaseModel) else self.parsed,
            "text": self.text[:4000],
        }


class LLMBackend(Protocol):
    def run(self, request: AgentRequest, spec: ModelSpec) -> AgentResponse: ...


# ============================================================================= Claude
class ClaudeBackend:
    """通过 Anthropic SDK 调用 Claude。

    * 有工具 -> ``client.beta.messages.tool_runner``（SDK 负责工具循环；``output_format`` 产出结构化决策）
    * 无工具 + 结构化 -> ``client.beta.messages.stream(..., output_format=Model)``
    * 无工具 + 自由文本 -> ``client.beta.messages.stream``
    默认开启服务端 refusal fallback（``fallbacks="default"``），遇到安全分类拒答时由 API 自动换模型重跑。
    """

    def __init__(self, client=None, enable_fallbacks: bool = True, extra_betas: Sequence[str] = ()):
        import anthropic

        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.enable_fallbacks = enable_fallbacks
        self.extra_betas = list(extra_betas)

    def _kwargs(self, spec: ModelSpec) -> Dict[str, Any]:
        kw = spec.request_kwargs()
        betas = list(self.extra_betas)
        if self.enable_fallbacks:
            betas.append(FALLBACK_BETA)
            kw["fallbacks"] = "default"
        if betas:
            kw["betas"] = betas
        return kw

    def run(self, request: AgentRequest, spec: ModelSpec) -> AgentResponse:
        t0 = time.time()
        resp = AgentResponse(task=request.task, model=spec.model, effort=spec.effort)
        kw = self._kwargs(spec)
        client = self.client.with_options(timeout=spec.timeout_s)
        system = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
        messages = [{"role": "user", "content": request.user}]
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}

        def add_usage(msg) -> None:
            u = getattr(msg, "usage", None)
            if u is None:
                return
            for k in usage:
                usage[k] += int(getattr(u, k, 0) or 0)

        try:
            if request.tools:
                extra: Dict[str, Any] = {}
                if request.output_model is not None:
                    extra["output_format"] = request.output_model
                runner = client.beta.messages.tool_runner(
                    messages=messages, system=system, tools=list(request.tools), max_iterations=request.max_tool_iterations, **extra, **kw
                )
                final = None
                for message in runner:
                    final = message
                    add_usage(message)
                    for block in message.content:
                        if getattr(block, "type", "") == "tool_use":
                            resp.tool_calls.append({"name": block.name, "input": block.input})
            else:
                extra = {"output_format": request.output_model} if request.output_model is not None else {}
                with client.beta.messages.stream(messages=messages, system=system, **extra, **kw) as stream:
                    final = stream.get_final_message()
                add_usage(final)
            if final is None:
                resp.error = "empty response"
                return resp
            resp.stop_reason = str(getattr(final, "stop_reason", "") or "")
            resp.model = str(getattr(final, "model", spec.model) or spec.model)
            resp.refused = resp.stop_reason == "refusal"
            resp.text = "".join(getattr(b, "text", "") for b in final.content if getattr(b, "type", "") == "text")
            parsed = getattr(final, "parsed_output", None)
            if parsed is None and request.output_model is not None and resp.text and not resp.refused:
                try:
                    parsed = request.output_model.model_validate_json(resp.text)
                except Exception as e:  # noqa: BLE001
                    resp.error = f"structured output parse failed: {e}"
            resp.parsed = parsed
            if resp.refused and getattr(final, "stop_details", None) is not None:
                resp.error = f"refusal: {getattr(final.stop_details, 'category', '')}"
        except self._anthropic.RateLimitError as e:
            resp.error = f"rate_limited: {e}"
        except self._anthropic.APIStatusError as e:
            resp.error = f"api_error {e.status_code}: {getattr(e, 'message', e)}"
        except self._anthropic.APIConnectionError as e:
            resp.error = f"connection_error: {e}"
        resp.usage = usage
        resp.latency_s = time.time() - t0
        return resp


# ============================================================================= Scripted（离线）
class ScriptedBackend:
    """确定性规则后端：按任务类读取 ``request.context``，产出与真实模型同构的结构化决策。

    它不是「假装的 LLM」，而是系统的 **安全基线**：真实模型不可用（无凭据 / 限流 / 拒答）时编排器回退到它，
    保证交易流水线永远有一个可解释的默认决策。
    """

    def run(self, request: AgentRequest, spec: ModelSpec) -> AgentResponse:
        t0 = time.time()
        ctx = request.context or {}
        task = request.task
        handler = getattr(self, f"_{task.value}", None)
        parsed = handler(ctx) if handler else None
        text = parsed.model_dump_json(indent=None) if isinstance(parsed, BaseModel) else json.dumps(ctx.get("summary", {}), ensure_ascii=False)[:2000]
        if parsed is None and request.output_model is not None:
            try:
                parsed = request.output_model()  # 允许全默认的模型
            except Exception:  # noqa: BLE001
                parsed = None
        return AgentResponse(task=task, text=text, parsed=parsed, model=f"scripted({spec.model})", effort=spec.effort, stop_reason="end_turn", latency_s=time.time() - t0)

    # ------------------------------------------------------------------ 执行规划
    def _execution_plan(self, ctx: Dict[str, Any]) -> D.ExecutionPlan:
        available: Dict[str, str] = ctx.get("available", {})
        plans = []
        for o in ctx.get("orders", []):
            urgency = float(o.get("urgency", 0.5))
            pct_adv = float(o.get("pct_adv", 0.0))
            horizon_s = float(o.get("alpha_horizon_s", 0.0))
            window_s = float(o.get("window_s", 1800.0))
            spread_bps = float(o.get("spread_bps", 5.0))
            if urgency >= 0.75 and "pov" in available:
                name, params, why = "pov", {"rate": float(min(max(o.get("max_participation", 0.1), 0.05), 0.25))}, "紧急度高：按参与率快速完成"
            elif (pct_adv > 0.02 or (0 < horizon_s < 2 * window_s)) and "almgren_chriss" in available:
                name, params, why = "almgren_chriss", {"risk_aversion": 1e-5 * (3.0 if horizon_s and horizon_s < window_s else 1.0)}, "母单较大或 alpha 衰减快：最优轨迹前置执行"
            elif urgency < 0.35 and spread_bps > 8 and "adaptive_passive" in available:
                name, params, why = "adaptive_passive", {}, "价差宽且不急：被动挂单赚价差与返佣"
            elif "twap" in available:
                name, params, why = "twap", {}, "常规：等时间切片，落后时追赶"
            else:
                name, params, why = next(iter(available), "twap"), {}, "默认"
            plans.append(D.OrderPlan(order_id=o["order_id"], strategy=name, params=params, rationale=why))
        return D.ExecutionPlan(plans=plans, notes="scripted baseline", confidence=0.55)

    # ------------------------------------------------------------------ 交易后复盘
    def _post_trade_review(self, ctx: Dict[str, Any]) -> D.PostTradeReview:
        tca = ctx.get("tca", {}) or {}
        w_is = float(tca.get("weighted_is_bps", 0.0) or 0.0)
        fill = float(tca.get("avg_fill_ratio", 1.0) or 1.0)
        anomalies = [f"{a.get('order_id')} {a.get('symbol')}: IS={a.get('is_bps')}bps fill={a.get('fill_ratio')}" for a in tca.get("anomalies", [])]
        limits = ctx.get("limits", {}) or {}
        changes: List[D.ParamChange] = []
        if w_is > 30 and "max_participation" in limits:
            changes.append(D.ParamChange(name="max_participation", value=round(max(0.05, float(limits["max_participation"]) * 0.8), 4), rationale="实施缺口偏高，降低参与率以减少冲击"))
        venue_fb = {}
        for v in ctx.get("venue_stats", []):
            if v.get("fills", 0) + v.get("rejects", 0) >= 20 and (v.get("fill_ratio", 1) < 0.5 or v.get("rejects", 0) > v.get("fills", 1)):
                venue_fb[v["venue"]] = "成交率低/拒单多：降低该场所的吃单权重，检查延迟 P99"
        strat_fb = {k: ("成本偏高，考虑更被动或拉长窗口" if v > 25 else "表现正常") for k, v in (tca.get("is_bps_by_strategy", {}) or {}).items()}
        incidents = ctx.get("incidents", [])
        escalate = bool(incidents) or w_is > 50 or fill < 0.7
        reason = "存在风控事故" if incidents else ("加权实施缺口超过 50bps" if w_is > 50 else ("平均成交率低于 70%" if fill < 0.7 else ""))
        summary = f"{tca.get('n_orders', 0)} 张母单，加权实施缺口 {w_is:.1f}bps，平均成交率 {fill:.1%}，maker 占比 {float(tca.get('maker_ratio', 0) or 0):.0%}。"
        return D.PostTradeReview(summary=summary, anomalies=anomalies, param_changes=changes, strategy_feedback=strat_fb, venue_feedback=venue_fb, escalate=escalate, escalation_reason=reason, confidence=0.6)

    # ------------------------------------------------------------------ 深度复盘 / 事故
    def _incident(self, ctx: Dict[str, Any]) -> D.IncidentResponse:
        incidents = ctx.get("incidents", [])
        tca = ctx.get("tca", {}) or {}
        desc = "; ".join(i.get("description", "") for i in incidents) or "执行异常：成本或成交率偏离基线"
        kill = any(i.get("kind") == "risk_breach" for i in incidents)
        actions = ["保持熔断，停止新母单" if kill else "继续交易但收紧参与率", "撤销全部在途被动挂单", "复核今日信号与持仓敞口"]
        changes = [D.ParamChange(name="max_participation", value=0.05, rationale="事故期间降低冲击与暴露")]
        return D.IncidentResponse(
            diagnosis=desc,
            root_causes=[a if isinstance(a, str) else json.dumps(a, ensure_ascii=False) for a in tca.get("anomalies", [])][:5] or ["待人工确认"],
            immediate_actions=actions,
            resume_trading=not kill,
            param_changes=changes,
            follow_ups=["对比异常母单的路由诊断与场所延迟 P99", "检查因子信号是否在该标的上失效"],
            confidence=0.5,
        )

    _post_mortem = _incident
    _risk_incident = _incident

    # ------------------------------------------------------------------ 研究评审
    def _research_review(self, ctx: Dict[str, Any]) -> D.ResearchDecision:
        deploy, retire = [], []
        for r in ctx.get("reports", []):
            ok = abs(float(r.get("ic_mean", 0))) >= 0.02 and abs(float(r.get("ic_ir", 0))) >= 0.3 and float(r.get("coverage", 1)) >= 0.8 and float(r.get("turnover", 0)) <= 0.6
            (deploy if ok else retire).append(r.get("name", "?"))
        return D.ResearchDecision(
            deploy_factors=deploy,
            retire_factors=retire,
            hypotheses=["对高换手因子做低频平滑版本", "行业/市值中性化后重评", "检验因子在高波动状态下的稳定性"],
            rationale="按 IC≥0.02、ICIR≥0.3、覆盖≥80%、换手≤60% 的基线门槛筛选",
            confidence=0.5,
        )

    _factor_hypothesis = _research_review

    def _data_quality(self, ctx: Dict[str, Any]) -> D.DataQualityReport:
        issues = list(ctx.get("issues", []))
        return D.DataQualityReport(ok=not issues, issues=issues, actions=["剔除问题标的后继续" if issues else "无"])

    def _daily_report(self, ctx: Dict[str, Any]) -> D.DailyReport:
        b = ctx.get("brief", {}) or {}
        tca = b.get("tca", {}) or {}
        pnl = float(b.get("nav_end", 0)) - float(b.get("nav_start", 0))
        return D.DailyReport(
            headline=f"当日 PnL {pnl:,.0f}，{b.get('n_orders', 0)} 张母单，实施缺口 {float(tca.get('weighted_is_bps', 0) or 0):.1f}bps",
            performance=f"净值 {b.get('nav_start')} -> {b.get('nav_end')}；总敞口 {b.get('gross_exposure')}，净敞口 {b.get('net_exposure')}",
            execution=f"策略分布 {b.get('plans')}；成交率 {tca.get('avg_fill_ratio')}；maker 占比 {tca.get('maker_ratio')}",
            routing=f"场所统计 {[(v.get('venue'), round(v.get('fill_ratio', 0), 2), round(v.get('latency_p50_us', 0))) for v in b.get('venue_stats', [])]}",
            risks=f"事故 {len(b.get('incidents', []))} 起；被拒母单 {b.get('n_rejected', 0)}",
            next_steps=["复核异常母单", "更新场所延迟先验"],
        )
