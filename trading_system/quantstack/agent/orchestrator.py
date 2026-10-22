"""TradingAgent：把 LLM 决策接到流水线的四个决策点上。

    每日周期：数据质量 -> 信号 -> [研究评审*] -> 目标/母单/风控 -> 执行规划 -> 执行循环 -> 交易后复盘 -> [深度复盘*] -> 日报
    事故：风控闸门触发 kill switch -> [事故响应*]
    研究：因子假设* -> 挖掘 -> 评估 -> 研究评审*        （* = 强推理：Fable 5.1 effort=max）

每个决策都：按 ``ModelPolicy`` 选模型与思考强度 -> 调后端 -> 结构化校验 -> 失败则沿阶梯升级重试 -> 仍失败则回退
``ScriptedBackend`` 基线 -> 记录到 ``decisions.jsonl``。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from ..contracts import US_PER_SEC, ExecutionOrder, Incident, Panel
from ..contracts.serialization import _to_jsonable
from . import decisions as D
from .backend import AgentRequest, AgentResponse, LLMBackend, ScriptedBackend
from .model_policy import ComplexitySignals, ModelPolicy, ModelSpec, TaskClass
from .roles import system_prompt
from .tools import ToolBox, build_tools


@dataclass
class DecisionLog:
    task: str
    spec: str
    response: Dict[str, Any]
    applied: Dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0


class TradingAgent:
    def __init__(
        self,
        pipeline,
        backend: Optional[LLMBackend] = None,
        policy: Optional[ModelPolicy] = None,
        *,
        fallback: Optional[LLMBackend] = None,
        log_dir: Optional[str] = None,
        impact_model=None,
        use_tools: bool = True,
        miner=None,
    ):
        self.pipeline = pipeline
        self.policy = policy or ModelPolicy()
        self.backend: LLMBackend = backend or ScriptedBackend()
        self.fallback: LLMBackend = fallback or ScriptedBackend()
        self.toolbox = ToolBox(pipeline, impact_model=impact_model)
        self.use_tools = use_tools and not isinstance(self.backend, ScriptedBackend)
        self._tools = None
        self.miner = miner
        self.log_dir = log_dir
        self.decisions: List[DecisionLog] = []
        self.deployed_factors: List[str] = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------ 决策内核
    def _tools_for(self, task: TaskClass):
        if not self.use_tools:
            return ()
        if self._tools is None:
            self._tools = build_tools(self.toolbox, include_write=True)
        if task in (TaskClass.EXECUTION_PLAN, TaskClass.DAILY_REPORT, TaskClass.DATA_QUALITY):
            return [t for t in self._tools if t.name not in ("set_risk_limit", "halt_trading")]
        return self._tools

    def decide(self, task: TaskClass, context: Dict[str, Any], output_model, signals: Optional[ComplexitySignals] = None, instruction: str = "") -> AgentResponse:
        """按模型策略执行一次决策，带升级重试与基线回退。"""
        t0 = time.time()
        ctx_json = json.dumps(_to_jsonable(context), ensure_ascii=False, default=str)
        if signals is None:
            signals = ComplexitySignals(input_chars=len(ctx_json))
        else:
            signals.input_chars = signals.input_chars or len(ctx_json)
        user = (instruction + "\n\n" if instruction else "") + "以下是结构化上下文（JSON）：\n" + ctx_json + "\n\n请直接给出结构化决策。"
        last: Optional[AgentResponse] = None
        max_attempts = 1 if self.policy.is_strong(task) else 3
        for attempt in range(max_attempts):
            spec = self.policy.resolve(task, signals, attempt)
            req = AgentRequest(task=task, system=system_prompt(task), user=user, context=context, tools=self._tools_for(task), output_model=output_model, signals=signals)
            resp = self.backend.run(req, spec)
            resp.attempts = attempt + 1
            last = resp
            if resp.ok and (output_model is None or isinstance(resp.parsed, output_model)):
                conf = float(getattr(resp.parsed, "confidence", 1.0) or 1.0)
                if conf >= 0.4 or attempt == max_attempts - 1 or self.policy.is_strong(task):
                    break
                signals.prior_failures += 1   # 低置信 -> 升档再试
            else:
                signals.prior_failures += 1
        assert last is not None
        if not (last.ok and (output_model is None or isinstance(last.parsed, output_model))):
            fb_spec = self.policy.resolve(task, signals, 0)
            fb = self.fallback.run(AgentRequest(task=task, system=system_prompt(task), user=user, context=context, output_model=output_model), fb_spec)
            fb.attempts = last.attempts + 1
            fb.error = (last.error or "primary backend failed") + " -> scripted fallback"
            last = fb
        self._log(task, last, time.time() - t0)
        return last

    def _log(self, task: TaskClass, resp: AgentResponse, wall_s: float, applied: Optional[Dict[str, Any]] = None) -> None:
        rec = DecisionLog(task=task.value, spec=f"{resp.model}@{resp.effort}", response=resp.to_record(), applied=applied or {}, wall_s=wall_s)
        self.decisions.append(rec)
        self.pipeline.bus.publish("agent_decision", {"task": task.value, "model": resp.model, "effort": resp.effort, "ok": resp.ok, "attempts": resp.attempts}, self.pipeline.now(), "agent")
        if self.log_dir:
            with open(os.path.join(self.log_dir, "decisions.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(_to_jsonable(rec), ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------ 决策点 0：数据质量（确定性检查 + 必要时自适应解释）
    def check_data_quality(self, panel: Panel) -> D.DataQualityReport:
        issues: List[str] = []
        close = panel["close"]
        nan_ratio = float(close.iloc[-20:].isna().mean().mean()) if len(close) >= 1 else 1.0
        if nan_ratio > 0.05:
            issues.append(f"最近 20 日缺失率 {nan_ratio:.1%}")
        stale = close.iloc[-5:].nunique() if len(close) >= 5 else None
        if stale is not None:
            n_stale = int((stale <= 1).sum())
            if n_stale > 0:
                issues.append(f"{n_stale} 个标的最近 5 日价格不变（疑似停牌/陈旧）")
        rets = close.pct_change(fill_method=None).iloc[-1] if len(close) >= 2 else None
        if rets is not None:
            jumps = rets[rets.abs() > 0.3]
            if len(jumps):
                issues.append(f"{len(jumps)} 个标的最新日收益超过 ±30%：{list(jumps.index[:5])}")
        if not issues:
            rep = D.DataQualityReport(ok=True)
            return rep
        resp = self.decide(TaskClass.DATA_QUALITY, {"issues": issues, "n_assets": int(close.shape[1]), "n_days": int(close.shape[0])}, D.DataQualityReport)
        return resp.parsed if isinstance(resp.parsed, D.DataQualityReport) else D.DataQualityReport(ok=False, issues=issues)

    # ------------------------------------------------------------------ 决策点 1：研究评审（强推理）
    def research_review(self, reports: List[Any]) -> D.ResearchDecision:
        ctx = {"reports": [self._report_row(r) for r in reports], "currently_deployed": list(self.deployed_factors)}
        n = len(reports)
        resp = self.decide(TaskClass.RESEARCH_REVIEW, ctx, D.ResearchDecision, ComplexitySignals(n_entities=n, stakes=0.8, novelty=0.5))
        dec = resp.parsed if isinstance(resp.parsed, D.ResearchDecision) else D.ResearchDecision()
        known = {row["name"] for row in ctx["reports"]}
        dec.deploy_factors = [f for f in dec.deploy_factors if f in known]   # 只能部署评审过的因子
        self.deployed_factors = sorted(set(self.deployed_factors) - set(dec.retire_factors) | set(dec.deploy_factors))
        self.decisions[-1].applied = {"deployed": self.deployed_factors}
        return dec

    @staticmethod
    def _report_row(r) -> Dict[str, Any]:
        d = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        spec = d.get("spec", {}) or {}
        return {"name": spec.get("name", d.get("name", "?")), "expression": spec.get("expression", ""), "family": spec.get("family", ""), "ic_mean": d.get("ic_mean"), "ic_ir": d.get("ic_ir"),
                "turnover": d.get("turnover"), "coverage": d.get("coverage"), "long_short_sharpe": d.get("long_short_sharpe"), "decay_half_life": d.get("decay_half_life"), "sample": d.get("sample")}

    def research_campaign(self, panel: Panel, budget: int = 50) -> Dict[str, Any]:
        """因子假设（强推理）-> 挖掘 -> 评估 -> 研究评审（强推理）。需要因子挖掘库的 miner。"""
        if self.miner is None:
            return {"error": "未配置 FactorMiner（因子挖掘库不可用）"}
        hyp = self.decide(TaskClass.FACTOR_HYPOTHESIS, {"universe": int(panel["close"].shape[1]), "days": int(panel["close"].shape[0]), "deployed": self.deployed_factors}, D.ResearchDecision, ComplexitySignals(stakes=0.6, novelty=0.8))
        specs = self.miner.mine(panel, budget=budget)
        reports = [self.miner.evaluate(s, panel) for s in specs]
        dec = self.research_review(reports)
        return {"hypotheses": getattr(hyp.parsed, "hypotheses", []), "n_mined": len(specs), "n_reports": len(reports), "decision": dec.model_dump()}

    # ------------------------------------------------------------------ 决策点 2：执行规划（自适应）
    def plan_execution(self, orders: List[ExecutionOrder]) -> Dict[str, Dict[str, Any]]:
        if not orders:
            return {}
        available = self.pipeline.execution_factory.available()
        market = self.pipeline.snapshots([o.symbol for o in orders])
        nav = max(self.pipeline.portfolio_snapshot().nav, 1.0)
        rows = []
        notional = 0.0
        for o in orders:
            snap = market.get(o.symbol)
            adv = snap.adv if snap and snap.adv else 1e6
            notional += o.quantity * (snap.mid if snap else o.arrival_price)
            rows.append({
                "order_id": o.order_id, "symbol": o.symbol, "side": o.side, "quantity": o.quantity, "pct_adv": round(o.quantity / adv, 4),
                "urgency": round(o.urgency, 3), "alpha_bps": round(o.alpha_bps, 2), "alpha_horizon_s": (o.alpha_horizon_us or 0) / US_PER_SEC,
                "window_s": (o.end_us - o.start_us) / US_PER_SEC, "max_participation": o.max_participation,
                "spread_bps": round(snap.spread_bps, 2) if snap else None, "volatility": round(snap.volatility, 4) if snap and snap.volatility else None,
                "displayed_far": snap.displayed_size(o.side) if snap else None,
                "impact_est_bps": json.loads(self.toolbox.estimate_impact(o.symbol, o.quantity, (o.end_us - o.start_us) / US_PER_SEC)).get("expected_cost_bps"),
            })
        ctx = {"orders": rows, "available": available, "venue_stats": [v.to_dict() for v in self.pipeline.router.venue_stats()], "risk_limits": self.pipeline.risk_gate.limits.to_dict()}
        sig = ComplexitySignals(n_entities=len(orders), stakes=min(notional / nav, 1.0), anomaly=float(np.mean([r["pct_adv"] > 0.05 for r in rows])))
        resp = self.decide(TaskClass.EXECUTION_PLAN, ctx, D.ExecutionPlan, sig, instruction="为每张母单选择执行策略与参数（strategy 必须来自 available）。")
        plan = resp.parsed if isinstance(resp.parsed, D.ExecutionPlan) else D.ExecutionPlan(plans=[])
        plans: Dict[str, Dict[str, Any]] = {}
        order_ids = {o.order_id for o in orders}
        for p in plan.plans:
            if p.order_id not in order_ids or p.strategy not in available:
                continue
            entry: Dict[str, Any] = {"strategy": p.strategy, "params": {k: float(v) for k, v in p.params.items() if isinstance(v, (int, float))}, "rationale": p.rationale}
            if p.urgency is not None:
                entry["urgency"] = float(min(max(p.urgency, 0.0), 1.0))
            plans[p.order_id] = entry
        self.decisions[-1].applied = {"n_planned": len(plans), "n_orders": len(orders)}
        return plans

    # ------------------------------------------------------------------ 决策点 3：交易后复盘（自适应 -> 必要时强推理）
    def post_trade_review(self, result) -> Dict[str, Any]:
        brief = result.brief()
        ctx = {"tca": brief["tca"], "venue_stats": brief["venue_stats"], "incidents": brief["incidents"], "event_counts": brief["event_counts"], "plans": result.plans, "limits": self.pipeline.risk_gate.limits.to_dict(), "rejected": result.rejected}
        w_is = float(brief["tca"].get("weighted_is_bps", 0.0) or 0.0)
        sig = ComplexitySignals(n_entities=len(result.orders), stakes=0.4, anomaly=min(abs(w_is) / 50.0 + (1.0 if brief["incidents"] else 0.0), 1.0))
        resp = self.decide(TaskClass.POST_TRADE_REVIEW, ctx, D.PostTradeReview, sig)
        review = resp.parsed if isinstance(resp.parsed, D.PostTradeReview) else D.PostTradeReview(summary="(无)")
        applied = self._apply_param_changes(review.param_changes)
        out: Dict[str, Any] = {"review": review.model_dump(), "applied_param_changes": applied}
        if review.escalate or brief["incidents"]:
            pm = self.decide(TaskClass.POST_MORTEM, {**ctx, "review": review.model_dump()}, D.IncidentResponse, ComplexitySignals(stakes=0.9, anomaly=1.0))
            inc = pm.parsed if isinstance(pm.parsed, D.IncidentResponse) else None
            if inc is not None:
                out["post_mortem"] = inc.model_dump()
                out["post_mortem_applied"] = self._apply_param_changes(inc.param_changes)
        self.decisions[-1].applied = applied
        return out

    def _apply_param_changes(self, changes: List[D.ParamChange]) -> List[Dict[str, Any]]:
        applied = []
        for c in changes:
            ok, msg = self.pipeline.risk_gate.set_limit(c.name, float(c.value))
            applied.append({"name": c.name, "value": c.value, "ok": ok, "message": msg, "rationale": c.rationale})
            self.pipeline.bus.publish("agent_param_change", {"name": c.name, "value": c.value, "ok": ok, "msg": msg}, self.pipeline.now(), "agent")
        return applied

    # ------------------------------------------------------------------ 决策点 4：事故响应（强推理）
    def handle_incident(self, incident: Incident) -> D.IncidentResponse:
        ctx = {"incidents": [incident.to_dict()], "tca": self.pipeline.tca.summary(), "venue_stats": [v.to_dict() for v in self.pipeline.router.venue_stats()], "portfolio": json.loads(self.toolbox.get_portfolio()), "recent_events": self.pipeline.bus.records(50)}
        resp = self.decide(TaskClass.RISK_INCIDENT, ctx, D.IncidentResponse, ComplexitySignals(stakes=1.0, anomaly=1.0))
        inc = resp.parsed if isinstance(resp.parsed, D.IncidentResponse) else D.IncidentResponse(diagnosis="(无)")
        self._apply_param_changes(inc.param_changes)
        # 恢复交易永远需要人工：agent 只给建议，不解除熔断
        return inc

    # ------------------------------------------------------------------ 日报（自适应）
    def daily_report(self, result, review: Dict[str, Any]) -> D.DailyReport:
        resp = self.decide(TaskClass.DAILY_REPORT, {"brief": result.brief(), "review": review.get("review"), "post_mortem": review.get("post_mortem")}, D.DailyReport)
        return resp.parsed if isinstance(resp.parsed, D.DailyReport) else D.DailyReport(headline="(无)", performance="", execution="", routing="", risks="")

    # ------------------------------------------------------------------ 每日周期
    def run_daily_cycle(self, panel: Panel, as_of=None, factor_reports: Optional[List[Any]] = None, session_us: Optional[int] = None) -> Dict[str, Any]:
        p = self.pipeline
        t0 = time.time()
        dq = self.check_data_quality(panel)
        out: Dict[str, Any] = {"data_quality": dq.model_dump()}
        if not dq.ok and any("停止" in a or "halt" in a.lower() for a in dq.actions):
            out["skipped"] = "数据质量不合格，今日不交易"
            return out
        pf_start = p.portfolio_snapshot()
        signals = p.generate_signals(panel, as_of)
        if factor_reports:
            out["research"] = self.research_review(factor_reports).model_dump()
        targets = p.construct_targets(signals)
        orders, rejected = p.create_orders(targets)
        plans = self.plan_execution(orders)
        actives = p.start_execution(orders, plans)
        p.run_until_done(session_us)
        from ..core.pipeline import CycleResult

        result = CycleResult(
            as_of=str(as_of) if as_of is not None else str(signals.ts_us), signals=signals, targets=targets, orders=orders, rejected=rejected,
            plans={a.order.order_id: {"strategy": a.strategy_name, "params": a.params, "urgency": a.order.urgency} for a in actives},
            portfolio_start=pf_start, portfolio_end=p.portfolio_snapshot(), tca_reports=[r for r in (p.tca.report(o.order_id) for o in orders) if r], tca_summary=p.tca.summary(),
            venue_stats=list(p.router.venue_stats()), incidents=list(p.risk_gate.incidents), event_counts=dict(p.bus.counts), duration_us=p.now() - pf_start.ts_us,
        )
        for inc in result.incidents:
            out.setdefault("incident_responses", []).append(self.handle_incident(inc).model_dump())
        review = self.post_trade_review(result)
        report = self.daily_report(result, review)
        out.update({"cycle": result.brief(), "plans": plans, "review": review, "report": report.model_dump(), "decisions": [{"task": d.task, "spec": d.spec, "ok": d.response.get("ok"), "attempts": d.response.get("attempts")} for d in self.decisions], "wall_s": round(time.time() - t0, 2)})
        if self.log_dir:
            with open(os.path.join(self.log_dir, "cycle_result.json"), "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(out), f, ensure_ascii=False, indent=1, default=str)
        return out
