import json

import pytest

from quantstack.agent import ComplexitySignals, ModelPolicy, TaskClass, TradingAgent
from quantstack.agent.backend import AgentRequest, ScriptedBackend
from quantstack.agent import decisions as D
from quantstack.contracts import RiskLimits
from quantstack.core.registry import build_sim_pipeline
from quantstack.sim.market import make_synthetic_panel

REF = {"alpha": "reference", "execution": "reference", "router": "reference"}


def test_model_policy_routing():
    p = ModelPolicy()
    strong = p.resolve(TaskClass.POST_MORTEM)
    assert strong.model == "claude-fable-5-1" and strong.effort == "max"
    assert p.resolve(TaskClass.RISK_INCIDENT, ComplexitySignals(), attempt=3).effort == "max"
    easy = p.resolve(TaskClass.DAILY_REPORT, ComplexitySignals())
    assert easy.model == "claude-opus-5-5" and easy.effort == "low"
    hard = p.resolve(TaskClass.EXECUTION_PLAN, ComplexitySignals(n_entities=40, stakes=1.0, anomaly=1.0))
    assert hard.model == "claude-fable-5-1"
    # 失败升级阶梯单调
    labels = [p.resolve(TaskClass.EXECUTION_PLAN, ComplexitySignals(), attempt=a).label for a in range(6)]
    assert labels[0] == "claude-opus-5-5@low" and labels[-1] == "claude-fable-5-1@xhigh"
    kw = strong.request_kwargs()
    assert kw["output_config"] == {"effort": "max"} and "thinking" not in kw and "budget_tokens" not in json.dumps(kw)


def test_scripted_backend_execution_plan():
    be = ScriptedBackend()
    ctx = {"orders": [{"order_id": "O1", "urgency": 0.9, "pct_adv": 0.01, "max_participation": 0.1}, {"order_id": "O2", "urgency": 0.3, "pct_adv": 0.05}], "available": {"twap": "", "pov": "", "almgren_chriss": ""}}
    resp = be.run(AgentRequest(TaskClass.EXECUTION_PLAN, "sys", "user", context=ctx, output_model=D.ExecutionPlan), ModelPolicy().resolve(TaskClass.EXECUTION_PLAN))
    assert resp.ok and isinstance(resp.parsed, D.ExecutionPlan)
    by = {p.order_id: p.strategy for p in resp.parsed.plans}
    assert by == {"O1": "pov", "O2": "almgren_chriss"}


def test_agent_daily_cycle_offline():
    panel = make_synthetic_panel(200, 20, seed=7)
    pipe = build_sim_pipeline(panel, kinds=REF, seed=7, initial_cash=3e6, limits=RiskLimits(max_order_notional=300_000, max_position_notional=500_000), portfolio_cfg={"top_k": 3, "gross_target": 0.6})
    agent = TradingAgent(pipe)
    out = agent.run_daily_cycle(panel)
    assert out["data_quality"]["ok"]
    assert out["cycle"]["n_orders"] > 0 and out["cycle"]["tca"]["avg_fill_ratio"] == pytest.approx(1.0, abs=1e-6)
    tasks = [d["task"] for d in out["decisions"]]
    assert tasks[:3] == ["execution_plan", "post_trade_review", "daily_report"]
    assert all(v["strategy"] in pipe.execution_factory.available() for v in out["plans"].values())
    assert out["report"]["headline"]


def test_agent_incident_path_uses_strong_model():
    panel = make_synthetic_panel(200, 20, seed=8)
    pipe = build_sim_pipeline(panel, kinds=REF, seed=8, initial_cash=3e6, limits=RiskLimits(max_daily_loss=1.0), portfolio_cfg={"top_k": 3, "gross_target": 0.8})
    agent = TradingAgent(pipe)
    out = agent.run_daily_cycle(panel)
    assert out.get("incident_responses"), "熔断后应有事故响应"
    specs = {d["task"]: d["spec"] for d in out["decisions"]}
    assert specs["risk_incident"].endswith("@max") and "claude-fable-5-1" in specs["risk_incident"]
    assert pipe.risk_gate.limits.kill_switch, "agent 不能解除熔断"


def test_tools_are_bounded():
    panel = make_synthetic_panel(120, 10, seed=9)
    pipe = build_sim_pipeline(panel, kinds=REF, seed=9)
    from quantstack.agent.tools import ToolBox

    tb = ToolBox(pipe)
    assert json.loads(tb.set_risk_limit("max_participation", 0.9))["ok"] is False
    assert json.loads(tb.set_risk_limit("max_participation", 0.05))["ok"] is True
    assert "nav" in json.loads(tb.get_portfolio())
    assert json.loads(tb.halt_trading("test"))["ok"] and pipe.halted
