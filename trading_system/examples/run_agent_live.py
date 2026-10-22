"""真实 Claude 模型驱动的每日周期。

    python examples/run_agent_live.py --dry-run   # 只打印每个决策点将使用的模型/effort 与请求参数，不联网
    python examples/run_agent_live.py             # 需要 ANTHROPIC_API_KEY 或 `ant auth login`

默认开启服务端 refusal fallback（betas=["server-side-fallback-2026-07-01"], fallbacks="default"）。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantstack.agent import ComplexitySignals, ModelPolicy, TaskClass, TradingAgent
from quantstack.contracts import RiskLimits
from quantstack.core.registry import build_sim_pipeline, make_impact_model
from quantstack.sim.market import make_synthetic_panel

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--kinds", default="auto")
args = ap.parse_args()

policy = ModelPolicy()
if args.dry_run:
    for task in (TaskClass.EXECUTION_PLAN, TaskClass.POST_TRADE_REVIEW, TaskClass.POST_MORTEM, TaskClass.RISK_INCIDENT, TaskClass.RESEARCH_REVIEW):
        spec = policy.resolve(task, ComplexitySignals(n_entities=10, stakes=0.6))
        print(f"{task.value:<20} -> {spec.label:<26} {json.dumps(spec.request_kwargs())}")
    sys.exit(0)

from quantstack.agent.backend import ClaudeBackend

panel = make_synthetic_panel(300, 40, seed=0)
pipe = build_sim_pipeline(panel, kinds={"alpha": args.kinds, "execution": args.kinds, "router": args.kinds}, seed=0, initial_cash=5e6,
                          limits=RiskLimits(max_order_notional=400_000, max_position_notional=600_000, max_daily_loss=200_000),
                          portfolio_cfg={"top_k": 5, "gross_target": 0.8}, alpha_cfg=({} if args.kinds == "reference" else {"mine_budget": 30}))
agent = TradingAgent(pipe, backend=ClaudeBackend(), policy=policy, impact_model=make_impact_model(args.kinds), log_dir="runs/live")
out = agent.run_daily_cycle(panel)
print(json.dumps(out["report"], ensure_ascii=False, indent=1))
print("决策:", [(d["task"], d["spec"], d["ok"]) for d in out["decisions"]])
