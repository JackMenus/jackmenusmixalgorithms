"""离线 agent 演示：ScriptedBackend（确定性基线）驱动一个完整的每日周期，并写出决策日志。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantstack.agent import TradingAgent
from quantstack.contracts import RiskLimits
from quantstack.core.registry import build_sim_pipeline, make_impact_model
from quantstack.sim.market import make_synthetic_panel

mode = sys.argv[1] if len(sys.argv) > 1 else "reference"
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", f"offline_{mode}")
panel = make_synthetic_panel(300, 40, seed=0)
pipe = build_sim_pipeline(panel, kinds={"alpha": mode, "execution": mode, "router": mode}, seed=0, initial_cash=5e6,
                          limits=RiskLimits(max_order_notional=400_000, max_position_notional=600_000, max_daily_loss=200_000),
                          portfolio_cfg={"top_k": 5, "gross_target": 0.8}, alpha_cfg=({} if mode == "reference" else {"mine_budget": 30}))
agent = TradingAgent(pipe, impact_model=make_impact_model(mode), log_dir=log_dir)
out = agent.run_daily_cycle(panel)
print(json.dumps(out["report"], ensure_ascii=False, indent=1))
print("决策:", [(d["task"], d["spec"]) for d in out["decisions"]])
print("日志目录:", os.path.abspath(log_dir))
