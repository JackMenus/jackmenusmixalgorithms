"""不经 agent，直接跑一次完整流水线周期（参考实现或自动接入三库）。

    python examples/run_end_to_end.py            # auto：优先兄弟库
    python examples/run_end_to_end.py reference  # 纯参考实现
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantstack.contracts import RiskLimits
from quantstack.core.registry import build_sim_pipeline, discover
from quantstack.sim.market import make_synthetic_panel

mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
print("兄弟库可用性:", discover())
panel = make_synthetic_panel(300, 40, seed=0)
pipe = build_sim_pipeline(
    panel,
    kinds={"alpha": mode, "execution": mode, "router": mode},
    seed=0,
    initial_cash=5e6,
    limits=RiskLimits(max_order_notional=400_000, max_position_notional=600_000, max_daily_loss=200_000),
    portfolio_cfg={"top_k": 5, "gross_target": 0.8},
    alpha_cfg=({} if mode == "reference" else {"mine_budget": 30, "seed": 0}),
)
res = pipe.run_cycle(panel)
brief = res.brief()
print(json.dumps({k: brief[k] for k in ("n_signals", "n_targets", "n_orders", "n_rejected", "plans", "nav_start", "nav_end", "duration_s")}, ensure_ascii=False, indent=1))
print("TCA:", json.dumps(brief["tca"], ensure_ascii=False))
for o in res.orders:
    r = pipe.tca.report(o.order_id)
    print(f"{o.order_id} {o.symbol} side={o.side:+d} qty={o.quantity} filled={o.filled_quantity} {o.status} {o.strategy_hint} IS={r.implementation_shortfall_bps:+.1f}bps maker={r.maker_ratio:.0%}")
