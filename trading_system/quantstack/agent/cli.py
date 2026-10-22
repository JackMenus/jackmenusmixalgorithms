"""命令行入口。

    python -m quantstack.agent.cli demo                 # 离线：ScriptedBackend + 参考实现/兄弟库自动发现
    python -m quantstack.agent.cli demo --llm           # 真实 Claude（需要 ANTHROPIC_API_KEY 或 ant auth login）
    python -m quantstack.agent.cli policy               # 打印模型路由表
    python -m quantstack.agent.cli plugins              # 列出四个兄弟库登记的全部实现
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="quantstack-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="跑一个完整的每日周期")
    d.add_argument("--llm", action="store_true", help="使用真实 Claude 模型（默认离线脚本后端）")
    d.add_argument("--kinds", default="auto", help="auto | reference")
    d.add_argument("--days", type=int, default=300)
    d.add_argument("--assets", type=int, default=40)
    d.add_argument("--top-k", type=int, default=5)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--log-dir", default=None)
    sub.add_parser("policy", help="打印模型路由表")
    sub.add_parser("plugins", help="列出可用实现（含跨学科库 crossquant 的估计器 / 控制器 / 冲击模型）")
    args = ap.parse_args(argv)

    if args.cmd == "policy":
        from .model_policy import ModelPolicy

        for row in ModelPolicy().table():
            print(f"{row['task']:<22} {row['tier']:<6} 起始 {row['start']:<24} 上限 {row['max']}")
        return 0

    if args.cmd == "plugins":
        from ..core.registry import plugin_registry

        for proto, impls in plugin_registry().items():
            print(proto)
            for name, src in impls.items():
                print(f"    {name:<42} {src}")
        return 0

    from ..contracts import RiskLimits
    from ..core.registry import build_sim_pipeline, discover, make_impact_model
    from ..sim.market import make_synthetic_panel
    from .backend import ScriptedBackend
    from .orchestrator import TradingAgent

    print("兄弟库可用性:", discover())
    panel = make_synthetic_panel(args.days, args.assets, seed=args.seed)
    kinds = {"alpha": args.kinds, "execution": args.kinds, "router": args.kinds}
    pipe = build_sim_pipeline(
        panel, kinds=kinds, seed=args.seed, initial_cash=5e6,
        limits=RiskLimits(max_order_notional=400_000, max_position_notional=600_000, max_daily_loss=200_000),
        portfolio_cfg={"top_k": args.top_k, "gross_target": 0.8},
        alpha_cfg=({} if args.kinds == "reference" else {"mine_budget": 30, "seed": args.seed}),
    )
    backend = None
    if args.llm:
        from .backend import ClaudeBackend

        backend = ClaudeBackend()
    agent = TradingAgent(pipe, backend=backend or ScriptedBackend(), impact_model=make_impact_model(args.kinds), log_dir=args.log_dir)
    out = agent.run_daily_cycle(panel)
    print(json.dumps({k: out[k] for k in ("data_quality", "plans", "review", "report", "decisions", "wall_s") if k in out}, ensure_ascii=False, indent=1, default=str)[:6000])
    c = out.get("cycle", {})
    print(f"\nNAV {c.get('nav_start')} -> {c.get('nav_end')} | 母单 {c.get('n_orders')} | IS {c.get('tca', {}).get('weighted_is_bps')} bps | 成交率 {c.get('tca', {}).get('avg_fill_ratio')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
