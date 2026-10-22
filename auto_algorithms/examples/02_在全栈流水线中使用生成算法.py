"""示例 2：把生成的算法接入 quantstack 全栈流水线（合成市场 + 撮合模拟器），跑一个完整周期并打印 TCA 与各层描述。

运行：python examples/02_在全栈流水线中使用生成算法.py
若 generated/ 为空，则退回模板族默认参数（autoalgo.family.*）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import autoalgo  # noqa: E402

autoalgo.ensure_siblings()

from quantstack.contracts import RiskLimits  # noqa: E402
from quantstack.core.registry import build_sim_pipeline, plugin_registry  # noqa: E402
from quantstack.sim.market import make_synthetic_panel  # noqa: E402

from autoalgo import registry  # noqa: E402

if __name__ == "__main__":
    kinds = registry.available_kinds()
    pick = lambda proto, fam: next((k for k in kinds.get(proto, []) if not k.startswith("autoalgo.family.")), f"autoalgo.family.{fam}")
    chosen = {"alpha": pick("AlphaModel", "alpha.regime_gated"), "portfolio": pick("PortfolioConstructor", "portfolio.denoised_riskparity"),
              "execution": pick("ExecutionStrategy", "execution.kalman_pid"), "selector": pick("StrategySelector", "selector.bandit"), "router": pick("Router", "router.bayesian_venue")}
    print("使用的 kind：", chosen)
    print("注册表中的 autoalgo 条目：", {p: [k for k in v if k.startswith("autoalgo.")] for p, v in plugin_registry().items() if any(k.startswith("autoalgo.") for k in v)})
    panel = make_synthetic_panel(300, 30, seed=5)
    pipe = build_sim_pipeline(panel, kinds=chosen, seed=5, initial_cash=5e6, limits=RiskLimits(max_order_notional=400_000, max_position_notional=600_000),
                              portfolio_cfg={"top_k": 5, "gross_target": 0.8})
    res = pipe.run_cycle(panel)
    print(f"\n母单 {len(res.orders)} 张，TCA 汇总：{res.tca_summary}")
    for r in res.tca_reports[:5]:
        print(f"  {r.parent_id} {r.symbol} side={r.side:+d} qty={r.quantity} IS={r.implementation_shortfall_bps:+.1f}bps fill={r.fill_ratio:.2f} 策略={r.strategy}")
    print("\n因子层：", {k: v for k, v in pipe.alpha_model.describe().items() if k in ("model_id", "family", "ic", "last_state", "in_sample_rank_ic")})
    print("组合层：", {k: v for k, v in pipe.portfolio.describe().items() if k in ("family", "denoise", "allocator", "tilt", "last_diagnostics")})
    print("选择器：", {k: v for k, v in pipe.strategy_selector.describe().items() if k in ("n_select", "n_update", "n_per_arm")})
    print("路由层：", {k: v for k, v in pipe.router.describe().items() if k in ("n_routes", "n_rebalanced", "n_rewards", "posterior_mean_reward_bps")})
