"""单母单执行跟踪：观察某个执行策略在模拟市场里的节奏与成本。

用法：python examples/trace_execution.py [strategy] [execution_kind]
  strategy        twap / almgren_chriss / pov / ...（默认 twap）
  execution_kind  reference / order_placement / auto（默认 auto）
"""
import sys

sys.path.insert(0, __file__.rsplit("/examples/", 1)[0])
from quantstack.contracts import RiskLimits
from quantstack.core.registry import build_sim_pipeline
from quantstack.sim.market import make_synthetic_panel

strategy = sys.argv[1] if len(sys.argv) > 1 else "twap"
kind = sys.argv[2] if len(sys.argv) > 2 else "auto"
panel = make_synthetic_panel(300, 40, seed=0)
pipe = build_sim_pipeline(
    panel, kinds={"alpha": "reference", "execution": kind, "router": "reference"}, seed=0, initial_cash=5e6,
    limits=RiskLimits(max_order_notional=400_000, max_position_notional=600_000, max_daily_loss=200_000), portfolio_cfg={"top_k": 1, "gross_target": 0.3},
)
signals = pipe.generate_signals(panel)
targets = pipe.construct_targets(signals)
orders, _ = pipe.create_orders(targets)
ao = pipe.start_execution(orders[:1], plans={orders[0].order_id: {"strategy": strategy}})[0]
o = ao.order
print(f"母单 {o.order_id} {o.symbol} side={o.side} qty={o.quantity} urgency={o.urgency:.2f} alpha={o.alpha_bps:.1f}bps 窗口={(o.end_us - o.start_us) / 1e6:.0f}s 策略={ao.strategy_name}")
while not ao.done:
    pipe.step()
    t = pipe.now() / 1e6
    if t % 60 == 0 or ao.done:
        st = ao.strategy.state()
        print(f"t={t:6.0f}s  " + "  ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in st.items()))
r = pipe.tca.report(o.order_id)
print(f"完成: {o.status} {o.filled_quantity}/{o.quantity}  IS={r.implementation_shortfall_bps:.2f}bps  maker={r.maker_ratio:.0%}  切片={r.n_slices}  场所单={r.n_venue_orders}")
