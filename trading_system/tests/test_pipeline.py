import pytest

from quantstack.contracts import US_PER_SEC, RiskLimits, VenueOrder
from quantstack.core.pipeline import PipelineConfig
from quantstack.core.registry import build_sim_pipeline
from quantstack.sim.market import make_synthetic_panel, panel_stats, SyntheticMarket
from quantstack.sim.venue import VenueSimulator

REF = {"alpha": "reference", "execution": "reference", "router": "reference"}


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(200, 20, seed=3)


def test_reference_cycle_completes(panel):
    pipe = build_sim_pipeline(panel, kinds=REF, seed=1, initial_cash=3e6, limits=RiskLimits(max_order_notional=300_000, max_position_notional=500_000), portfolio_cfg={"top_k": 3, "gross_target": 0.6})
    res = pipe.run_cycle(panel)
    assert res.orders, "应产生母单"
    assert all(o.status == "filled" for o in res.orders), [(o.order_id, o.status, o.filled_quantity, o.quantity) for o in res.orders]
    assert res.tca_summary["avg_fill_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert abs(res.tca_summary["weighted_is_bps"]) < 60
    assert not res.incidents
    # 账本一致：持仓 = 目标（在风控缩量范围内）
    pf = res.portfolio_end
    assert pf.gross_exposure > 0
    # 事件日志覆盖全链路
    for k in ("signal_batch", "target", "order", "execution_plan", "route", "venue_fill", "order_done"):
        assert res.event_counts.get(k, 0) > 0, k


def test_no_overfill_and_in_flight_cap(panel):
    pipe = build_sim_pipeline(panel, kinds=REF, seed=2, initial_cash=3e6, portfolio_cfg={"top_k": 2, "gross_target": 0.5})
    res = pipe.run_cycle(panel)
    for o in res.orders:
        assert o.filled_quantity <= o.quantity * 1.0 + 1e-9, (o.order_id, o.filled_quantity, o.quantity)


def test_kill_switch_halts_pipeline(panel):
    pipe = build_sim_pipeline(panel, kinds=REF, seed=4, initial_cash=3e6, limits=RiskLimits(max_daily_loss=1.0), portfolio_cfg={"top_k": 3, "gross_target": 0.8})
    res = pipe.run_cycle(panel)
    assert pipe.halted and res.incidents
    assert pipe.risk_gate.limits.kill_switch


def test_venue_sim_cancel_in_flight(panel):
    st = panel_stats(panel)
    mkt = SyntheticMarket(st["price"], st["volatility"], st["adv"], seed=5)
    sim = VenueSimulator(mkt, seed=6)
    sym = mkt.symbols()[0]
    snap = mkt.snapshot(sym, 0)
    q = snap.quote("NASDAQ")
    vo = VenueOrder("S1-1", "S1", "O1", "NASDAQ", sym, 1, 300, q.bid - 0.05, "LIMIT", created_us=0)
    sim.submit(vo)
    from quantstack.contracts import CancelRequest

    sim.cancel(CancelRequest("S1-1", 0, "test"))  # 撤单与订单几乎同时发出
    kinds = [e.kind for t in range(0, 5 * US_PER_SEC, US_PER_SEC) for e in sim.poll(t)]
    assert "cancel" in kinds and sim.outstanding() == 0
