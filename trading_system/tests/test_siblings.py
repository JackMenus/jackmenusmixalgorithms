"""与兄弟库的联调测试：库不可用时跳过。"""
import pytest

from quantstack.contracts import AlphaModel, ExecutionStrategyFactory, RiskLimits, Router
from quantstack.core.registry import build_sim_pipeline, discover, make_execution_factory, make_router, make_alpha_model
from quantstack.sim.market import make_synthetic_panel

STATUS = discover()


@pytest.mark.skipif(not STATUS.get("latency_routing"), reason="latency_routing 不可用")
def test_latency_routing_adapter_roundtrip():
    r = make_router("latency_routing", ["NYSE", "NASDAQ", "ARCA", "IEX"])
    assert isinstance(r, Router)
    from quantstack.contracts import MarketSnapshot, OrderSlice, VenueQuote

    snap = MarketSnapshot("AAA", 0, [VenueQuote(v, "AAA", 99.99, 100.01, 300, 300, ts_us=0, taker_fee=0.003, maker_fee=-0.002) for v in ["NYSE", "NASDAQ", "ARCA", "IEX"]], last_price=100.0, adv=1e6, volatility=0.02)
    dec = r.route(OrderSlice("O1-1", "O1", "AAA", 1, 500, "IOC", limit_price=100.01, urgency=0.8), snap)
    assert dec.venue_orders and all(vo.venue_order_id.startswith("O1-1-") for vo in dec.venue_orders)
    assert r.venue_stats()


@pytest.mark.skipif(not STATUS.get("order_placement"), reason="order_placement 不可用")
def test_order_placement_factory():
    f = make_execution_factory("order_placement")
    assert isinstance(f, ExecutionStrategyFactory)
    assert {"twap", "pov", "almgren_chriss"} <= set(f.available())


@pytest.mark.skipif(not STATUS.get("factormine"), reason="factormine 不可用")
def test_factormine_alpha_model():
    m = make_alpha_model("factormine", mine_budget=5, seed=0)
    assert isinstance(m, AlphaModel)


@pytest.mark.skipif(not all(STATUS.get(k) for k in ("factormine", "order_placement", "latency_routing")), reason="三库未齐")
def test_full_stack_cycle_with_sibling_libraries():
    panel = make_synthetic_panel(200, 20, seed=11)
    pipe = build_sim_pipeline(panel, kinds={"alpha": "auto", "execution": "auto", "router": "auto"}, seed=11, initial_cash=3e6,
                              limits=RiskLimits(max_order_notional=300_000, max_position_notional=500_000), portfolio_cfg={"top_k": 3, "gross_target": 0.6}, alpha_cfg={"mine_budget": 5, "seed": 0})
    res = pipe.run_cycle(panel)
    assert res.orders and all(o.filled_quantity > 0 for o in res.orders)
    assert all(o.filled_quantity <= o.quantity for o in res.orders)
    assert res.tca_summary["avg_fill_ratio"] > 0.9


# ----------------------------------------------------------------------------- 跨学科库 crossquant
CROSSQUANT_OK = discover(["crossquant"]).get("crossquant")


@pytest.mark.skipif(not CROSSQUANT_OK, reason="crossquant 不可用（需要 numpy + scipy）")
def test_crossquant_hooks_satisfy_protocols():
    from quantstack.contracts import Controller, ImpactModel, MarketSnapshot, StateEstimator, VenueQuote
    from quantstack.core.registry import crossquant_entries, make_controller, make_impact_model, make_state_estimator

    entries = crossquant_entries()
    assert {"StateEstimator", "Controller", "ImpactModel"} <= set(entries)
    protos = {"StateEstimator": StateEstimator, "Controller": Controller, "ImpactModel": ImpactModel}
    for proto, cls_map in entries.items():
        for name, cls in cls_map.items():
            assert name.startswith("crossquant.") and isinstance(cls(), protos[proto])

    est = make_state_estimator("crossquant.KalmanTrendEstimator")
    for i in range(30):
        est.update(100.0 + 0.01 * i, i * 1_000_000)          # 每秒 +0.01 的线性趋势
    assert abs(est.predict(1_000_000) - 100.30) < 0.1        # 外推到第 30 秒
    assert make_state_estimator("auto") is not None and make_controller("auto") is not None

    ctrl = make_controller("crossquant.PIDParticipationController", kp=1.0, ki=0.0)
    assert ctrl.step(0.1, 1_000_000) > 0 > ctrl.step(-0.1, 1_000_000)   # 落后 → 提速，超前 → 减速
    lqr = make_controller("crossquant.LQRTrackingController")
    assert lqr.step(0.1, 1_000_000) > 0

    snap = MarketSnapshot("AAA", 0, [VenueQuote("NYSE", "AAA", 99.99, 100.01, 300, 300, ts_us=0, taker_fee=0.003, maker_fee=-0.002)], last_price=100.0, adv=1e6, volatility=0.02)
    for kind in ("crossquant", "crossquant.SquareRootImpactModel", "crossquant.AlmgrenChrissImpactModel"):
        m = make_impact_model(kind)
        assert isinstance(m, ImpactModel)
        small, big = m.expected_cost_bps(1_000, snap, 600_000_000), m.expected_cost_bps(100_000, snap, 600_000_000)
        assert 0 < small < big
    with pytest.raises(KeyError):
        make_impact_model("crossquant.NoSuchModel")


@pytest.mark.skipif(not CROSSQUANT_OK, reason="crossquant 不可用（需要 numpy + scipy）")
def test_crossquant_kalman_tracks_simulated_mid():
    from quantstack.core.registry import make_state_estimator

    panel = make_synthetic_panel(120, 5, seed=3)
    pipe = build_sim_pipeline(panel, kinds={"alpha": "reference", "execution": "reference", "router": "reference"}, seed=3)
    sym = list(panel["close"].columns)[0]
    est = make_state_estimator("auto")
    mids = []
    for k in range(1, 61):
        ts = k * 1_000_000
        pipe.feed.advance_to(ts)
        snap = pipe.feed.snapshot(sym, ts)
        mids.append(snap.mid)
        level = est.update(snap.mid, ts)
    assert abs(level - mids[-1]) / mids[-1] < 0.01            # 滤波水平贴近最新中间价
    assert abs(est.predict(5_000_000) - mids[-1]) / mids[-1] < 0.02


# ----------------------------------------------------------------------------- 自主生成算法库 autoalgo
AUTOALGO_OK = all(discover([k]).get(k) for k in ("autoalgo", "factormine", "order_placement", "latency_routing", "crossquant"))


@pytest.mark.skipif(not AUTOALGO_OK, reason="autoalgo 或其依赖的兄弟库不可用")
def test_autoalgo_kinds_wire_into_pipeline():
    from quantstack.contracts import PortfolioConstructor
    from quantstack.core.registry import make_portfolio, make_strategy_selector, plugin_registry

    reg = plugin_registry()
    assert any(k.startswith("autoalgo.") for k in reg["AlphaModel"]) and any(k.startswith("autoalgo.") for k in reg["Router"])
    assert "StrategySelector" in reg and "PortfolioConstructor" in reg
    assert isinstance(make_portfolio("autoalgo.family.portfolio.denoised_riskparity", top_k=5), PortfolioConstructor)
    assert callable(make_strategy_selector("autoalgo.family.selector.bandit"))

    panel = make_synthetic_panel(160, 12, seed=21)
    kinds = {"alpha": "autoalgo.family.alpha.signature", "execution": "autoalgo.family.execution.kalman_pid", "router": "autoalgo.family.router.bayesian_venue",
             "portfolio": "autoalgo.family.portfolio.denoised_riskparity", "selector": "autoalgo.family.selector.bandit"}
    pipe = build_sim_pipeline(panel, kinds=kinds, seed=21, initial_cash=2e6, limits=RiskLimits(max_order_notional=200_000, max_position_notional=300_000),
                              portfolio_cfg={"top_k": 3, "gross_target": 0.5})
    assert isinstance(pipe.alpha_model, AlphaModel) and isinstance(pipe.execution_factory, ExecutionStrategyFactory) and isinstance(pipe.router, Router)
    assert "kalman_pid" in pipe.execution_factory.available()
    res = pipe.run_cycle(panel)
    assert res.orders and all(o.filled_quantity <= o.quantity for o in res.orders)
    assert res.tca_summary["avg_fill_ratio"] > 0.8
    assert pipe.strategy_selector.describe()["n_select"] == len(res.orders)
