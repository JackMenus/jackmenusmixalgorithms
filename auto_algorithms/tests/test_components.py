"""组件级测试：算子对照 crossquant、各协议实现满足 quantstack 协议、快路径与逐日预测一致。"""
import numpy as np
import pandas as pd
import pytest

from conftest import STATUS, need

ALL = need("quantstack", "factormine", "crossquant", "order_placement", "latency_routing")


@need("factormine", "crossquant")
def test_operators_register_and_match_crossquant(panel):
    from crossquant.mathematics.fracdiff import frac_diff_ffd
    from crossquant.mathematics.signatures import levy_area
    from factormine import OPERATORS, Evaluator, parse

    from autoalgo.components import operators as ops

    names = ops.register_all()
    assert set(names) <= set(OPERATORS)
    x = panel["close"].iloc[:, 0].to_numpy()
    mine = ops.ts_fracdiff(panel["close"], 0.5).iloc[:, 0].to_numpy()
    ref = frac_diff_ffd(x, 0.5, threshold=1e-3)
    m = np.isfinite(mine) & np.isfinite(ref)
    assert m.sum() > 50 and np.allclose(mine[m], ref[m])
    A = panel["close"].iloc[-21:, 0].to_numpy()
    B = panel["volume"].iloc[-21:, 0].to_numpy()
    dA, dB = np.diff(A), np.diff(B)
    ref_area = levy_area(np.column_stack([A, B]))[0, 1] / (20 * dA.std() * dB.std())
    assert abs(ops.ts_levy_area(panel["close"], panel["volume"], 20).iloc[-1, 0] - ref_area) < 1e-9
    # 表达式里可直接使用
    f = Evaluator(panel)(parse("cs_rank(ts_kalman_resid(close, 0.1))"))
    assert f.shape == panel["close"].shape and f.notna().mean().mean() > 0.9
    pe = ops.ts_perm_entropy(panel["returns"], 40)
    assert ((pe.dropna() >= 0) & (pe.dropna() <= 1.0001)).all().all()


@need("crossquant")
def test_regime_detector_is_causal(panel):
    from autoalgo.components.regime import RegimeDetector, market_returns

    det = RegimeDetector(window=60, step=5)
    full = det.transform(market_returns(panel))
    cut = 200
    part = det.transform(market_returns({k: v.iloc[:cut] for k, v in panel.items()}))
    pd.testing.assert_series_equal(full["prob_change"].iloc[:cut], part["prob_change"], check_names=False)
    assert set(full["regime"].unique()) <= {"trend", "neutral", "meanrev"}


@ALL
@pytest.mark.parametrize("cls,kw", [("RegimeGatedAlpha", {}), ("SignatureAlpha", {}), ("CrossOperatorMinedAlpha", {"budget": 20})])
def test_alpha_models_satisfy_protocol_and_fast_path(panel, cls, kw):
    from quantstack.contracts import AlphaModel, SignalBatch

    from autoalgo.components import alpha as A
    from autoalgo.evaluate import _score_panel

    m = getattr(A, cls)(**kw)
    assert isinstance(m, AlphaModel)
    train = {k: v.iloc[:180] for k, v in panel.items()}
    m.fit(train)
    batch = m.predict(panel, as_of=panel["close"].index[200])
    assert isinstance(batch, SignalBatch) and len(batch.signals) > 0
    assert all(0.0 <= s.confidence <= 1.0 for s in batch.signals)
    dates = list(panel["close"].index[190:250:10])
    fast, slow = _score_panel(m, panel, dates, True), _score_panel(m, panel, dates, False)
    a, b = fast.to_numpy(float), slow.to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 0 and np.allclose(a[ok], b[ok], atol=1e-9)
    d = m.describe()
    assert d["library"] == "autoalgo" and "family" in d


@ALL
def test_signature_features_match_crossquant():
    from crossquant.mathematics.signatures import signature

    from autoalgo.components.alpha import signature_features

    P = np.random.default_rng(0).normal(size=(30, 2, 3)).cumsum(0)
    F = signature_features(P, 10, normalize=False)[-1, 1]
    ref = signature(P[-10:, 1, :], 2)
    assert np.allclose(F, np.concatenate([ref[0], ref[1].reshape(-1)]))


@ALL
def test_portfolio_constructor(panel):
    from quantstack.contracts import MarketSnapshot, PortfolioConstructor, PortfolioSnapshot, RiskLimits, VenueQuote
    from quantstack.reference.alpha import ReversalVolumeAlpha

    from autoalgo.components.portfolio import DenoisedRiskParityConstructor

    pc = DenoisedRiskParityConstructor(top_k=6, max_weight=0.1, kelly_fraction=0.5)
    assert isinstance(pc, PortfolioConstructor)
    t = panel["close"].index[220]
    pc.fit(panel, as_of=t)
    alpha = ReversalVolumeAlpha()
    alpha.fit({k: v.iloc[:200] for k, v in panel.items()})
    batch = alpha.predict(panel, as_of=t)
    px = panel["close"].loc[t]
    market = {s: MarketSnapshot(s, batch.ts_us, [VenueQuote("SIM", s, px[s] * 0.9999, px[s] * 1.0001, 1000, 1000)], adv=1e6, volatility=0.02) for s in px.index}
    targets = pc.construct(batch, PortfolioSnapshot(ts_us=batch.ts_us, cash=1e7), market, RiskLimits())
    assert targets
    gross = sum(abs(x.target_weight) for x in targets)
    assert 0.2 <= gross <= 1.0 + 1e-6 and max(abs(x.target_weight) for x in targets) <= 0.1 + 1e-9
    assert pc.describe()["allocator"] == "hrp"


@ALL
def test_execution_factory_and_strategy(small_panel):
    from quantstack.contracts import ExecutionStrategy, ExecutionStrategyFactory
    from quantstack.core.registry import build_sim_pipeline

    from autoalgo.components.execution import AutoExecutionFactory
    from autoalgo.evaluate import make_orders, _run_orders

    pipe = build_sim_pipeline(small_panel, kinds={"alpha": "reference", "execution": "order_placement", "router": "reference"}, seed=2)
    fac = AutoExecutionFactory(base=pipe.execution_factory)
    assert isinstance(fac, ExecutionStrategyFactory) and "kalman_pid" in fac.available() and "twap" in fac.available()
    pipe.execution_factory = fac
    orders = make_orders(pipe, 2, 2, duration_s=240)
    rows = _run_orders(pipe, orders, {o.order_id: {"strategy": "kalman_pid", "params": {"controller": "lqr"}} for o in orders})
    assert len(rows) == 2 and all(r["fill_ratio"] > 0.9 for r in rows)
    st = pipe.active[orders[0].order_id].strategy
    assert isinstance(st, ExecutionStrategy) and st.name == "kalman_pid" and "pred_move_bps" in st.state()


@ALL
def test_bandit_selector_learns_and_attaches(small_panel):
    from quantstack.core.registry import build_sim_pipeline

    from autoalgo.components.execution import AutoExecutionFactory
    from autoalgo.components.selector import BanditStrategySelector
    from autoalgo.evaluate import make_orders, _run_orders

    pipe = build_sim_pipeline(small_panel, kinds={"alpha": "reference", "execution": "order_placement", "router": "reference"}, seed=3)
    pipe.execution_factory = AutoExecutionFactory(base=pipe.execution_factory)
    sel = BanditStrategySelector(seed=0, warmup=1).attach(pipe)
    assert pipe.strategy_selector is sel
    orders = make_orders(pipe, 3, 3, duration_s=240)
    _run_orders(pipe, orders)
    d = sel.describe()
    assert d["n_select"] == 3 and d["n_update"] == 3 and sum(d["n_per_arm"].values()) == 3


@ALL
def test_bayesian_router_wraps_base(small_panel):
    from quantstack.contracts import Router
    from quantstack.core.registry import build_sim_pipeline

    from autoalgo.components.routing import BayesianVenueRouter
    from autoalgo.evaluate import make_orders, _run_orders

    pipe = build_sim_pipeline(small_panel, kinds={"alpha": "reference", "execution": "order_placement", "router": "reference"}, seed=4)
    venues = [v.name for v in pipe.feed.venues]
    pipe.router = BayesianVenueRouter(pipe.router, venues, seed=0, mix=0.8)
    assert isinstance(pipe.router, Router)
    orders = make_orders(pipe, 2, 4, duration_s=240, adv_frac=(0.005, 0.015))
    rows = _run_orders(pipe, orders, {o.order_id: {"strategy": "almgren_chriss"} for o in orders})
    assert rows and all(r["fill_ratio"] > 0.9 for r in rows)
    d = pipe.router.describe()
    assert d["n_routes"] > 0 and d["n_rewards"] > 0 and len(pipe.router.venue_stats()) == len(venues)
