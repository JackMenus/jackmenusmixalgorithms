import math

import numpy as np
import pytest

from order_placement import (
    AlphaSignal, AvellanedaStoikov, GLFT, AdaptiveGLFT, MarketMakerConfig, StrategyState,
    LatencyProfile, ParentOrder, Side, ScheduledExecutor, OrderPlacementEngine, OrderType,
    OrderPlacementStrategy, Fill,
)
from order_placement.execution import (
    ACParams, almgren_chriss, almgren_chriss_closed_form, ac_cost_moments, obizhaeva_wang,
    propagator_optimal, propagator_cost, exp_kernel, power_kernel, twap, vwap, pov,
)
from order_placement.microstructure import (
    imbalance, ofi_step, ofi_series, integrated_ofi, prob_up_diffusion, EmpiricalMicroprice,
    IntensityParams, fit_from_hit_counts, simulate_hits, kyle_lambda,
)
from order_placement.queue import (
    QueuePositionTracker, fill_probability, queue_value, VenueSpec, cont_kukanov_allocation,
)
from order_placement.simulation import make_book, synthetic_books, MarketMakingSim, ExecutionSim
from order_placement.rl import ExecutionEnv, ExecEnvConfig, QLearningExecutor


# ---------- 微结构 ----------
def test_imbalance_and_ofi():
    b0 = make_book("X", 0, 100.0, 0.01, 10, 10)
    b1 = make_book("X", 1, 100.0, 0.01, 15, 10)
    assert imbalance(b1) == pytest.approx(0.2)
    assert ofi_step(b0, b1) == pytest.approx(5.0)
    assert ofi_series([b0, b1, b0]).shape == (2, 1)


def test_prob_up_symmetry():
    assert prob_up_diffusion(10, 10) == pytest.approx(0.5)
    assert prob_up_diffusion(100, 1) > 0.95
    assert prob_up_diffusion(1, 100) < 0.05


def test_microprice_learns_direction():
    books = synthetic_books(20000, seed=1)
    mp = EmpiricalMicroprice().fit(books, horizon_steps=5)
    hi = make_book("SIM", 0, 100.0, 0.01, 30, 2)
    lo = make_book("SIM", 0, 100.0, 0.01, 2, 30)
    assert mp.adjustment(hi) > 0 > mp.adjustment(lo)


def test_intensity_fit_recovers_params():
    rng = np.random.default_rng(0)
    true = IntensityParams(A=2.0, k=30.0)
    d = np.linspace(0.0, 0.1, 11)
    hits, exp_ = simulate_hits(true, d, 5000, rng)
    est = fit_from_hit_counts(d, hits, exp_)
    assert est.A == pytest.approx(2.0, rel=0.1)
    assert est.k == pytest.approx(30.0, rel=0.1)


def test_integrated_ofi_and_kyle():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(500, 3))
    assert integrated_ofi(m).shape == (500,)
    x = rng.normal(size=1000)
    assert kyle_lambda(0.3 * x + 0.01 * rng.normal(size=1000), x) == pytest.approx(0.3, rel=0.05)


# ---------- 做市 ----------
def _book():
    return make_book("X", 0, 100.0, 0.01, 10, 10, spread_ticks=1)


def test_as_inventory_skew():
    cfg = MarketMakerConfig(gamma=0.1, sigma=0.05, k=20.0, max_inventory=100)
    mm = AvellanedaStoikov(cfg)
    flat = mm.quotes(_book(), StrategyState(inventory=0, horizon_end=60))
    long = mm.quotes(_book(), StrategyState(inventory=10, horizon_end=60))
    assert long[0] <= flat[0] and long[1] <= flat[1]  # 多头时整体下移报价


def test_glft_symmetry_and_skew():
    cfg = MarketMakerConfig(gamma=0.05, sigma=0.02, A=1.0, k=50.0)
    mm = GLFT(cfg)
    db0, da0 = mm._offsets(0, 0, cfg.sigma, 0)
    assert db0 == pytest.approx(da0)
    db, da = mm._offsets(3, 0, cfg.sigma, 0)
    assert db > db0 and da < da0
    dbm, dam = mm._offsets(0, 0, cfg.sigma, mu=0.001)  # 看涨 → 买价靠近、卖价远离
    assert dbm < db0 and dam > da0


def test_inventory_limit_and_post_only():
    cfg = MarketMakerConfig(max_inventory=2, order_qty=1)
    mm = GLFT(cfg)
    bid, ask = mm.quotes(_book(), StrategyState(inventory=2))
    assert bid is None and ask is not None
    b = _book()
    bid, ask = mm.quotes(b, StrategyState(inventory=0))
    assert bid < b.best_ask and ask > b.best_bid


def test_latency_widens():
    cfg = MarketMakerConfig(sigma=0.5, latency_z=3.0)
    mm = GLFT(cfg)
    b0, a0 = mm.quotes(_book(), StrategyState())
    b1, a1 = mm.quotes(_book(), StrategyState(), latency=LatencyProfile(one_way_ms=50))
    assert b1 <= b0 and a1 >= a0 and (a1 - b1) > (a0 - b0)


def test_mm_sim_runs_and_inventory_controlled():
    sim = MarketMakingSim(T=300, seed=3)
    r_glft = sim.run(GLFT(MarketMakerConfig(gamma=0.5, sigma=0.02, A=1.0, k=50.0, max_inventory=20)))
    r_low = sim.run(GLFT(MarketMakerConfig(gamma=0.01, sigma=0.02, A=1.0, k=50.0, max_inventory=20)))
    assert r_glft.n_fills > 0
    assert r_glft.stats["mean_abs_inventory"] <= r_low.stats["mean_abs_inventory"] + 1e-9


def test_alpha_improves_mm_pnl():
    kw = dict(T=2000, alpha_kappa=0.05, alpha_eta=0.002, sigma=0.01)
    pnls = []
    for w in (0.0, 1.0):
        tot = 0.0
        for seed in range(4):
            cfg = MarketMakerConfig(gamma=0.2, sigma=0.01, A=1.0, k=50.0, max_inventory=30, alpha_weight=w)
            tot += MarketMakingSim(seed=seed, **kw).run(GLFT(cfg)).stats["final_pnl"]
        pnls.append(tot)
    assert pnls[1] > pnls[0]


def test_adaptive_glft_runs():
    r = MarketMakingSim(T=200, seed=1).run(AdaptiveGLFT(MarketMakerConfig(max_inventory=10)))
    assert np.isfinite(r.stats["final_pnl"])


# ---------- 执行 ----------
P = ACParams(sigma=0.3, eta=0.05, gamma=0.001, risk_aversion=1e-3)


def test_ac_matches_closed_form():
    n1 = almgren_chriss(1000, 100, 20, P)
    n2 = almgren_chriss_closed_form(1000, 100, 20, P)
    assert n1.sum() == pytest.approx(1000)
    assert np.allclose(n1, n2, rtol=1e-6)
    assert n1[0] > n1[-1]  # 风险厌恶 → 前置


def test_ac_zero_risk_is_twap_and_drift():
    q = ACParams(sigma=0.3, eta=0.05, risk_aversion=0.0)
    assert np.allclose(almgren_chriss(1000, 100, 10, q), 100)
    up = ACParams(sigma=0.3, eta=0.05, risk_aversion=0.0, drift=0.01)
    n = almgren_chriss(1000, 100, 10, up, is_sell=True)  # 看涨时卖出应后置
    assert n[-1] > n[0]
    e, v = ac_cost_moments(n, 1000, 100, up)
    assert v > 0


def test_ow_and_propagator():
    n = obizhaeva_wang(1000, 10, 10, rho=0.5)
    assert n.sum() == pytest.approx(1000) and n[0] > n[5]
    t = np.linspace(0, 10, 11)
    opt = propagator_optimal(1000, t, exp_kernel(0.5))
    assert opt.sum() == pytest.approx(1000)
    assert propagator_cost(opt, t, exp_kernel(0.5)) <= propagator_cost(np.full(11, 1000 / 11), t, exp_kernel(0.5)) + 1e-9
    opt_p = propagator_optimal(1000, t, power_kernel())
    assert (opt_p >= 0).all() and opt_p.sum() == pytest.approx(1000)


def test_benchmarks():
    assert twap(100, 4).tolist() == [25] * 4
    assert vwap(100, [1, 3]).tolist() == [25, 75]
    n = pov(100, [100, 100, 100], 0.2)
    assert n.sum() == pytest.approx(100)


def test_executor_completes():
    parent = ParentOrder("SIM", Side.BUY, 500, 0.0, 100.0)
    ex = ScheduledExecutor(parent, twap(500, 10))
    res = ExecutionSim(seed=0).run(ex)
    assert res["completion"] == pytest.approx(1.0)
    assert abs(res["shortfall_bps"]) < 100


# ---------- 队列 ----------
def test_queue_tracker():
    q = QueuePositionTracker(queue_ahead=10, my_qty=5)
    assert q.on_trade(8) == 0 and q.ahead == 2
    q.on_level_change(new_level_qty=5, old_level_qty=10)
    assert q.ahead <= 2
    assert q.on_trade(10) == 5 and q.done


def test_fill_prob_and_decision():
    assert fill_probability(0, 1, 0) == 1.0
    assert fill_probability(10, 1, 1) < fill_probability(1, 1, 1)
    good = queue_value(spread=0.02, queue_ahead=1, depletion_rate=10, adverse_rate=0.1, adverse_selection=0.0)
    bad = queue_value(spread=0.02, queue_ahead=500, depletion_rate=1, adverse_rate=1, adverse_selection=0.0)
    assert good.use_limit and not bad.use_limit


def test_cont_kukanov():
    venues = [VenueSpec("A", 100, 0.002, 300), VenueSpec("B", 500, 0.003, 300)]
    M, L = cont_kukanov_allocation(1000, 0.005, 0.003, venues, lambda_under=0.05, lambda_over=0.05)
    assert 0 <= M <= 1000 and (L >= 0).all()
    assert M + L.sum() >= 500  # 欠成交惩罚大时应充分下单


# ---------- RL ----------
def test_rl_beats_or_matches_twap():
    cfg = ExecEnvConfig(drift_vol=0.05, sigma=0.02)
    agent = QLearningExecutor(cfg).train(ExecutionEnv(cfg, seed=0), episodes=8000)
    q = agent.evaluate(ExecutionEnv(cfg, seed=1), 1000, "q")
    tw = agent.evaluate(ExecutionEnv(cfg, seed=1), 1000, "twap")
    assert q["mean_cost_bps"] <= tw["mean_cost_bps"] + 0.5


# ---------- 集成 ----------
class _Router:
    def __init__(self):
        self.sent = []

    def route(self, orders):
        self.sent += orders
        return [o.order_id for o in orders]

    def latency_profile(self, venue=None):
        return LatencyProfile(one_way_ms=2.0)


class _Alpha:
    def latest(self, symbol, ts):
        return AlphaSignal(symbol, ts, value=3.0, horizon_s=10.0)


def test_engine_wiring():
    r = _Router()
    eng = OrderPlacementEngine(router=r, alpha_provider=_Alpha())
    mm = GLFT(MarketMakerConfig())
    assert isinstance(mm, OrderPlacementStrategy)
    eng.add_strategy("mm", "X", mm)
    parent = ParentOrder("X", Side.SELL, 100, 0.0, 10.0)
    eng.add_strategy("exe", "X", ScheduledExecutor(parent, twap(100, 5)))
    out = eng.on_book(_book())
    assert len(r.sent) == len(out) >= 3
    o = next(o for o in out if o.strategy == "glft" and o.side is Side.BUY)
    eng.on_fill(Fill(o.order_id, "X", Side.BUY, o.price, 1.0, 0.0))
    assert eng.states["mm"].inventory == 1.0
    assert out[0].to_dict()["side"] in ("BUY", "SELL")
