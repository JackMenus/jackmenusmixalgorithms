"""适配器与端到端流水线测试。"""
import numpy as np

from crossquant import adapters
from crossquant.pipelines import pairs_trading, portfolio_risk, regime


class Snap:
    mid = 100.0; spread_bps = 5.0; adv = 2e6; volatility = 0.02; last_price = 100.0


def test_state_estimators_follow_protocol():
    rng = np.random.default_rng(0)
    kt = adapters.KalmanTrendEstimator(level_var=1e-3, slope_var=1e-4, obs_var=0.01)
    ts = 0
    for i in range(200):
        ts += 1_000_000
        kt.update(100 + 0.05 * i + 0.1 * rng.standard_normal(), ts)
    assert abs(kt.predict(10_000_000) - 110.45) < 0.3
    ou = adapters.OUSpreadEstimator(0.99); x = 0.0
    for i in range(500):
        x = 0.95 * x + 0.1 * rng.standard_normal(); ou.update(x, (i + 1) * 1_000_000)
    assert 0 < ou.params["kappa"] < 0.3 and abs(ou.predict(100_000_000)) < abs(x) + 0.05


def test_controllers_and_impact_models():
    pc = adapters.PIDParticipationController()
    assert pc.step(0.1, 1_000_000) > 0
    lq = adapters.LQRTrackingController()
    assert lq.step(0.1, 1_000_000) > 0 and lq.step(-0.1, 1_000_000) < 0
    sq = adapters.SquareRootImpactModel().expected_cost_bps(100_000, Snap(), 3_600_000_000)
    ac = adapters.AlmgrenChrissImpactModel().expected_cost_bps(100_000, Snap(), 3_600_000_000)
    assert sq > 2.5 and ac > 2.5  # 至少半价差
    assert adapters.SquareRootImpactModel().expected_cost_bps(400_000, Snap(), 3_600_000_000) > sq
    reg = adapters.registry_entries()
    assert set(reg) == {"StateEstimator", "Controller", "ImpactModel"}


def test_pairs_pipeline():
    rng = np.random.default_rng(0)
    px = 100 + np.cumsum(rng.normal(0, 1, 1500)); spread = np.zeros(1500)
    for t in range(1, 1500):
        spread[t] = 0.9 * spread[t - 1] + 0.5 * rng.normal()
    py = 1.5 * px + 10 + spread
    pp = pairs_trading.pairs_trading_pipeline(py, px, use_kalman=True, mc_paths=1000, seed=0)
    assert abs(pp["hedge"]["beta"] - 1.5) < 0.05 and pp["adf"]["reject"]["5%"]
    assert 3 < pp["ou"]["half_life"] < 12 and pp["backtest"]["n_trades"] > 0
    assert "kalman" in pp and abs(pp["kalman"]["beta"][-1] - 1.5) < 0.1
    R = rng.normal(0, 0.01, (400, 15)) + np.outer(rng.normal(0, 0.01, 400), rng.normal(1, 0.3, 15))
    ep = pairs_trading.eigenportfolio_residuals(R, 2)
    assert ep["s_score"].shape == (15,) and ep["factor_returns"].shape == (400, 2)


def test_portfolio_and_regime_pipelines():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (500, 20)) + np.outer(rng.normal(0, 0.01, 500), rng.normal(1, 0.3, 20))
    ac = portfolio_risk.allocation_comparison(R[:350], R[350:])
    assert len(ac) == 16 and all(np.isclose(v["weights"].sum(), 1) for v in ac.values())
    r = np.concatenate([rng.normal(0, 0.01, 500), rng.normal(0, 0.03, 300)])
    rd = regime.regime_dashboard(r, window=120, step=10)
    assert rd["wasserstein"][-1] > rd["wasserstein"][0]
    after = (rd["index"] > 500) & (rd["index"] < 560)
    assert rd["bocpd_run_length"][after].min() < 40  # 变点后 run length 归零重计
