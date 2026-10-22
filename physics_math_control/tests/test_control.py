"""控制论模块测试：用闭式解 / 已知结果校验。"""
import numpy as np
import pytest

from crossquant.control import bandits, changepoint, execution, hjb, kalman, lqr, market_making, mpc, optimal_stopping, particle_filter, pid, rls


def test_kalman_hedge_ratio_tracks_beta():
    rng = np.random.default_rng(0)
    T = 2000
    x = np.cumsum(rng.standard_normal(T)) + 100
    beta = np.linspace(1.0, 1.5, T)
    y = beta * x + 5 + 0.5 * rng.standard_normal(T)
    out = kalman.dynamic_hedge_ratio(y, x, delta=1e-4, r=0.25)
    assert abs(out["beta"][-1] - 1.5) < 0.15
    assert np.isfinite(out["loglik"])
    assert out["zscore"].shape == (T,)


def test_kalman_smoother_reduces_error():
    rng = np.random.default_rng(1)
    level = np.cumsum(0.1 * rng.standard_normal(500))
    z = level + rng.standard_normal(500)
    out = kalman.local_level_filter(z, q_ratio=0.01)
    assert np.mean((out["smooth"] - level) ** 2) < np.mean((out["level"] - level) ** 2) < np.mean((z - level) ** 2)


def test_sv_qml_recovers_parameters():
    rng = np.random.default_rng(2)
    h = np.zeros(3000); mu, phi, se = -1.0, 0.95, 0.3
    for t in range(1, 3000):
        h[t] = mu + phi * (h[t - 1] - mu) + se * rng.standard_normal()
    r = np.exp(h / 2) * rng.standard_normal(3000)
    q = kalman.sv_qml_filter(r)
    assert abs(q["params"]["phi"] - phi) < 0.05
    assert abs(q["params"]["sigma_eta"] - se) < 0.1
    pf = particle_filter.sv_particle_filter(r, mu, phi, se, 400, seed=1)
    assert np.corrcoef(pf["h_filt"], h)[0, 1] > 0.7


def test_garleanu_pedersen_matches_closed_form():
    gamma, lam, phi = 2.0, 5.0, 0.1
    gp = lqr.GarleanuPedersen(Sigma=[[0.04]], Lambda=[[lam * 0.04]], B=[[0.01]], Phi=[[phi]], gamma=gamma, rho=0.0)
    a = lqr.gp_single_asset_rate(gamma, lam, 0.0)
    assert np.isclose(gp.A_xx[0, 0], a * 0.04, rtol=1e-6)
    assert np.isclose(gp.trade_matrix[0, 0], a / lam, rtol=1e-6)
    # ρ=0 时 aim = B f / (γΣ + aΣφ)（GP 2013 Prop. 4）
    assert np.isclose(gp.aim([1.0])[0], 0.01 / (gamma * 0.04 + a * 0.04 * phi), rtol=1e-6)
    # 多资产：交易矩阵特征值在 (0,1)
    Sig = np.array([[0.04, 0.01], [0.01, 0.09]])
    gp2 = lqr.GarleanuPedersen(Sig, 3 * Sig, np.eye(2) * 0.01, np.diag([0.1, 0.3]), gamma=2.0, rho=0.01)
    ev = np.linalg.eigvals(gp2.trade_matrix)
    assert np.all((ev.real > 0) & (ev.real < 1))


def test_lqr_and_execution_tracking():
    out = lqr.lqr_discrete([[1.0]], [[1.0]], [[1.0]], [[1.0]])
    assert abs(out["closed_loop_eigs"][0]) < 1
    tr = lqr.execution_tracking_lqr(T=10, risk_aversion=1.0, cost=1.0)
    assert tr["trajectory"][0] == 1.0 and abs(tr["trajectory"][-1]) < 0.05
    assert np.all(np.diff(tr["trajectory"]) <= 1e-9)


def test_mpc_single_period_is_mean_variance():
    Sig = np.array([[0.04, 0.01], [0.01, 0.09]])
    m = mpc.MultiPeriodMPC(Sig, gamma=1.0)
    res = m.plan([0.5, 0.5], [[0.05, 0.02]])
    assert res["success"] and np.isclose(res["next_weights"].sum(), 1.0)
    assert res["next_weights"][0] > 0.8
    costly = mpc.MultiPeriodMPC(Sig, gamma=1.0, tc_linear=0.05).plan([0.5, 0.5], [[0.05, 0.02]])
    assert abs(costly["trade"]).sum() < abs(res["trade"]).sum()


def test_almgren_chriss_and_transient_schedules():
    ac = execution.almgren_chriss(1e6, 5, 5, sigma=0.95, eta=2.5e-6, lam=1e-6, gamma=2.5e-7, epsilon=0.0625)
    assert np.isclose(ac["kappa"], 0.609, atol=0.01)  # AC 2000 原文示例 κ≈0.6
    assert np.isclose(ac["holdings"][0], 1e6) and np.isclose(ac["holdings"][-1], 0)
    twap = execution.almgren_chriss(1e6, 5, 5, 0.95, 2.5e-6, lam=0.0)
    assert np.allclose(twap["trades"], 2e5)
    ow = execution.obizhaeva_wang(1.0, 1.0, 10, rho=2.0)
    gss = execution.optimal_schedule_transient(ow["times"], lambda d: np.exp(-2.0 * d), 1.0)
    assert np.allclose(ow["trades"], gss["trades"], atol=1e-8)
    assert np.isclose(ow["trades"].sum(), 1.0)
    assert execution.square_root_impact(1e5, 1e6, 0.02) == pytest.approx(0.5 * 0.02 * np.sqrt(0.1))


def test_glft_exact_converges_to_asymptotic_and_closed_form():
    ex = market_making.glft_exact_quotes(gamma=0.1, sigma=2.0, k=1.5, A=140, Q=5, T=600, t=0)
    asy = market_making.glft_asymptotic_quotes(0.1, 2.0, 1.5, 140, 5)
    cf = market_making.glft_closed_form_quotes(asy["q"], 0.1, 2.0, 1.5, 140)
    assert np.allclose(ex["delta_b"][:-1], asy["delta_b"][:-1], atol=1e-3)
    assert abs(asy["delta_b"][5] - cf["delta_b"][5]) < 0.05  # q=0 处闭式近似
    sim = market_making.simulate_market_making(market_making.as_strategy(0.1, 2.0, 1.5, 1.0), seed=3)
    assert len(sim["pnl"]) == 201 and np.isfinite(sim["final_pnl"])


def test_hjb_recovers_merton_ratio():
    m = hjb.merton_hjb_numeric(mu=0.08, r=0.02, sigma=0.2, gamma=3.0, T=1.0)
    interior = m["policy_t0"][30:91]
    assert np.allclose(interior, m["pi_closed"], atol=0.03)


def test_changepoint_detectors():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(1.5, 1, 200), rng.normal(0, 3, 200)])
    b = changepoint.bocpd(x, hazard=1 / 200, beta0=1.0)
    assert 300 <= np.argmax(b["prob_change"][290:330]) + 290 <= 315
    assert b["prob_change"][500:510].max() > 0.5
    pc = changepoint.page_cusum(x, 0, 1.0, 1.0, h=5)
    assert 300 <= pc["alarms"][0] <= 320
    cf = changepoint.cusum_filter(np.cumsum(rng.normal(0, 1, 1000)), threshold=5)
    assert 20 < len(cf["events"]) < 200


def test_bandits_low_regret():
    r = bandits.venue_fill_reward([0.3, 0.5, 0.7])
    s = bandits.simulate_bandit(bandits.ThompsonBernoulli(3, seed=0), r, 3000, seed=0)
    assert s["regret"][-1] < 60
    s2 = bandits.simulate_bandit(bandits.UCB1(3), r, 3000, seed=0)
    assert s2["regret"][-1] < 150
    lin = bandits.LinUCB(2, 2, alpha=0.5)
    ctx = np.random.default_rng(0).normal(size=(500, 2))

    def rew(arm, rng, c):
        return float(c[arm] + 0.1 * rng.standard_normal())
    out = bandits.simulate_bandit(lin, rew, 500, seed=0, contexts=ctx)
    assert np.mean(out["arms"][-200:] == np.argmax(ctx[-200:], axis=1)) > 0.8


def test_optimal_stopping():
    lsm = optimal_stopping.american_option_lsm(36, 40, 0.06, 0.2, 1.0, n_steps=50, n_paths=30000, seed=0)
    crr = optimal_stopping.crr_binomial_american(36, 40, 0.06, 0.2, 1.0, 800)
    assert abs(lsm["price"] - crr) < 0.06
    oe = optimal_stopping.ou_optimal_entry_exit(theta=0.0, kappa=5.0, sigma=0.5, r=0.05, cost=0.01, dt=1 / 252, n_grid=201)
    assert oe["exit_threshold"] > 0 > oe["entry_threshold"]
    assert oe["exit_threshold"] < 6 * 0.5 / np.sqrt(10)


def test_pid_and_rls():
    ctrl = pid.PID(2.0, 0.5, 0.1, out_min=-1, out_max=1)
    sim = pid.simulate_execution_progress(ctrl, np.linspace(0, 1, 100), 1.0, dt=0.01, base_rate=0.8, noise_std=0.1, seed=0)
    assert sim["executed"][-1] > 0.9 and np.abs(sim["error"]).max() < 0.15
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2)); y = X @ np.array([1.0, -2.0]) + 0.1 * rng.normal(size=500)
    ar = rls.adaptive_regression(X, y, 0.98)
    assert np.allclose(ar["theta_path"][-1], [1.0, -2.0], atol=0.1)
