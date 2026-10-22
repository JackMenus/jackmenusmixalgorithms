"""数学模块测试。"""
import numpy as np
import pytest

from crossquant.mathematics import (copulas, dynamical_systems, extreme_value, fracdiff, information_geometry, kelly, koopman, manifold,
                                    optimal_transport, queueing, sde, signatures, sparse, spectral, tda, timeseries, wavelets)
from crossquant.physics.pde import black_scholes_price


def test_optimal_transport():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 3000); b = rng.normal(1, 1, 3000)
    assert abs(optimal_transport.wasserstein_1d(a, b) - 1.0) < 0.1
    assert abs(optimal_transport.wasserstein_1d(a, b, wx=np.ones(3000), wy=np.ones(3000)) - 1.0) < 0.1
    C = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]).astype(float)
    sk = optimal_transport.sinkhorn(np.ones(5), np.ones(5), C, eps=0.01)
    assert sk["cost"] < 1e-6 and np.allclose(sk["plan"].sum(1), 0.2)
    assert np.isclose(optimal_transport.gaussian_w2([0, 0], np.eye(2), [1, 0], np.eye(2)), 1.0)
    Sig = np.array([[0.04, 0.01], [0.01, 0.09]])
    w0 = optimal_transport.wasserstein_dro_mean_variance([0.05, 0.08], Sig, 0.0)["weights"]
    w1 = optimal_transport.wasserstein_dro_mean_variance([0.05, 0.08], Sig, 0.05)["weights"]
    assert np.sum(w1 ** 2) < np.sum(w0 ** 2)  # 鲁棒化 ⇒ 更分散
    assert optimal_transport.mmd(a[:300], b[:300]) > 10 * abs(optimal_transport.mmd(a[:300], a[300:600]))


def test_signatures_algebra():
    rng = np.random.default_rng(0)
    d = np.array([1.0, 2.0])
    s = signatures.signature(np.array([[0, 0], d]), 4)
    assert np.allclose(s[1], np.outer(d, d) / 2) and np.allclose(s[3], np.multiply.outer(np.multiply.outer(np.outer(d, d), d), d) / 24)
    p1 = rng.normal(size=(10, 2)).cumsum(0); p2 = p1[-1] + rng.normal(size=(8, 2)).cumsum(0)
    chen = signatures.tensor_product(signatures.signature(p1, 4), signatures.signature(np.vstack([p1[-1:], p2]), 4), 4)
    assert all(np.allclose(x, y) for x, y in zip(signatures.signature(np.vstack([p1, p2]), 4), chen))
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0.0]])
    assert np.isclose(signatures.levy_area(sq)[0, 1], 1.0)
    th = np.linspace(0, 2 * np.pi, 400); circ = np.stack([np.cos(th), np.sin(th)], 1)
    ls = signatures.log_signature(signatures.signature(circ, 3))
    assert np.allclose(ls[0], 0, atol=1e-10) and abs(ls[1][0, 1] - np.pi) < 0.01
    x = np.zeros(1500)
    for t in range(1, 1500):
        x[t] = 0.9 * x[t - 1] + rng.normal()
    st = signatures.SignatureTrading(window=10, depth=2, ridge=1e-3).fit(x[:1000])
    assert st.backtest(x[1000:])["sharpe"] > 0.05


def test_information_geometry():
    assert np.isclose(information_geometry.fisher_rao_gaussian(0, 1, 0, 2), np.sqrt(2) * np.log(2))
    assert information_geometry.fisher_rao_gaussian(0, 1, 1, 1) > 0
    assert np.isclose(information_geometry.kl_divergence([0.5, 0.5], [0.5, 0.5]), 0, atol=1e-9)
    rng = np.random.default_rng(0)
    X = np.exp(rng.normal(0.0005, 0.02, (800, 4))); X[:, 0] = np.exp(rng.normal(0.003, 0.02, 800))
    eg = information_geometry.exponentiated_gradient(X, 0.05)
    up = information_geometry.universal_portfolio(X, 300, seed=0)
    uni = information_geometry.constant_rebalanced(X, np.full(4, 0.25))[-1]
    bcrp = information_geometry.best_crp_in_hindsight(X)["wealth"]
    assert eg["wealth"][-1] > uni and up["wealth"][-1] > uni and bcrp >= eg["wealth"][-1]


def test_sde_and_qmc():
    S = sde.simulate_gbm(100, 0.05, 0.2, 1.0, 100, 5000, seed=0)
    assert abs(S[:, -1].mean() - 100 * np.exp(0.05)) < 1.0
    assert sde.simulate_cir(0.04, 2.0, 0.04, 0.3, 1.0, 252, 200, seed=0).min() >= 0
    h = sde.simulate_heston(100, 0.04, 0.0, 2.0, 0.04, 0.3, -0.7, 1.0, 100, 2000, seed=0)
    assert abs(h["S"][:, -1].mean() - 100) < 1.5
    Zq = sde.qmc_normals(1024, 32, seed=0)
    assert abs(Zq.mean()) < 0.02 and abs(Zq.var() - 1) < 0.05
    bs = black_scholes_price(100, 100, 0.05, 0.2, 1.0, "call")
    Sq = sde.simulate_gbm(100, 0.05, 0.2, 1.0, 32, 1024, Z=Zq)
    assert abs(np.exp(-0.05) * np.maximum(Sq[:, -1] - 100, 0).mean() - bs) < 0.2
    Y = np.maximum(Sq[:, -1] - 100, 0) * np.exp(-0.05)
    cv = sde.control_variate(Y, Sq[:, -1], 100 * np.exp(0.05))
    assert cv["variance_reduction"] > 1


def test_fracdiff_and_adf():
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.normal(size=2000))
    assert not timeseries.adf_test(rw)["reject"]["5%"]
    assert timeseries.adf_test(rng.normal(size=2000))["reject"]["1%"]
    assert np.allclose(fracdiff.frac_diff_ffd(rw, 1.0)[1:], np.diff(rw))
    assert np.allclose(fracdiff.frac_diff_weights(0.5, size=4), [1, -0.5, -0.125, -0.0625])
    md = fracdiff.find_min_d(rw)
    assert md["d_min"] is not None and 0 < md["d_min"] < 1
    assert timeseries.ljung_box(rng.normal(size=1000))["p_value"] > 0.01


def test_koopman_dmd_ssa():
    A = np.array([[0.9, -0.3], [0.3, 0.9]]); X = np.zeros((2, 60)); X[:, 0] = [1, 0]
    for t in range(1, 60):
        X[:, t] = A @ X[:, t - 1]
    m = koopman.dmd(X)
    assert np.allclose(np.sort_complex(m["eigs"]), np.sort_complex(np.linalg.eigvals(A)))
    assert np.abs(koopman.dmd_predict(m, 59) - X).max() < 1e-8
    rng = np.random.default_rng(0)
    sig = np.sin(2 * np.pi * np.arange(600) / 50); noisy = sig + 0.5 * rng.normal(size=600)
    ss = koopman.ssa(noisy, 100, groups=[[0, 1]])
    assert np.corrcoef(ss["components"][0], sig)[0, 1] > 0.98
    assert np.abs(koopman.ssa_forecast(noisy[:550], 100, 2, 50) - sig[550:]).mean() < 0.15
    assert koopman.hankel_dmd(noisy, delay=30, rank=4)["predict"](5).shape == (5,)


def test_tda_circle_vs_cloud():
    rng = np.random.default_rng(0)
    th = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    circ = np.stack([np.cos(th), np.sin(th)], 1) + 0.02 * rng.normal(size=(30, 2))
    dg = tda.vietoris_rips_persistence(circ)
    pers = dg[1][:, 1] - dg[1][:, 0]
    assert pers.max() > 1.0 and len(dg[0]) == 30
    cloud = tda.vietoris_rips_persistence(rng.normal(size=(30, 2)))
    assert tda.landscape_norm(dg[1]) > 5 * tda.landscape_norm(cloud[1])
    ind = tda.tda_crash_indicator(rng.normal(size=(100, 4)), window=40, step=30)
    assert ind["L1"].shape == (3,) and np.all(ind["L1"] >= 0)


def test_extreme_value():
    rng = np.random.default_rng(0)
    par = rng.pareto(3.0, 20000) + 1
    assert abs(extreme_value.hill_estimator(par, 500)["alpha"] - 3.0) < 0.4
    tr = rng.standard_t(4, 5000) * 0.01
    pv = extreme_value.pot_var_es(tr)
    assert abs(pv["VaR"][0.99] - (-np.quantile(tr, 0.01))) < 0.005 and pv["ES"][0.99] > pv["VaR"][0.99]
    g = extreme_value.fit_gev_block_maxima(tr, 21, return_period=100)
    assert g["return_level"] > g["loc"]


def test_sparse_recovery():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (500, 30)); y = 0.5 * R[:, 3] + 0.3 * R[:, 7] + 0.2 * R[:, 15] + 0.0005 * rng.normal(size=500)
    assert sorted(sparse.omp(R, y, 3)["support"]) == [3, 7, 15]
    sit = sparse.sparse_index_tracking(R, y, 3)
    assert np.allclose(sit["weights"][[3, 7, 15]], [0.5, 0.3, 0.2], atol=0.03)
    assert np.sum(np.abs(sparse.lasso_ista(R, y, 0.001)) > 1e-4) == 3


def test_kelly():
    assert np.isclose(kelly.kelly_binary(0.6, 1.0), 0.2) and np.isclose(kelly.kelly_continuous(0.08, 0.2), 2.0)
    f = np.linspace(0, 4, 401)
    assert np.isclose(f[np.argmax(kelly.growth_rate(f, 0.08, 0.2))], 2.0, atol=0.01)
    assert np.isclose(kelly.drawdown_probability(0.5, 1.0), 0.5) and np.isclose(kelly.drawdown_probability(0.5, 0.5), 0.125)
    sg = kelly.simulate_growth(1.0, 0.08, 0.3, T=20, n_steps=500, n_paths=3000, seed=0)
    assert abs(sg["time_avg_growth"] - sg["theory_time_avg"]) < 0.01 and sg["ensemble_mean"] > sg["median"]


def test_spectral_carr_madan():
    cm = spectral.carr_madan_call(100, 0.05, 1.0, spectral.bs_char_fn(100, 0.05, 0.2, 1.0), strikes=[80, 100, 120])
    assert np.allclose(cm["prices"], black_scholes_price(100, np.array([80, 100, 120.0]), 0.05, 0.2, 1.0, "call"), atol=2e-3)
    hc = spectral.carr_madan_call(100, 0.05, 1.0, spectral.heston_char_fn(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7, 1.0), strikes=[100])
    assert 9.5 < hc["prices"][0] < 11.0
    x = np.sin(2 * np.pi * np.arange(1000) / 40)
    assert abs(spectral.dominant_cycles(x, 1)["periods"][0] - 40) < 1


def test_queueing():
    p = queueing.fill_probability_pure_death(5, 1.0, 0.1, 0.5)
    mc = queueing.queue_race_mc(5, 1.0, 0.1, 0.5, 20000, seed=0)
    assert abs(p - mc["p_fill"]) < 0.01
    assert np.isclose(queueing.fill_probability_pure_death(0, 1.0, 0.0, 1.0), 0.5)
    assert queueing.prob_mid_up(5, 2, 1.0, 1.0, 0.2, 2000, seed=0) > 0.6


def test_copulas():
    rho = np.array([[1, 0.6], [0.6, 1]])
    U = copulas.t_copula_sample(rho, 4, 3000, seed=0)
    tf = copulas.t_copula_fit(U)
    assert abs(tf["corr"][0, 1] - 0.6) < 0.05 and tf["nu"] <= 8
    assert np.isclose(copulas.tail_dependence_t(1.0, 4), 1.0, atol=1e-6)
    assert abs(copulas.gaussian_copula_fit(copulas.gaussian_copula_sample(rho, 3000, seed=0))["corr"][0, 1] - 0.6) < 0.05


def test_dynamical_systems():
    lm = dynamical_systems.logistic_map(4.0, 3000)
    lam = dynamical_systems.largest_lyapunov_rosenstein(lm, dim=2, tau=1, min_tsep=5, max_t=6)["lambda_max"]
    assert abs(lam - np.log(2)) < 0.1
    assert 0.7 < dynamical_systems.correlation_dimension(dynamical_systems.takens_embedding(lm, 2))["D"] < 1.3
    r = dynamical_systems.rqa(dynamical_systems.recurrence_matrix(dynamical_systems.takens_embedding(lm[:300], 2), 0.05))
    assert 0 < r["RR"] < 1


def test_manifold():
    rng = np.random.default_rng(0)
    blobs = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(4, 0.3, (30, 2))])
    lab = manifold.spectral_clustering(manifold.gaussian_affinity(blobs), 2, seed=0)
    assert len(set(lab[:30])) == 1 and len(set(lab[30:])) == 1 and lab[0] != lab[-1]
    assert manifold.diffusion_maps(blobs, 2)["embedding"].shape == (60, 2)


def test_wavelets():
    rng = np.random.default_rng(0)
    x = rng.normal(size=256)
    for w in ["haar", "db2"]:
        assert np.allclose(wavelets.idwt(wavelets.dwt(x, w)["coeffs"], w), x)
        assert np.allclose(wavelets.multiresolution(x, w, 3).sum(0), x)
    wv = wavelets.wavelet_variance(x, "haar", 6)
    assert abs(wv["total"] - x.var()) < 1e-9
    sig = np.sin(2 * np.pi * np.arange(512) / 50); noisy = sig + 0.3 * rng.normal(size=512)
    den = wavelets.wavelet_denoise(noisy, "db2", level=3, mode="hard")
    assert np.abs(den - sig).mean() < np.abs(noisy - sig).mean()
