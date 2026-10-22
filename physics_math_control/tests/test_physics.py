"""物理模块测试。"""
import numpy as np
import pytest

from crossquant.physics import annealing, entropy, first_passage, hawkes, hurst, ising, mcmc, networks, ou_process, pde, random_matrix, rough_volatility


def test_ou_fit_recovers_parameters():
    x = ou_process.simulate_ou(1.0, 2.0, 0.5, 1.0, dt=1 / 252, n=8000, seed=1)
    p = ou_process.fit_ou(x, 1 / 252)
    assert abs(p.theta - 1.0) < 0.15 and abs(p.kappa - 2.0) < 0.8 and abs(p.sigma - 0.5) < 0.05
    assert np.isclose(p.half_life, np.log(2) / p.kappa)
    bt = ou_process.ou_signal_backtest(x, p, entry_z=1.5)
    assert bt["n_trades"] > 0 and bt["pnl"].sum() > 0


def test_hawkes_mle_and_residuals():
    ev = hawkes.simulate_hawkes(0.5, 0.8, 1.2, 3000, seed=2)
    f = hawkes.fit_hawkes(ev, 3000)
    assert abs(f["mu"] - 0.5) < 0.1 and abs(f["branching_ratio"] - 0.8 / 1.2) < 0.08
    res = hawkes.hawkes_residuals(ev[:1500], f["mu"], f["alpha"], f["beta"])
    assert res["ks_pvalue"] > 0.01
    mt, mk = hawkes.simulate_multivariate_hawkes([0.3, 0.3], [[0.4, 0.3], [0.2, 0.5]], [[1.5, 1.5], [1.5, 1.5]], 1500, seed=3)
    mf = hawkes.fit_multivariate_hawkes(mt, mk, 1500)
    assert mf["spectral_radius"] < 1 and np.all(mf["alpha"] > 0)
    assert np.allclose(mf["alpha"], [[0.4, 0.3], [0.2, 0.5]], atol=0.2)


def test_random_matrix_cleaning():
    rng = np.random.default_rng(0)
    T, N = 1000, 100
    X = rng.standard_normal((T, N))
    w = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))
    lo, hi = random_matrix.mp_bounds(N / T)
    assert np.mean((w > lo - 0.05) & (w < hi + 0.05)) > 0.97
    beta = rng.normal(1, 0.3, N); R = np.outer(rng.standard_normal(T), beta) + 2 * rng.standard_normal((T, N))
    true_cov = np.outer(beta, beta) + 4 * np.eye(N); d = np.sqrt(np.diag(true_cov)); true_corr = true_cov / np.outer(d, d)
    Cs = np.corrcoef(R, rowvar=False)
    err = lambda M: np.linalg.norm(M - true_corr)
    clip = random_matrix.clip_eigenvalues(Cs, N / T)
    assert clip["n_signal"] == 1
    assert err(clip["corr_clean"]) < 0.8 * err(Cs)
    assert err(random_matrix.rie_cleaning(Cs, N / T)["corr_clean"]) < 0.8 * err(Cs)
    lw = random_matrix.ledoit_wolf_shrinkage(R)
    assert 0 < lw["shrinkage"] < 1
    assert 0 < random_matrix.absorption_ratio(true_cov, 1) < 1


def test_hurst_and_variance_ratio():
    rng = np.random.default_rng(0)
    wn = rng.standard_normal(4000)
    assert abs(hurst.dfa(wn)["H"] - 0.5) < 0.1
    vr = hurst.variance_ratio(wn, 5)
    assert abs(vr["VR"] - 1) < 0.1 and abs(vr["z"]) < 3
    fb = rough_volatility.fbm_davies_harte(4000, 0.7, seed=5)
    inc = np.diff(fb)
    assert abs(inc.var() - (1 / 4000) ** 1.4) / (1 / 4000) ** 1.4 < 0.15
    assert abs(hurst.dfa(inc)["H"] - 0.7) < 0.12
    assert abs(hurst.generalized_hurst(fb, 1.0)["H"] - 0.7) < 0.1


def test_rough_volatility_estimator_and_rbergomi():
    lv = rough_volatility.fbm_davies_harte(5000, 0.12, seed=6)
    assert abs(rough_volatility.hurst_from_realized_variance(lv)["H"] - 0.12) < 0.04
    rb = rough_volatility.rbergomi_simulate(100, 1500, H=0.1, eta=1.9, rho=-0.9, xi0=0.04, T=1.0, seed=7)
    assert abs(rb["Y"][:, -1].var() - 1.0) < 0.15
    assert abs(rb["S"][:, -1].mean() - 1.0) < 0.05
    assert np.all(rb["V"] > 0)


def test_entropy_tools():
    rng = np.random.default_rng(0)
    n = 3000
    a = rng.standard_normal(n); rho = 0.6; b = rho * a + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    assert abs(entropy.mutual_information_ksg(a, b) - (-0.5 * np.log(1 - rho ** 2))) < 0.05
    y = np.zeros(n); x = rng.standard_normal(n)
    for t in range(1, n):
        y[t] = 0.6 * x[t - 1] + 0.5 * rng.standard_normal()
    assert entropy.transfer_entropy(x, y) > 5 * entropy.transfer_entropy(y, x)
    ete = entropy.effective_transfer_entropy(x, y, n_shuffle=20, seed=0)
    assert ete["z"] > 5 and ete["p_value"] == 0
    assert entropy.permutation_entropy(rng.standard_normal(2000)) > 0.98
    assert entropy.permutation_entropy(np.sin(np.arange(2000) / 10)) < 0.6
    assert np.isclose(entropy.shannon_entropy([0.5, 0.5], base=2), 1.0)


def test_networks_and_hrp():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (500, 10)); R[:, :5] += np.outer(rng.normal(0, 0.01, 500), np.ones(5)); R[:, 5:] += np.outer(rng.normal(0, 0.01, 500), np.ones(5))
    corr = np.corrcoef(R, rowvar=False)
    mst = networks.minimum_spanning_tree(networks.correlation_distance(corr))
    assert len(mst["edges"]) == 9
    cross = sum(1 for i, j, _ in mst["edges"] if (i < 5) != (j < 5))
    assert cross == 1  # 两簇之间只有一条边
    cov = np.cov(R, rowvar=False)
    hrp = networks.hierarchical_risk_parity(cov)
    assert np.isclose(hrp["weights"].sum(), 1) and np.all(hrp["weights"] > 0)
    rp = networks.risk_parity_weights(cov)
    assert np.allclose(networks.risk_contributions(rp, cov), 0.1, atol=1e-3)
    labels = networks.correlation_clusters(corr, 2)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1


def test_first_passage_closed_forms_vs_mc():
    tb = first_passage.bm_two_barrier(1.0, 2.0, 0.0, 1.0)
    assert np.isclose(tb["p_upper"], 2 / 3) and np.isclose(tb["expected_exit_time"], 2.0)
    tb2 = first_passage.bm_two_barrier(1.0, 1.0, 0.5, 1.0)
    mc = first_passage.triple_barrier_mc(0.5, 1.0, 1.0, 1.0, T=60, dt=0.002, n_paths=4000, seed=1)
    assert abs(mc["p_take_profit"] - tb2["p_upper"]) < 0.04
    assert abs(mc["mean_exit_time"] - tb2["expected_exit_time"]) < 0.1
    assert np.isclose(first_passage.bm_prob_hit_before_T(1.0, 0.0, 1.0, 1.0), 2 * (1 - 0.8413447460685429), atol=1e-6)


def test_mcmc_samplers():
    lp = lambda x: -0.5 * np.sum(x ** 2); gl = lambda x: -x
    mh = mcmc.metropolis_hastings(lp, np.zeros(2), 8000, 1.0, seed=0)
    hm = mcmc.hamiltonian_monte_carlo(lp, gl, np.zeros(2), 4000, 0.5, 8, seed=0)
    assert np.allclose(mh["samples"].mean(0), 0, atol=0.15) and np.allclose(mh["samples"].var(0), 1, atol=0.2)
    assert np.allclose(hm["samples"].mean(0), 0, atol=0.1) and np.allclose(hm["samples"].var(0), 1, atol=0.2)
    rng = np.random.default_rng(0)
    rt = 0.01 * rng.standard_t(4, 1500)
    bs = mcmc.bayesian_student_t(rt, n_samples=500, burn_in=200, seed=0)
    assert bs["accept_rate"] > 0.6
    assert 2.5 < bs["summary"]["nu"][0] < 7 and abs(bs["summary"]["sigma"][0] - 0.01) < 0.002


def test_pde_pricing():
    closed = pde.black_scholes_price(100, 100, 0.05, 0.2, 1.0, "put")
    for m in ["explicit", "implicit", "crank_nicolson"]:
        assert abs(pde.bs_pde_fd(100, 100, 0.05, 0.2, 1.0, kind="put", method=m, n_s=200)["price"] - closed) < 0.03
    am = pde.bs_pde_fd(36, 40, 0.06, 0.2, 1.0, kind="put", american=True, n_s=400, n_t=400)["price"]
    assert abs(am - 4.487) < 0.02


def test_annealing_matches_brute_force():
    import itertools
    rng = np.random.default_rng(0)
    mu = rng.normal(0.05, 0.03, 8); A = rng.normal(size=(8, 8)); S = A @ A.T / 100
    cc = annealing.cardinality_constrained_portfolio(mu, S, 3, gamma=2.0, n_steps=400, seed=0)
    best = max(-annealing._mv_long_only(mu[list(c)], S[np.ix_(c, c)], 2.0)[1] for c in itertools.combinations(range(8), 3))
    assert abs(cc["objective"] - best) < 1e-6
    sa = annealing.simulated_annealing(lambda x: (x - 3) ** 2, lambda x, r: x + r.normal(), 0.0, n_steps=2000, seed=0)
    assert abs(sa["best"] - 3) < 0.1


def test_ising_runs():
    out = ising.bornholdt_ising(L=8, n_sweeps=30, burn_in=5, seed=0)
    assert len(out["magnetization"]) == 30 and np.all(np.abs(out["magnetization"]) <= 1)
