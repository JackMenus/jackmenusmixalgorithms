"""
端到端流水线：配对交易 / 统计套利
=================================

把库中的几个“跨学科零件”串起来：
1. OLS 对冲比率 → 价差（Engle–Granger 第一步）
2. ADF 单位根检验（数学/计量）→ 是否协整
3. OU 参数估计（物理/Langevin）→ 半衰期、平稳标准差、z-score
4. 首达时间（物理/扩散）→ 从开仓水平回到均值的时间分布、是否值得等
5. 可选：Kalman 动态对冲比率（控制）→ 自适应 β
6. 样本外回测

以及 Avellaneda–Lee (2010) 的 PCA 本征组合残差 → OU → s-score 流程。
"""
from __future__ import annotations

import numpy as np

from ..control.kalman import dynamic_hedge_ratio
from ..mathematics.timeseries import adf_test
from ..physics.first_passage import ou_first_passage_mc
from ..physics.ou_process import OUParams, fit_ou, hedge_ratio_ols, ou_signal_backtest, ou_zscore


def pairs_trading_pipeline(price_y, price_x, train_frac: float = 0.6, dt: float = 1.0, entry_z: float = 1.5,
                           exit_z: float = 0.0, cost: float = 0.0, use_kalman: bool = False, kalman_delta: float = 1e-4,
                           mc_paths: int = 5000, seed=None) -> dict:
    """返回 dict：``hedge``、``adf``、``ou``（参数与半衰期）、``revert_time``（首达时间统计）、``backtest``、``kalman``（可选）。"""
    y = np.asarray(price_y, dtype=float); x = np.asarray(price_x, dtype=float)
    n_train = int(len(y) * train_frac)
    hedge = hedge_ratio_ols(y[:n_train], x[:n_train])
    spread_all = y - hedge["beta"] * x - hedge["alpha"]
    adf = adf_test(spread_all[:n_train])
    ou = fit_ou(spread_all[:n_train], dt)
    entry_level = ou.theta + entry_z * ou.stationary_std
    revert = ou_first_passage_mc(ou.theta, max(ou.kappa, 1e-8), ou.sigma, entry_level, ou.theta, dt=dt / 10,
                                 T_max=20 * ou.half_life if np.isfinite(ou.half_life) else 100 * dt, n_paths=mc_paths, seed=seed)
    bt = ou_signal_backtest(spread_all[n_train:], ou, entry_z, exit_z, cost)
    out = dict(hedge=dict(beta=hedge["beta"], alpha=hedge["alpha"]), adf=adf,
               ou=dict(theta=ou.theta, kappa=ou.kappa, sigma=ou.sigma, half_life=ou.half_life, stationary_std=ou.stationary_std),
               revert_time=dict(mean=revert["mean"], median=revert["median"], q90=revert["q90"], p_within_T=revert["p_within_T"]),
               backtest=dict(sharpe_per_period=bt["sharpe"], n_trades=bt["n_trades"], total_pnl=float(bt["pnl"].sum())),
               spread=spread_all, zscore=ou_zscore(spread_all, ou), n_train=n_train)
    if use_kalman:
        k = dynamic_hedge_ratio(y, x, delta=kalman_delta)
        out["kalman"] = dict(beta=k["beta"], zscore=k["zscore"], spread=k["spread"])
    return out


def eigenportfolio_residuals(returns, n_factors: int = 3, dt: float = 1.0) -> dict:
    """Avellaneda–Lee 统计套利：PCA 本征组合 → 回归残差 → 累计残差拟合 OU → s-score。

    返回 ``s_score (N,)``（最近一期）、``ou_params``（每个资产）、``half_life``、``factor_returns``、``loadings``。
    """
    R = np.asarray(returns, dtype=float)
    T, N = R.shape
    mu = R.mean(axis=0); sd = R.std(axis=0, ddof=1)
    Z = (R - mu) / sd
    corr = Z.T @ Z / (T - 1)
    w, V = np.linalg.eigh(corr)
    idx = np.argsort(w)[::-1][:n_factors]
    V = V[:, idx]
    weights = V / sd[:, None]                       # 本征组合权重（按波动率缩放）
    F = R @ weights                                  # 因子收益 (T, k)
    A = np.hstack([np.ones((T, 1)), F])
    coef, *_ = np.linalg.lstsq(A, R, rcond=None)
    resid = R - A @ coef
    X = np.cumsum(resid, axis=0)                     # 累计残差
    s_score = np.zeros(N); params = []; hl = np.zeros(N)
    for i in range(N):
        p = fit_ou(X[:, i], dt)
        params.append(p); hl[i] = p.half_life
        s_score[i] = (X[-1, i] - p.theta) / p.stationary_std if p.kappa > 0 else np.nan
    return dict(s_score=s_score, ou_params=params, half_life=hl, factor_returns=F, loadings=coef[1:].T, residuals=resid)
