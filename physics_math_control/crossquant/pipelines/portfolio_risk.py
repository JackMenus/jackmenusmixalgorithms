"""
端到端流水线：协方差去噪 → 组合构建 → 样本外比较
=================================================

零件：随机矩阵理论（物理）、Ledoit–Wolf 收缩（统计）、HRP/风险平价（网络/层次）、最小方差（线代）。
"""
from __future__ import annotations

import numpy as np

from .._utils import cov_to_corr
from ..physics.networks import hierarchical_risk_parity, inverse_variance_weights, risk_parity_weights
from ..physics.random_matrix import clip_eigenvalues, ledoit_wolf_shrinkage, rie_cleaning


def min_variance_weights(cov, long_only: bool = False) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if not long_only:
        w = np.linalg.solve(cov, np.ones(len(cov)))
        return w / w.sum()
    from scipy.optimize import minimize
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), bounds=[(0, 1)] * n,
                   constraints=[dict(type="eq", fun=lambda w: w.sum() - 1)], method="SLSQP")
    return res.x


def denoised_covariances(returns) -> dict:
    """返回 {'sample','clip','rie','lw'} 四种协方差估计。"""
    R = np.asarray(returns, dtype=float)
    T, N = R.shape
    cov = np.cov(R, rowvar=False)
    std = np.sqrt(np.diag(cov)); corr = cov_to_corr(cov)
    q = N / T
    out = dict(sample=cov)
    out["clip"] = clip_eigenvalues(corr, q)["corr_clean"] * np.outer(std, std)
    out["rie"] = rie_cleaning(corr, q)["corr_clean"] * np.outer(std, std)
    out["lw"] = ledoit_wolf_shrinkage(R)["cov"]
    return out


def allocation_comparison(train_returns, test_returns) -> dict:
    """在训练集估计协方差并构建组合，在测试集评估年化波动（按 252 期）与权重集中度。"""
    covs = denoised_covariances(train_returns)
    test = np.asarray(test_returns, dtype=float)
    rows = {}
    for name, cov in covs.items():
        for alloc, w in [("minvar", min_variance_weights(cov, long_only=True)), ("hrp", hierarchical_risk_parity(cov)["weights"]),
                         ("riskparity", risk_parity_weights(cov)), ("invvar", inverse_variance_weights(cov))]:
            pnl = test @ w
            rows[f"{name}/{alloc}"] = dict(oos_vol_annual=float(pnl.std() * np.sqrt(252)), herfindahl=float(np.sum(w ** 2)),
                                           max_weight=float(w.max()), weights=w)
    return rows
