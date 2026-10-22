"""
时间序列统计检验（无 statsmodels 依赖）
======================================

- ADF 单位根检验（临界值用 MacKinnon 2010 的响应面近似）
- Ljung–Box 自相关检验
- 自相关函数、AR(1) 半衰期

参考：MacKinnon (2010). Critical Values for Cointegration Tests. QED Working Paper 1227.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .._utils import as_1d

# MacKinnon (2010) Table 2: τ 统计量临界值的响应面系数 [β∞, β1, β2, β3]，cv(T) = β∞ + β1/T + β2/T² + β3/T³
_MACKINNON = {
    "nc": {"1%": [-2.56574, -2.2358, -3.627, 0.0], "5%": [-1.94100, -0.2686, -3.365, 31.223], "10%": [-1.61682, 0.2656, -2.714, 25.364]},
    "c": {"1%": [-3.43035, -6.5393, -16.786, -79.433], "5%": [-2.86154, -2.8903, -4.234, -40.040], "10%": [-2.56677, -1.5384, -2.809, 0.0]},
    "ct": {"1%": [-3.95877, -9.0531, -28.428, -134.155], "5%": [-3.41049, -4.3904, -9.036, -45.374], "10%": [-3.12705, -2.5856, -3.925, -22.380]},
}


def adf_critical_values(T: int, regression: str = "c") -> dict:
    out = {}
    for lvl, (b0, b1, b2, b3) in _MACKINNON[regression].items():
        out[lvl] = b0 + b1 / T + b2 / T ** 2 + b3 / T ** 3
    return out


def adf_test(x, max_lag: int | None = None, regression: str = "c", autolag: str = "aic") -> dict:
    """Augmented Dickey–Fuller 检验。

    regression ∈ {'nc'（无常数）, 'c'（常数）, 'ct'（常数+趋势）}；autolag='aic' 时在 0..max_lag 中按 AIC 选滞后阶。
    返回 ``stat``、``lag``、``critical_values``、``reject``（各水平是否拒绝单位根）。
    """
    x = as_1d(x, "x")
    T = len(x)
    max_lag = int(np.ceil(12 * (T / 100) ** 0.25)) if max_lag is None else max_lag
    dx = np.diff(x)

    def fit(p):
        n = len(dx) - p
        y = dx[p:]
        cols = [x[p:-1]]
        for i in range(1, p + 1):
            cols.append(dx[p - i:len(dx) - i])
        if regression in ("c", "ct"):
            cols.append(np.ones(n))
        if regression == "ct":
            cols.append(np.arange(n, dtype=float))
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        k = A.shape[1]
        sigma2 = resid @ resid / (n - k)
        cov = sigma2 * np.linalg.pinv(A.T @ A)
        t_stat = coef[0] / np.sqrt(cov[0, 0])
        aic = n * np.log(resid @ resid / n) + 2 * k
        return t_stat, aic

    if autolag == "aic":
        best = min(range(max_lag + 1), key=lambda p: fit(p)[1])
    else:
        best = max_lag
    stat, _ = fit(best)
    cv = adf_critical_values(T - best - 1, regression)
    return dict(stat=float(stat), lag=int(best), critical_values=cv, reject={k: bool(stat < v) for k, v in cv.items()})


def autocorrelation(x, nlags: int = 20) -> np.ndarray:
    x = as_1d(x, "x") - np.mean(x)
    denom = x @ x
    return np.array([1.0] + [x[k:] @ x[:-k] / denom for k in range(1, nlags + 1)])


def ljung_box(x, lags: int = 10) -> dict:
    """Ljung–Box Q 统计量与 p 值。"""
    x = as_1d(x, "x"); n = len(x)
    rho = autocorrelation(x, lags)[1:]
    Q = n * (n + 2) * np.sum(rho ** 2 / (n - np.arange(1, lags + 1)))
    return dict(Q=float(Q), p_value=float(1 - stats.chi2.cdf(Q, lags)))


def ar1_half_life(x) -> float:
    """AR(1) 系数 φ 对应的半衰期 −ln2/ln φ。"""
    x = as_1d(x, "x")
    phi = np.corrcoef(x[1:], x[:-1])[0, 1]
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else np.inf
