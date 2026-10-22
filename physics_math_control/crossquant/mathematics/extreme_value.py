"""
极值理论（EVT）：尾部指数、POT/GPD 与 GEV
==========================================

学科来源
--------
Fisher–Tippett (1928)、Gnedenko (1943) 的极值定理：区块极大值的极限分布只能是 GEV 三型之一；
Pickands–Balkema–de Haan (1974/75)：超阈值的极限分布是广义 Pareto（GPD）。Hill (1975) 的尾指数估计。
最初用于水文（洪水）、保险与可靠性工程。

量化映射
--------
- **尾部风险**：VaR/ES 在 99%+ 置信度下用 GPD 外推比经验分位数稳定得多；
- **厚尾程度**：Hill 尾指数 α（股票日收益约 3–4，“立方律” Gabaix 等 2003）；α 的下降是脆弱性的信号；
- **极端事件标度**：幂律尾 ⇒ 极端波动与成交量的标度关系，用于压力测试情景；
- **止损设计 / 保证金**：按尾部分布而非正态设定。

核心公式
--------
Hill： ξ̂ = (1/k) Σ_{i=1}^k log( X_{(n−i+1)} / X_{(n−k)} )，α̂ = 1/ξ̂
GPD（超阈值 y = x − u）：G(y) = 1 − (1 + ξ y/β)^{−1/ξ}
VaR_p = u + (β/ξ)[ ((n/N_u)(1−p))^{−ξ} − 1 ]，ES_p = VaR_p/(1−ξ) + (β − ξu)/(1−ξ)

参考文献
--------
- Embrechts, Klüppelberg & Mikosch (1997). Modelling Extremal Events for Insurance and Finance.
- McNeil & Frey (2000). Estimation of tail-related risk measures for heteroscedastic financial time series. JEF.
- Hill (1975). A simple general approach to inference about the tail of a distribution. Ann. Stat.
- Gabaix, Gopikrishnan, Plerou & Stanley (2003). A theory of power-law distributions in financial market fluctuations. Nature.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .._utils import as_1d


def hill_estimator(x, k: int, tail: str = "right") -> dict:
    """Hill 尾指数估计（取最大的 k 个样本）。返回 ``xi``、``alpha``、``std_err``（α 的渐近标准误 α/√k）。"""
    x = as_1d(x, "x")
    y = np.sort(x if tail == "right" else -x)
    y = y[y > 0]
    n = len(y)
    if k >= n or k < 2:
        raise ValueError("k 需满足 2 ≤ k < 正样本数")
    top = y[n - k:]; thr = y[n - k - 1]
    xi = np.mean(np.log(top / thr))
    return dict(xi=float(xi), alpha=float(1 / xi), std_err=float((1 / xi) / np.sqrt(k)), threshold=float(thr))


def hill_plot(x, ks=None, tail: str = "right") -> dict:
    x = as_1d(x, "x")
    n = int(np.sum((x if tail == "right" else -x) > 0))
    ks = np.arange(10, max(n // 2, 12)) if ks is None else np.asarray(ks)
    alphas = np.array([hill_estimator(x, int(k), tail)["alpha"] for k in ks])
    return dict(k=ks, alpha=alphas)


def fit_gpd(excesses) -> dict:
    """对超阈值 y ≥ 0 做 GPD 极大似然（scipy genpareto，位置固定为 0）。返回 ``xi``、``beta``。"""
    y = as_1d(excesses, "excesses")
    xi, loc, beta = stats.genpareto.fit(y, floc=0)
    return dict(xi=float(xi), beta=float(beta), loglik=float(np.sum(stats.genpareto.logpdf(y, xi, 0, beta))))


def pot_var_es(x, threshold: float | None = None, p=(0.99, 0.999), tail: str = "left", quantile_u: float = 0.95) -> dict:
    """峰值超阈值法（POT）估计 VaR 与 ES。tail='left' 时对损失 −x 建模。

    返回 ``threshold``、``xi``、``beta``、``VaR``（dict）、``ES``（dict）、``n_exceed``。
    """
    x = as_1d(x, "x")
    loss = -x if tail == "left" else x
    u = np.quantile(loss, quantile_u) if threshold is None else threshold
    exc = loss[loss > u] - u
    n, Nu = len(loss), len(exc)
    g = fit_gpd(exc)
    xi, beta = g["xi"], g["beta"]
    VaR, ES = {}, {}
    for q in np.atleast_1d(p):
        if abs(xi) < 1e-8:
            var_q = u + beta * np.log((n / Nu) / (1 - q))
        else:
            var_q = u + beta / xi * (((n / Nu) * (1 - q)) ** (-xi) - 1)
        es_q = var_q / (1 - xi) + (beta - xi * u) / (1 - xi) if xi < 1 else np.inf
        VaR[float(q)] = float(var_q); ES[float(q)] = float(es_q)
    return dict(threshold=float(u), xi=xi, beta=beta, VaR=VaR, ES=ES, n_exceed=int(Nu))


def fit_gev_block_maxima(x, block_size: int, return_period: float | None = None) -> dict:
    """区块极大值 GEV 拟合。返回 ``shape``（scipy 约定 c = −ξ）、``xi``、``loc``、``scale``、可选 ``return_level``。"""
    x = as_1d(x, "x")
    n_blocks = len(x) // block_size
    maxima = x[:n_blocks * block_size].reshape(n_blocks, block_size).max(axis=1)
    c, loc, scale = stats.genextreme.fit(maxima)
    out = dict(shape=float(c), xi=float(-c), loc=float(loc), scale=float(scale), maxima=maxima)
    if return_period is not None:
        out["return_level"] = float(stats.genextreme.ppf(1 - 1 / return_period, c, loc, scale))
    return out


def mean_excess(x, thresholds=None, n: int = 30) -> dict:
    """平均超额函数 e(u) = E[X − u | X > u]，GPD 尾部时为 u 的线性函数（用于选阈值）。"""
    x = as_1d(x, "x")
    thresholds = np.quantile(x, np.linspace(0.5, 0.98, n)) if thresholds is None else np.asarray(thresholds)
    me = np.array([np.mean(x[x > u] - u) if np.any(x > u) else np.nan for u in thresholds])
    return dict(thresholds=thresholds, mean_excess=me)
