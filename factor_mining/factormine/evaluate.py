"""因子评估：IC / RankIC / ICIR / 分层收益 / 多空收益 / 换手率 / 衰减 / 相关性。

所有指标都是横截面口径：每个时间点在资产之间计算，再沿时间汇总。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def rowwise_corr(a: pd.DataFrame, b: pd.DataFrame, min_obs: int = 5) -> pd.Series:
    """逐行（每个时间点）计算两张表在资产维度上的皮尔逊相关，自动处理缺失值。"""
    x, y = a.to_numpy(float), b.reindex_like(a).to_numpy(float)
    m = ~(np.isnan(x) | np.isnan(y))
    n = m.sum(1)
    x = np.where(m, x, 0.0)
    y = np.where(m, y, 0.0)
    with np.errstate(all="ignore"):
        mx = x.sum(1) / n
        my = y.sum(1) / n
        xd = np.where(m, x - mx[:, None], 0.0)
        yd = np.where(m, y - my[:, None], 0.0)
        cov = (xd * yd).sum(1)
        den = np.sqrt((xd**2).sum(1) * (yd**2).sum(1))
        r = cov / den
    r[(n < min_obs) | (den < 1e-12)] = np.nan
    return pd.Series(r, index=a.index)


def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame, rank: bool = True) -> pd.Series:
    """每期 IC。rank=True 时为 RankIC（Spearman），对异常值更稳健。"""
    if rank:
        factor = factor.rank(axis=1)
        fwd = fwd.rank(axis=1)
    return rowwise_corr(factor, fwd)


def turnover(factor: pd.DataFrame) -> float:
    """基于横截面排名的自相关换手代理：1 - corr(rank_t, rank_{t-1})。越低越便于执行。"""
    r = factor.rank(axis=1)
    ac = rowwise_corr(r, r.shift(1)).mean()
    return float(1 - ac) if np.isfinite(ac) else np.nan


def quantile_returns(factor: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    """按因子分 n_q 组的每期平均未来收益（列 1 为最低组）。"""
    pct = factor.rank(axis=1, pct=True)
    q = np.ceil(pct * n_q).clip(1, n_q)
    out = {}
    for k in range(1, n_q + 1):
        out[k] = fwd.where(q == k).mean(axis=1)
    return pd.DataFrame(out)


def long_short_returns(factor: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.Series:
    qr = quantile_returns(factor, fwd, n_q)
    return qr[n_q] - qr[1]


@dataclass
class FactorReport:
    ic: float
    rank_ic: float
    icir: float
    rank_icir: float
    ic_t: float
    ls_ann_return: float
    ls_sharpe: float
    turnover: float
    monotonicity: float
    coverage: float
    n_periods: int

    def to_dict(self) -> Dict[str, float]:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def summary(self) -> str:
        return (
            f"IC={self.ic:.4f} RankIC={self.rank_ic:.4f} RankICIR={self.rank_icir:.3f} "
            f"多空年化={self.ls_ann_return:.2%} 多空夏普={self.ls_sharpe:.2f} 换手={self.turnover:.3f}"
        )


def evaluate_factor(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    periods_per_year: int = 252,
    horizon: int = 1,
    n_q: int = 5,
) -> FactorReport:
    """完整评估一个因子。fwd 是与因子对齐的未来收益（见 data.forward_returns）。"""
    ic = ic_series(factor, fwd, rank=False)
    ric = ic_series(factor, fwd, rank=True)
    ls = long_short_returns(factor, fwd, n_q).dropna()
    qr = quantile_returns(factor, fwd, n_q).mean()
    ann = periods_per_year / horizon
    mono = float(pd.Series(qr.values).rank().corr(pd.Series(np.arange(n_q, dtype=float))))
    sd = lambda s: s.std() if s.std() > 0 else np.nan
    return FactorReport(
        ic=float(ic.mean()),
        rank_ic=float(ric.mean()),
        icir=float(ic.mean() / sd(ic) * np.sqrt(ann)) if ic.notna().sum() > 2 else np.nan,
        rank_icir=float(ric.mean() / sd(ric) * np.sqrt(ann)) if ric.notna().sum() > 2 else np.nan,
        ic_t=float(ric.mean() / sd(ric) * np.sqrt(ric.notna().sum())) if ric.notna().sum() > 2 else np.nan,
        ls_ann_return=float(ls.mean() * ann) if len(ls) else np.nan,
        ls_sharpe=float(ls.mean() / sd(ls) * np.sqrt(ann)) if len(ls) > 2 else np.nan,
        turnover=turnover(factor),
        monotonicity=mono,
        coverage=float(factor.notna().to_numpy().mean()),
        n_periods=int(ric.notna().sum()),
    )


def ic_decay(factor: pd.DataFrame, close: pd.DataFrame, horizons: Sequence[int] = (1, 2, 5, 10, 20),
             marginal: bool = False) -> pd.Series:
    """不同持有期的 RankIC，用来决定信号半衰期——交给挂单/执行层选择下单节奏。

    marginal=True 时用第 h 期的单期收益（衡量信号在第 h 期还剩多少预测力），否则用累计收益。
    """
    out = {}
    for h in horizons:
        if marginal:
            fwd = close.shift(-h) / close.shift(-h + 1) - 1
        else:
            fwd = close.shift(-h) / close - 1
        out[h] = ic_series(factor, fwd).mean()
    return pd.Series(out, name="rank_ic")


def factor_correlation(f1: pd.DataFrame, f2: pd.DataFrame) -> float:
    """两个因子横截面秩相关的时间均值，用于去冗余。"""
    return float(rowwise_corr(f1.rank(axis=1), f2.rank(axis=1)).mean())


def correlation_matrix(factors: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(factors)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            m.loc[a, b] = m.loc[b, a] = factor_correlation(factors[a], factors[b])
    return m


def fitness(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    rows: Optional[slice] = None,
    min_coverage: float = 0.5,
) -> float:
    """挖掘器使用的快速适应度：|RankIC 均值| ，覆盖率过低则判 0。rows 用于只在训练段打分。"""
    if rows is not None:
        factor, fwd = factor.iloc[rows], fwd.iloc[rows]
    if factor.notna().to_numpy().mean() < min_coverage:
        return 0.0
    ric = ic_series(factor, fwd).mean()
    return float(abs(ric)) if np.isfinite(ric) else 0.0
