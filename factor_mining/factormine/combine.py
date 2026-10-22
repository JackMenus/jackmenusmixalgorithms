"""多因子合成与去相关。

提供静态权重（等权 / IC 加权 / ICIR 加权 / 最大化 IC_IR / 岭回归）、
动态权重（AlphaForge 式滚动重估 + 每期选 Top-K）以及正交化（施密特 / 对称正交）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .evaluate import ic_series
from .operators import cs_winsorize, cs_zscore

Factors = Dict[str, pd.DataFrame]
Method = Literal["equal", "ic", "icir", "max_ir", "ridge"]


def standardize(f: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    """MAD 去极值 + 横截面 z-score，缺失填 0（=中性）。"""
    return cs_zscore(cs_winsorize(f, clip)).clip(-clip, clip)


def _stack(factors: Factors) -> tuple[np.ndarray, List[str]]:
    names = list(factors)
    arr = np.stack([standardize(factors[n]).to_numpy(float) for n in names], axis=-1)  # T,N,K
    return arr, names


def _ic_matrix(factors: Factors, fwd: pd.DataFrame, rows: Optional[slice]) -> pd.DataFrame:
    ics = {}
    for n, f in factors.items():
        s = ic_series(f, fwd)
        ics[n] = s.iloc[rows] if rows is not None else s
    return pd.DataFrame(ics)


def fit_weights(
    factors: Factors,
    fwd: pd.DataFrame,
    method: Method = "max_ir",
    rows: Optional[slice] = None,
    shrink: float = 0.3,
    ridge_lambda: float = 1.0,
) -> pd.Series:
    """在训练区间 rows 内估计合成权重。

    * equal  —— 按 IC 符号等权；
    * ic     —— w ∝ mean(IC)；
    * icir   —— w ∝ mean(IC)/std(IC)；
    * max_ir —— w ∝ Σ_IC^{-1} μ_IC（Qian & Hua 最大化复合 IC_IR，协方差向对角收缩）；
    * ridge  —— 把所有 (t, asset) 样本堆叠做岭回归（Gu-Kelly-Xiu 2020 中的线性基准）。
    """
    names = list(factors)
    if method == "ridge":
        arr, _ = _stack(factors)
        y = standardize(fwd).to_numpy(float)
        if rows is not None:
            arr, y = arr[rows], y[rows]
        X = np.nan_to_num(arr.reshape(-1, len(names)))
        yv = y.reshape(-1)
        m = np.isfinite(yv)
        X, yv = X[m], yv[m]
        w = np.linalg.solve(X.T @ X + ridge_lambda * len(yv) * 1e-3 * np.eye(len(names)), X.T @ yv)
        return pd.Series(w, index=names)

    ic = _ic_matrix(factors, fwd, rows)
    mu = ic.mean().fillna(0.0)
    if method == "equal":
        w = np.sign(mu)
    elif method == "ic":
        w = mu
    elif method == "icir":
        w = mu / ic.std().replace(0, np.nan)
    elif method == "max_ir":
        cov = ic.cov().fillna(0.0).to_numpy()
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        cov += 1e-8 * np.eye(len(names))
        w = pd.Series(np.linalg.solve(cov, mu.to_numpy()), index=names)
    else:
        raise ValueError(method)
    w = pd.Series(w, index=names).fillna(0.0)
    s = w.abs().sum()
    return w / s if s > 0 else w


def combine(factors: Factors, weights: pd.Series) -> pd.DataFrame:
    """按权重线性合成标准化后的因子，输出再做一次横截面标准化。"""
    out = None
    for n, w in weights.items():
        if w == 0 or n not in factors:
            continue
        term = standardize(factors[n]).fillna(0.0) * w
        out = term if out is None else out + term
    if out is None:
        raise ValueError("没有可用因子")
    return cs_zscore(out)


@dataclass
class DynamicCombiner:
    """AlphaForge（AAAI 2025）式动态合成：每期用过去 lookback 期的 IC 重新筛选 Top-K 并估权重。

    只使用 t 之前已经实现的 IC（IC_t 需要 t+horizon 的收益，故额外滞后 horizon 期），无前视。
    """

    lookback: int = 60
    top_k: int = 10
    horizon: int = 1
    method: Literal["ic", "icir"] = "icir"
    weights_: Optional[pd.DataFrame] = field(default=None, init=False)

    def fit_transform(self, factors: Factors, fwd: pd.DataFrame) -> pd.DataFrame:
        ic = _ic_matrix(factors, fwd, None).shift(self.horizon)  # t 时刻只能看到已实现的 IC
        mu = ic.rolling(self.lookback, min_periods=self.lookback // 2).mean()
        score = mu
        if self.method == "icir":
            score = mu / ic.rolling(self.lookback, min_periods=self.lookback // 2).std()
        rank = score.abs().rank(axis=1, ascending=False)
        w = score.where(rank <= self.top_k, 0.0).fillna(0.0)
        w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        self.weights_ = w
        out = None
        for n in factors:
            term = standardize(factors[n]).fillna(0.0).mul(w[n], axis=0)
            out = term if out is None else out + term
        return cs_zscore(out)


def orthogonalize(factors: Factors, method: Literal["schmidt", "symmetric"] = "symmetric") -> Factors:
    """逐期横截面正交化。

    * schmidt   —— 施密特正交，结果依赖因子顺序（越靠前越“保真”）；
    * symmetric —— 对称（Löwdin）正交，S^{-1/2} 变换，与顺序无关且与原因子整体距离最小。
    """
    arr, names = _stack(factors)
    arr = np.nan_to_num(arr)
    T, N, K = arr.shape
    out = np.zeros_like(arr)
    for t in range(T):
        F = arr[t]
        if not F.any():
            continue
        if method == "schmidt":
            Q = np.zeros_like(F)
            for k in range(K):
                v = F[:, k].copy()
                for j in range(k):
                    nj = Q[:, j] @ Q[:, j]
                    if nj > 1e-12:
                        v -= (v @ Q[:, j]) / nj * Q[:, j]
                Q[:, k] = v
            out[t] = Q
        else:
            M = F.T @ F / N
            vals, vecs = np.linalg.eigh(M)
            vals = np.clip(vals, 1e-8, None)
            out[t] = F @ (vecs @ np.diag(vals**-0.5) @ vecs.T)
    idx, cols = next(iter(factors.values())).index, next(iter(factors.values())).columns
    return {n: pd.DataFrame(out[..., k], index=idx, columns=cols) for k, n in enumerate(names)}


def greedy_decorrelate(
    factors: Factors, scores: Dict[str, float], corr: pd.DataFrame, max_corr: float = 0.7
) -> List[str]:
    """按得分从高到低贪心入选，与已入选因子相关性绝对值超过 max_corr 的剔除。"""
    chosen: List[str] = []
    for n in sorted(scores, key=lambda k: -abs(scores[k])):
        if all(abs(corr.loc[n, c]) <= max_corr for c in chosen):
            chosen.append(n)
    return chosen
