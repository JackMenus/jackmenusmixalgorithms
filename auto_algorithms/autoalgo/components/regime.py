"""市场状态（regime）探测组件：把 crossquant 的 Hurst/DFA 与贝叶斯在线变点检测（BOCPD）封装成
因子层与组合层都能复用的"状态标签 + 变点概率"序列。

来源组件
  * ``crossquant.physics.hurst.dfa``            去趋势涨落分析，估计 Hurst 指数 H（H>0.5 趋势持续，H<0.5 均值回复）
  * ``crossquant.control.changepoint.bocpd``    Adams–MacKay 在线变点检测，输出"最近 window 步内发生过变点"的后验概率

设计约定
  * 全部函数只用"当前及之前"的数据：滚动 Hurst 在窗口末端取值，BOCPD 是在线滤波，天然无未来函数；
  * 输入 1 维收益序列（numpy / pandas），输出与输入等长、前段用 NaN 或先验值填充。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .._paths import available

REGIME_TREND, REGIME_NEUTRAL, REGIME_MEANREV = "trend", "neutral", "meanrev"
REGIMES = (REGIME_MEANREV, REGIME_NEUTRAL, REGIME_TREND)


def _dfa_hurst(seg: np.ndarray) -> float:
    """优先用 crossquant 的 DFA；库不可用时退化为 Lo–MacKinlay 方差比的 Hurst 近似 H ≈ ½ + ½·log(VR(k))/log(k)。"""
    seg = np.asarray(seg, float)
    seg = seg[np.isfinite(seg)]
    if len(seg) < 32:
        return float("nan")
    if len(seg) >= 64 and available("crossquant"):
        from crossquant.physics.hurst import dfa

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(np.clip(dfa(seg, min_scale=8, n_scales=10)["H"], 0.0, 1.5))
        except Exception:
            pass
    k = 5
    mu = seg.mean()
    var1 = np.var(seg - mu, ddof=1)
    rk = np.convolve(seg, np.ones(k), "valid")
    vark = np.var(rk - k * mu, ddof=1) / k
    vr = vark / max(var1, 1e-18)
    return float(np.clip(0.5 + 0.5 * np.log(max(vr, 1e-9)) / np.log(k), 0.0, 1.5))


def rolling_hurst(returns, window: int = 120, step: int = 5) -> np.ndarray:
    """滚动 Hurst 指数（窗口末端取值，中间用前值填充），长度与输入相同，前 ``window-1`` 个为 NaN。"""
    r = np.asarray(pd.Series(returns).astype(float).values)
    out = np.full(len(r), np.nan)
    last = np.nan
    for end in range(window, len(r) + 1):
        if (end - window) % step == 0:
            last = _dfa_hurst(r[end - window:end])
        out[end - 1] = last
    return out


def change_probability(x, hazard: float = 1 / 120, window: int = 10, standardize: bool = True) -> np.ndarray:
    """BOCPD 变点概率序列 P(r_t < window | x_{1:t})。crossquant 不可用时退化为 CUSUM 风格的 z 统计量映射。"""
    s = pd.Series(x).astype(float)
    v = s.to_numpy()
    good = np.isfinite(v)
    if good.sum() < 20:
        return np.full(len(v), hazard)
    z = v.copy()
    if standardize:  # 扩展窗口标准化：只用 t 及之前的均值/标准差，保证因果性（截断样本与全样本结果一致）
        ser = pd.Series(np.where(good, v, np.nan))
        mu = ser.expanding(min_periods=20).mean().bfill()
        sd = ser.expanding(min_periods=20).std().bfill().replace(0.0, 1.0).fillna(1.0)
        z = ((ser - mu) / sd).to_numpy()
    z = np.where(good & np.isfinite(z), z, 0.0)
    if available("crossquant"):
        from crossquant.control.changepoint import bocpd

        res = bocpd(z, hazard=hazard, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0, max_run=min(500, max(50, len(z))), window=window)
        return np.clip(np.asarray(res["prob_change"], float), 0.0, 1.0)
    # 退化方案：滚动均值偏移的 z 统计量 → logistic 映射
    roll = pd.Series(z).rolling(window, min_periods=3).mean() * np.sqrt(window)
    return 1.0 / (1.0 + np.exp(-(roll.abs().fillna(0.0).to_numpy() - 2.0)))


def label_regimes(hurst: np.ndarray, lo: float = 0.45, hi: float = 0.55) -> np.ndarray:
    """按 Hurst 阈值打标签：H<lo 均值回复，H>hi 趋势，其余中性；NaN → 中性。"""
    h = np.asarray(hurst, float)
    lab = np.full(len(h), REGIME_NEUTRAL, dtype=object)
    lab[h < lo] = REGIME_MEANREV
    lab[h > hi] = REGIME_TREND
    return lab


@dataclass
class RegimeState:
    hurst: float
    regime: str
    prob_change: float

    def to_dict(self) -> Dict[str, float]:
        return {"hurst": self.hurst, "regime": self.regime, "prob_change": self.prob_change}


class RegimeDetector:
    """把一条市场收益序列变成 ``DataFrame(hurst, prob_change, regime)``。

    参数
      window    滚动 Hurst 窗口（bar 数）
      step      每隔多少 bar 重新估计一次 Hurst（DFA 较贵，步长 >1 时前值填充）
      lo / hi   均值回复 / 趋势 的 Hurst 阈值
      hazard    BOCPD 先验变点概率（期望段长 1/hazard）
      cp_window BOCPD "最近多少步内有变点"的窗口
    """

    def __init__(self, window: int = 120, step: int = 5, lo: float = 0.45, hi: float = 0.55, hazard: float = 1 / 120, cp_window: int = 10):
        self.window, self.step, self.lo, self.hi, self.hazard, self.cp_window = int(window), int(step), float(lo), float(hi), float(hazard), int(cp_window)

    def transform(self, market_returns: pd.Series) -> pd.DataFrame:
        r = pd.Series(market_returns).astype(float)
        h = rolling_hurst(r.to_numpy(), self.window, self.step)
        pc = change_probability(r.to_numpy(), self.hazard, self.cp_window)
        return pd.DataFrame({"hurst": h, "prob_change": pc, "regime": label_regimes(h, self.lo, self.hi)}, index=r.index)

    def current(self, market_returns: pd.Series) -> RegimeState:
        df = self.transform(market_returns)
        last = df.iloc[-1]
        return RegimeState(hurst=float(last["hurst"]) if np.isfinite(last["hurst"]) else float("nan"), regime=str(last["regime"]), prob_change=float(last["prob_change"]))

    def describe(self) -> Dict[str, object]:
        return {"window": self.window, "step": self.step, "lo": self.lo, "hi": self.hi, "hazard": self.hazard, "cp_window": self.cp_window}


def market_returns(panel: Dict[str, pd.DataFrame]) -> pd.Series:
    """等权市场收益（面板 ``returns`` 字段的横截面均值；缺失时用 close 计算）。"""
    if "returns" in panel:
        return panel["returns"].mean(axis=1)
    return panel["close"].pct_change(fill_method=None).mean(axis=1)
