"""行情面板数据结构与合成数据生成。

全库统一的数据格式是 ``Panel``：一个 ``dict[str, pd.DataFrame]``，
每个 DataFrame 的 index 为时间（日期或 bar 时间戳），columns 为资产代码，
字段名使用小写：open / high / low / close / volume / vwap / returns。
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

Panel = Dict[str, pd.DataFrame]

BASE_FIELDS = ("open", "high", "low", "close", "volume", "vwap", "returns")


def validate_panel(panel: Panel) -> Panel:
    """检查字段对齐，并补齐 vwap / returns 等派生字段。"""
    if "close" not in panel:
        raise ValueError("panel 至少需要 close 字段")
    ref = panel["close"]
    out: Panel = {}
    for k, df in panel.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"字段 {k} 必须是 DataFrame")
        out[k] = df.reindex(index=ref.index, columns=ref.columns).astype(float)
    if "returns" not in out:
        out["returns"] = out["close"].pct_change(fill_method=None)
    if "vwap" not in out and {"high", "low", "close"} <= out.keys():
        out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0
    return out


def forward_returns(panel: Panel, horizon: int = 1) -> pd.DataFrame:
    """t 时刻的未来 horizon 期收益（close(t+h)/close(t) - 1），用作因子评估的标签。"""
    close = panel["close"]
    return close.shift(-horizon) / close - 1.0


def make_synthetic_panel(
    n_days: int = 500,
    n_assets: int = 100,
    seed: int = 0,
    signal_strength: float = 0.08,
) -> Panel:
    """生成带有可挖掘信号的合成行情，用于测试与演示。

    植入的真实 alpha（挖掘器应能部分找回）：
      1. 短期反转：未来收益与过去 5 日收益负相关；
      2. 量价背离：放量（volume / 20 日均量）的资产未来收益偏低。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    assets = [f"A{i:04d}" for i in range(n_assets)]

    market = rng.normal(0.0003, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_assets)
    idio_vol = rng.uniform(0.01, 0.03, n_assets)
    base_vol = rng.lognormal(13, 0.5, n_assets)

    rets = np.zeros((n_days, n_assets))
    vols = np.zeros((n_days, n_assets))
    vol_shock = rng.lognormal(0, 0.35, (n_days, n_assets))
    for t in range(n_days):
        alpha = np.zeros(n_assets)
        if t >= 20:
            past5 = rets[t - 5 : t].sum(0)
            rev = -_zs(past5)
            vratio = vols[t - 1] / vols[t - 20 : t].mean(0)
            vsig = -_zs(np.log(vratio))
            alpha = signal_strength * idio_vol * (0.7 * rev + 0.5 * vsig)
        rets[t] = beta * market[t] + alpha + idio_vol * rng.standard_normal(n_assets)
        vols[t] = base_vol * vol_shock[t] * (1 + 8 * np.abs(rets[t]))

    close = 50 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    open_ = close / (1 + rets) * (1 + rng.normal(0, 0.002, close.shape))
    spread = np.abs(rng.normal(0, 0.01, close.shape)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    mk = lambda a: pd.DataFrame(a, index=dates, columns=assets)
    return validate_panel(
        {
            "open": mk(open_),
            "high": mk(high),
            "low": mk(low),
            "close": mk(close),
            "volume": mk(vols),
            "returns": mk(rets),
        }
    )


def split_panel(panel: Panel, ratio: float = 0.7) -> tuple[Panel, Panel]:
    """按时间切分训练 / 验证集（不打乱，避免未来信息泄漏）。"""
    cut = int(len(panel["close"].index) * ratio)
    return (
        {k: v.iloc[:cut] for k, v in panel.items()},
        {k: v.iloc[cut:] for k, v in panel.items()},
    )


def _zs(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)
