"""参考 alpha 模型：短期反转 + 量价背离（合成面板中植入的两类信号）。

因子挖掘库交付后由 ``factormine.make_alpha_model`` 替换；本实现演示 ``AlphaModel`` 协议的正确用法：
``fit`` 用历史估计「分数 -> 期望收益」的标度与 IC，``predict`` 只使用 ``as_of`` 及之前的数据。
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..contracts import AlphaSignal, Panel, SignalBatch

DAY_US = 86_400_000_000


class ReversalVolumeAlpha:
    model_id = "ref_reversal_volume"

    def __init__(self, lookback: int = 5, vol_window: int = 20, horizon: int = 1, horizon_us: int = 23_400_000_000):
        self.lookback = lookback
        self.vol_window = vol_window
        self.horizon = horizon
        self.horizon_us = horizon_us
        self.ic_mean = 0.0
        self.ic_ir = 0.0
        self.coef_bps = 0.0   # 每单位分数对应的期望收益（bps）
        self._fitted = False

    # ------------------------------------------------------------------ 内部
    def _scores(self, panel: Panel) -> pd.DataFrame:
        close, volume = panel["close"], panel["volume"]
        rets = close.pct_change(fill_method=None)
        past = rets.rolling(self.lookback).sum()
        vratio = np.log(volume / volume.rolling(self.vol_window).mean())
        rev = -_cs_z(past)
        vsig = -_cs_z(vratio)
        return _cs_z(0.7 * rev + 0.5 * vsig)

    # ------------------------------------------------------------------ 协议
    def fit(self, panel: Panel, **kwargs) -> None:
        scores = self._scores(panel)
        fwd = panel["close"].shift(-self.horizon) / panel["close"] - 1.0
        ics = scores.corrwith(fwd, axis=1).dropna()
        self.ic_mean = float(ics.mean()) if len(ics) else 0.0
        self.ic_ir = float(ics.mean() / ics.std()) if len(ics) > 1 and ics.std() > 0 else 0.0
        # Grinold：E[r] ≈ IC · σ · z
        sigma = float(panel["close"].pct_change(fill_method=None).stack().std())
        self.coef_bps = self.ic_mean * sigma * 1e4
        self._fitted = True

    def predict(self, panel: Panel, as_of=None, **kwargs) -> SignalBatch:
        close = panel["close"]
        if as_of is not None:
            if isinstance(as_of, (int, np.integer)):
                as_of = pd.Timestamp(int(as_of) * 1000, unit="ns")
            panel = {k: v.loc[:as_of] for k, v in panel.items()}
            close = panel["close"]
        if not self._fitted:
            self.fit(panel)
        scores = self._scores(panel).iloc[-1]
        ts = close.index[-1]
        ts_us = int(pd.Timestamp(ts).value // 1000)
        confidence = float(min(abs(self.ic_ir) / 0.5, 1.0)) if self.ic_ir else 0.3
        rets = close.pct_change(fill_method=None).iloc[-self.vol_window:]
        idio = rets.std()
        signals = []
        for sym, z in scores.dropna().items():
            signals.append(
                AlphaSignal(
                    symbol=str(sym),
                    ts_us=ts_us,
                    horizon_us=self.horizon_us,
                    expected_return_bps=float(self.coef_bps * z),
                    confidence=confidence * float(min(abs(z) / 2.0, 1.0)),
                    score=float(z),
                    source=self.model_id,
                    factor_exposures={"rev5": float(z)},
                    metadata={"idio_vol": float(idio.get(sym, 0.02)), "ic": self.ic_mean, "half_life_bars": 3.0, "urgency": 0.33},
                )
            )
        return SignalBatch(ts_us=ts_us, signals=signals, model_id=self.model_id, universe_size=close.shape[1])

    def describe(self) -> Dict[str, object]:
        return {"model_id": self.model_id, "lookback": self.lookback, "vol_window": self.vol_window, "ic_mean": self.ic_mean, "ic_ir": self.ic_ir, "coef_bps": self.coef_bps}


def _cs_z(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)
