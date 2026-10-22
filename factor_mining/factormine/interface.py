"""对外接口：因子库 → 交易信号。这是挂单策略库、路由延迟优化库与全栈 agent 对接的唯一契约。

数据流::

    行情 Panel ──► AlphaModel.signal() ──► AlphaSignal ──► 挂单库（决定下单价格/节奏）
                                                     └──► 路由库（按紧迫度选择通道）

``AlphaSignal`` 字段都是普通 Python / pandas 类型，``to_dict()`` 可直接 JSON 序列化走消息总线。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from .combine import combine, fit_weights
from .data import Panel, forward_returns
from .evaluate import ic_decay, ic_series
from .expression import Evaluator, parse
from .library import FactorLibrary

SCHEMA_VERSION = "factormine.signal/1"


@dataclass
class AlphaSignal:
    """某一时刻的横截面 alpha 信号。

    * scores           —— 横截面 z-score（均值 0、标准差 1），正 = 看多；
    * expected_return  —— Grinold 公式 E[r] ≈ IC × σ × z，单位为 horizon 期收益；
    * ic               —— 合成信号的样本内 RankIC 估计（信号强度/置信度）；
    * horizon          —— 预测持有期（bar 数）；
    * half_life        —— 信号半衰期（bar 数），执行层据此决定“急不急”：
                          半衰期短 → 更激进的挂单/更低延迟的路由；
    * urgency          —— 0~1 的执行紧迫度（由 half_life 推出），供挂单/路由直接使用。
    """

    timestamp: pd.Timestamp
    scores: pd.Series
    expected_return: pd.Series
    ic: float
    horizon: int
    half_life: float
    urgency: float
    meta: Dict[str, object] = field(default_factory=dict)

    def target_weights(
        self,
        gross: float = 1.0,
        mode: Literal["long_short", "long_only"] = "long_short",
        max_weight: float = 0.05,
        top_frac: Optional[float] = None,
    ) -> pd.Series:
        """把得分转换成目标权重（挂单库的输入）。"""
        s = self.scores.dropna()
        if mode == "long_short":
            s = s - s.mean()
        if top_frac is not None:
            k = max(1, int(len(s) * top_frac))
            keep = s.rank(ascending=False) <= k
            if mode == "long_short":
                keep |= s.rank() <= k
            s = s.where(keep, 0.0)
        if mode == "long_only":
            return _cap_scale(s.clip(lower=0), gross, max_weight)
        # 多空两侧各占一半总杠杆，分别截断，保证净敞口为 0
        long = _cap_scale(s.clip(lower=0), gross / 2, max_weight)
        short = _cap_scale((-s).clip(lower=0), gross / 2, max_weight)
        return long - short

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_VERSION,
            "timestamp": str(self.timestamp),
            "scores": {k: float(v) for k, v in self.scores.dropna().items()},
            "expected_return": {k: float(v) for k, v in self.expected_return.dropna().items()},
            "ic": self.ic,
            "horizon": self.horizon,
            "half_life": self.half_life,
            "urgency": self.urgency,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "AlphaSignal":
        return AlphaSignal(
            timestamp=pd.Timestamp(d["timestamp"]),
            scores=pd.Series(d["scores"], dtype=float),
            expected_return=pd.Series(d["expected_return"], dtype=float),
            ic=d["ic"], horizon=d["horizon"], half_life=d["half_life"], urgency=d["urgency"],
            meta=d.get("meta", {}),
        )


class AlphaModel:
    """把因子库中的若干因子合成为一个可上线的信号模型。

    用法::

        model = AlphaModel(library, method="max_ir").fit(history_panel)
        sig = model.signal(live_panel)          # 最新时刻的 AlphaSignal
        weights = sig.target_weights(gross=1.0)
    """

    def __init__(self, library: FactorLibrary, names: Optional[List[str]] = None,
                 method: str = "max_ir", horizon: int = 1, vol_window: int = 20):
        self.library = library
        self.names = names or list(library.records)
        self.method = method
        self.horizon = horizon
        self.vol_window = vol_window
        self.weights: Optional[pd.Series] = None
        self.ic_: float = np.nan
        self.half_life_: float = np.nan

    def _factors(self, panel: Panel) -> Dict[str, pd.DataFrame]:
        ev = Evaluator(panel)
        return {n: ev(parse(self.library.records[n].expression)) for n in self.names}

    def fit(self, panel: Panel) -> "AlphaModel":
        facs = self._factors(panel)
        fwd = forward_returns(panel, self.horizon)
        self.weights = fit_weights(facs, fwd, method=self.method)
        comb = combine(facs, self.weights)
        self.ic_ = float(ic_series(comb, fwd).mean())
        decay = ic_decay(comb, panel["close"], horizons=(1, 2, 3, 5, 10, 20), marginal=True).abs()
        self.half_life_ = _half_life(decay)
        return self

    def scores(self, panel: Panel) -> pd.DataFrame:
        if self.weights is None:
            raise RuntimeError("请先调用 fit()")
        return combine(self._factors(panel), self.weights)

    def signal(self, panel: Panel, at: Optional[pd.Timestamp] = None) -> AlphaSignal:
        sc = self.scores(panel)
        at = sc.index[-1] if at is None else pd.Timestamp(at)
        z = sc.loc[at]
        sigma = panel["returns"].rolling(self.vol_window).std().loc[at] * np.sqrt(self.horizon)
        er = self.ic_ * sigma * z
        return AlphaSignal(
            timestamp=at, scores=z, expected_return=er, ic=self.ic_, horizon=self.horizon,
            half_life=self.half_life_, urgency=float(np.clip(1.0 / max(self.half_life_, 1.0), 0, 1)),
            meta={"method": self.method, "factors": self.names,
                  "weights": {k: float(v) for k, v in self.weights.items()}},
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "names": self.names, "method": self.method, "horizon": self.horizon,
            "vol_window": self.vol_window, "weights": self.weights.to_dict(),
            "ic": self.ic_, "half_life": self.half_life_,
        }, ensure_ascii=False, indent=2))

    @staticmethod
    def load(path: str | Path, library: FactorLibrary) -> "AlphaModel":
        d = json.loads(Path(path).read_text())
        m = AlphaModel(library, d["names"], d["method"], d["horizon"], d["vol_window"])
        m.weights = pd.Series(d["weights"], dtype=float)
        m.ic_, m.half_life_ = d["ic"], d["half_life"]
        return m


def _cap_scale(s: pd.Series, total: float, cap: float) -> pd.Series:
    """把非负得分按比例缩放到总和 total，单票不超过 cap（注水法）。"""
    w = s * 0.0
    free = s > 0
    remaining = total
    for _ in range(len(s) + 1):
        if not free.any() or remaining <= 0:
            break
        trial = s[free] / s[free].sum() * remaining
        over = trial > cap
        if not over.any():
            w[free] = trial
            break
        idx = trial[over].index
        w[idx] = cap
        free[idx] = False
        remaining = total - w.sum()
    return w


def _half_life(decay: pd.Series) -> float:
    """IC 衰减到 1 期 IC 一半时对应的持有期（线性插值）。"""
    if decay.empty or not np.isfinite(decay.iloc[0]) or decay.iloc[0] <= 0:
        return float("nan")
    half = decay.iloc[0] / 2
    hs, vs = list(decay.index), list(decay.values)
    for i in range(1, len(hs)):
        if vs[i] <= half:
            h0, h1, v0, v1 = hs[i - 1], hs[i], vs[i - 1], vs[i]
            return float(h0 + (v0 - half) / (v0 - v1) * (h1 - h0)) if v0 != v1 else float(h1)
    return float(hs[-1])
