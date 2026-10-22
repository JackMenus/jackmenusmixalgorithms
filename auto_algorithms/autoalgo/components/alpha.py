"""因子层新算法（实现 quantstack ``AlphaModel`` 协议：``fit`` / ``predict(panel, as_of) -> SignalBatch`` / ``describe``）。

三个模板族都建立在 factormine（表达式求值、IC 评估、权重合成、GP 挖掘）与 crossquant（Hurst/BOCPD、路径签名、
分数阶差分等）之上，只负责"把两边拼起来"：

* ``RegimeGatedAlpha``            —— 市场状态门控的多因子合成：按 Hurst 状态分别估计因子权重，变点概率高时收缩置信度。
* ``SignatureAlpha``              —— 路径签名因子：把 [时间, log 价格, log 成交量] 的窗口路径截断签名（深度 2）作为特征，池化岭回归预测未来收益。
* ``CrossOperatorMinedAlpha``     —— 在扩展了跨学科算子的算子表上做 GP 因子挖掘，再用 factormine 的 max_ir 合成。

约定：``predict`` 只用 ``as_of`` 及之前的数据；期望收益按 Grinold 法则 E[r] = IC·σ·z 给出（单位 bps）。
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .._paths import require
from .regime import REGIMES, RegimeDetector, market_returns

US_PER_TRADING_DAY = 23_400_000_000

DEFAULT_EXPRESSIONS: Dict[str, str] = {
    "rev5": "neg(cs_rank(ts_pct(close, 5)))",
    "vol_ratio": "neg(cs_rank(div(volume, ts_mean(volume, 20))))",
    "mom20": "cs_rank(ts_pct(close, 20))",
    "lowvol": "neg(ts_std(returns, 20))",
}


# ---------------------------------------------------------------------------- 工具
def _as_timestamp(as_of):
    if as_of is None:
        return None
    if isinstance(as_of, (int, np.integer)):
        return pd.Timestamp(int(as_of) * 1000, unit="ns")
    return pd.Timestamp(as_of)


def truncate(panel: Dict[str, pd.DataFrame], as_of) -> Dict[str, pd.DataFrame]:
    """只保留 as_of 及之前的行，从结构上杜绝未来函数。"""
    ts = _as_timestamp(as_of)
    if ts is None:
        return panel
    return {k: v.loc[:ts] for k, v in panel.items()}


def _ts_us(ts) -> int:
    return int(pd.Timestamp(ts).value // 1000)


def cs_z(df: pd.DataFrame, clip: float = 5.0) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0).clip(-clip, clip)


def _idio_vol(panel, window: int = 20) -> pd.Series:
    rets = panel["returns"] if "returns" in panel else panel["close"].pct_change(fill_method=None)
    return rets.iloc[-window:].std().fillna(0.02)


def make_signal_batch(model_id: str, ts, z: pd.Series, er_bps: pd.Series, confidence: pd.Series, horizon_us: int,
                      exposures: Optional[Dict[str, pd.Series]] = None, metadata: Optional[Dict[str, float]] = None, universe_size: int = 0,
                      notes: str = ""):
    """把横截面分数 / 期望收益 / 置信度打包成 quantstack ``SignalBatch``。"""
    require("quantstack")
    from quantstack.contracts import AlphaSignal, SignalBatch

    ts_us = _ts_us(ts)
    exposures = exposures or {}
    metadata = {k: float(v) for k, v in (metadata or {}).items()}
    out = []
    for sym, zi in z.dropna().items():
        expo = {}
        for n, e in exposures.items():
            v = e.get(sym, np.nan)
            if np.isfinite(v):
                expo[n] = float(v)
        out.append(AlphaSignal(
            symbol=str(sym), ts_us=ts_us, horizon_us=int(horizon_us),
            expected_return_bps=float(er_bps.get(sym, 0.0)) if np.isfinite(er_bps.get(sym, np.nan)) else 0.0,
            confidence=float(np.clip(confidence.get(sym, 0.0), 0.0, 1.0)),
            score=float(zi), source=model_id, factor_exposures=expo, metadata=dict(metadata),
        ))
    return SignalBatch(ts_us=ts_us, signals=out, model_id=model_id, universe_size=universe_size or len(out), notes=notes)


# ============================================================================ 1. 市场状态门控多因子
class RegimeGatedAlpha:
    """市场状态门控的多因子合成（AlphaModel）。

    机制
      1. 用 factormine 表达式求值得到 K 个因子；
      2. 用等权市场收益的滚动 DFA-Hurst（crossquant）把每个交易日标为 趋势 / 中性 / 均值回复 三种状态；
      3. 在每个状态的子样本上分别用 factormine ``fit_weights``（默认 max_ir）估计合成权重，样本不足回退到全局权重；
      4. 预测时取当前状态的权重合成分数；用 BOCPD 变点概率 p 把置信度乘以 (1 − shrink·p)：状态刚切换时少下注。

    参数
      expressions     因子表达式（factormine 语法），默认 4 个经典因子
      hurst_window    滚动 Hurst 窗口；lo / hi：均值回复 / 趋势阈值
      hazard          BOCPD 先验变点概率
      method          合成方法 equal / ic / icir / max_ir / ridge
      shrink_on_change 变点概率对置信度的收缩强度 ∈ [0, 1]
      min_regime_obs  某状态样本少于该值时使用全局权重
    """

    def __init__(self, expressions: Optional[Dict[str, str]] = None, hurst_window: int = 120, hurst_step: int = 5, lo: float = 0.45, hi: float = 0.55,
                 hazard: float = 1 / 120, cp_window: int = 10, method: str = "max_ir", horizon: int = 1, shrink_on_change: float = 0.5,
                 min_regime_obs: int = 40, bar_us: int = US_PER_TRADING_DAY, model_id: str = "autoalgo.regime_gated_alpha"):
        self.expressions = dict(expressions or DEFAULT_EXPRESSIONS)
        self.detector = RegimeDetector(window=hurst_window, step=hurst_step, lo=lo, hi=hi, hazard=hazard, cp_window=cp_window)
        self.method, self.horizon, self.bar_us = method, int(horizon), int(bar_us)
        self.shrink_on_change = float(np.clip(shrink_on_change, 0.0, 1.0))
        self.min_regime_obs = int(min_regime_obs)
        self.model_id = model_id
        self.weights_: Dict[str, pd.Series] = {}
        self.ic_: Dict[str, float] = {}
        self.regime_counts_: Dict[str, int] = {}
        self.global_weights_: Optional[pd.Series] = None
        self.last_state_: Dict[str, float] = {}

    # ---------------------------------------------------------------- 内部
    def _factors(self, panel) -> Dict[str, pd.DataFrame]:
        require("factormine")
        from factormine import Evaluator, parse

        ev = Evaluator(panel)
        return {n: ev(parse(e)) for n, e in self.expressions.items()}

    # ---------------------------------------------------------------- 协议
    def fit(self, panel, **kwargs) -> None:
        require("factormine")
        from factormine import combine, fit_weights, forward_returns, ic_series

        facs = self._factors(panel)
        fwd = forward_returns(panel, self.horizon)
        reg = self.detector.transform(market_returns(panel))
        self.global_weights_ = fit_weights(facs, fwd, method=self.method)
        g_ic = float(ic_series(combine(facs, self.global_weights_), fwd).mean())
        self.weights_, self.ic_, self.regime_counts_ = {}, {}, {}
        for r in REGIMES:
            mask = (reg["regime"] == r).to_numpy()
            self.regime_counts_[r] = int(mask.sum())
            if mask.sum() >= self.min_regime_obs:
                sub = {n: f[mask] for n, f in facs.items()}
                try:
                    w = fit_weights(sub, fwd[mask], method=self.method)
                    ic = float(ic_series(combine(sub, w), fwd[mask]).mean())
                except Exception:
                    w, ic = self.global_weights_, g_ic
            else:
                w, ic = self.global_weights_, g_ic
            self.weights_[r] = w
            self.ic_[r] = ic if np.isfinite(ic) else 0.0
        self.ic_["global"] = g_ic

    def predict(self, panel, as_of=None, **kwargs):
        if not self.weights_:
            self.fit(panel)
        from factormine import combine

        hist = truncate(panel, as_of)
        facs = self._factors(hist)
        state = self.detector.current(market_returns(hist))
        w = self.weights_.get(state.regime, self.global_weights_)
        z = combine(facs, w).iloc[-1]
        ic = self.ic_.get(state.regime, self.ic_.get("global", 0.0))
        sigma = _idio_vol(hist) * np.sqrt(self.horizon)
        er = ic * sigma * z * 1e4
        conf = (np.clip(abs(ic) / 0.1, 0.0, 1.0) * (1.0 - self.shrink_on_change * state.prob_change)) * (z.abs() / 2.0).clip(upper=1.0)
        expo = {n: f.iloc[-1] for n, f in facs.items()}
        regime_code = float(REGIMES.index(state.regime))
        self.last_state_ = {"hurst": state.hurst, "prob_change": state.prob_change, "regime_code": regime_code}
        meta = {"ic": ic, "half_life_bars": 3.0, "hurst": state.hurst if np.isfinite(state.hurst) else 0.5, "prob_change": state.prob_change, "regime_code": regime_code}
        ts = hist["close"].index[-1]
        return make_signal_batch(self.model_id, ts, z, er, conf, self.horizon * self.bar_us, expo, meta, hist["close"].shape[1],
                                 notes=f"regime={state.regime} H={state.hurst:.3f} p_change={state.prob_change:.2f}")

    def score_panel(self, panel) -> pd.DataFrame:
        """全样本一次性出分（每行只用该行及之前的数据，与逐日 ``predict(as_of).score`` 等价；供评估台加速）。"""
        if not self.weights_:
            self.fit(panel)
        from factormine import combine

        facs = self._factors(panel)
        reg = self.detector.transform(market_returns(panel))
        out = pd.DataFrame(np.nan, index=panel["close"].index, columns=panel["close"].columns)
        for r in REGIMES:
            mask = (reg["regime"] == r).to_numpy()
            if mask.any():
                comb = combine(facs, self.weights_.get(r, self.global_weights_))
                out[mask] = comb[mask]
        return out

    def describe(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id, "library": "autoalgo", "family": "alpha.regime_gated",
            "expressions": dict(self.expressions), "method": self.method, "horizon": self.horizon,
            "detector": self.detector.describe(), "shrink_on_change": self.shrink_on_change,
            "weights": {r: {k: float(v) for k, v in w.items()} for r, w in self.weights_.items()},
            "ic": dict(self.ic_), "regime_counts": dict(self.regime_counts_), "last_state": dict(self.last_state_),
        }


# ============================================================================ 2. 路径签名因子
def signature_features(paths: np.ndarray, window: int, normalize: bool = True, chunk: int = 64) -> np.ndarray:
    """窗口路径的深度-2 截断签名，向量化实现（与 ``crossquant.mathematics.signatures.signature`` 等价）。

    paths: (T, N, d) 的路径点；返回 (T, N, d + d²)，前 window−1 行为 NaN。
    对分段线性路径，Chen 恒等式给出 S¹ = Σ_k Δ_k，S² = Σ_k (Σ_{i<k} Δ_i) ⊗ Δ_k + ½ Σ_k Δ_k ⊗ Δ_k。
    normalize=True 时每个通道的增量先除以窗口内标准差（只保留"形状"信息，去掉尺度）。
    """
    T, N, d = paths.shape
    m = d + d * d
    out = np.full((T, N, m), np.nan)
    w1 = window - 1
    if T - 1 < w1 or w1 < 1:
        return out
    D = np.diff(paths, axis=0)  # (T-1, N, d)
    for s in range(0, N, chunk):
        Dc = np.nan_to_num(D[:, s:s + chunk])                                # (T-1, n, d)
        Wn = np.lib.stride_tricks.sliding_window_view(Dc, w1, axis=0)         # (T-w1, n, d, w1)
        Wn = np.transpose(Wn, (0, 1, 3, 2))                                   # (T-w1, n, w1, d)
        if normalize:
            sd = Wn.std(axis=2, keepdims=True)
            Wn = Wn / np.where(sd > 0, sd, 1.0)
        S1 = Wn.sum(axis=2)                                                   # (T-w1, n, d)
        cum_prev = np.cumsum(Wn, axis=2) - Wn
        S2 = np.einsum("tnki,tnkj->tnij", cum_prev, Wn) + 0.5 * np.einsum("tnki,tnkj->tnij", Wn, Wn)
        feats = np.concatenate([S1, S2.reshape(S2.shape[0], S2.shape[1], d * d)], axis=-1)
        out[w1:, s:s + chunk, :] = feats
    return out


class SignatureAlpha:
    """路径签名因子（AlphaModel）。

    机制
      1. 每个资产在长度 window 的窗口内构造路径 X = [t, log(close), log(volume)]（crossquant 的时间增广约定）；
      2. 计算深度-2 截断签名：一阶项是各通道净变化，二阶项（含 Lévy 面积）刻画通道间的领先-滞后与路径弯曲；
      3. 特征在每个截面做 z 标准化后，与标准化的未来收益做池化岭回归（Gu–Kelly–Xiu 线性基准的做法）；
      4. 预测分数 = 当前截面特征 · β，再做横截面标准化。

    参数
      window     路径窗口长度（bar）
      ridge      岭回归惩罚（相对样本量）
      channels   通道选择：("logprice", "logvolume") 子集，时间通道总是加入
      normalize  是否对每个通道的增量按窗口标准差归一（只保留形状）
    """

    FEATURE_CHANNELS = ("time", "logprice", "logvolume")

    def __init__(self, window: int = 20, ridge: float = 1.0, channels: Sequence[str] = ("logprice", "logvolume"), normalize: bool = True,
                 horizon: int = 1, bar_us: int = US_PER_TRADING_DAY, model_id: str = "autoalgo.signature_alpha"):
        self.window = int(max(window, 4))
        self.ridge = float(ridge)
        self.channels = tuple(c for c in channels if c in ("logprice", "logvolume")) or ("logprice",)
        self.normalize = bool(normalize)
        self.horizon, self.bar_us, self.model_id = int(horizon), int(bar_us), model_id
        self.beta_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.ic_: float = 0.0
        self.n_train_: int = 0

    def _paths(self, panel) -> np.ndarray:
        close = panel["close"].to_numpy(float)
        chans = [np.broadcast_to(np.arange(close.shape[0], dtype=float)[:, None] / max(self.window - 1, 1), close.shape)]
        if "logprice" in self.channels:
            chans.append(np.log(np.where(close > 0, close, np.nan)))
        if "logvolume" in self.channels:
            vol = panel["volume"].to_numpy(float)
            chans.append(np.log1p(np.where(vol > 0, vol, 0.0)))
        return np.stack(chans, axis=-1)

    def _features(self, panel) -> np.ndarray:
        F = signature_features(self._paths(panel), self.window, self.normalize)
        # 截面标准化（逐日逐特征）；窗口预热段全为 NaN，直接置 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(F, axis=1, keepdims=True)
            sd = np.nanstd(F, axis=1, keepdims=True)
        Z = (F - mu) / np.where(sd > 0, sd, 1.0)
        return np.clip(np.nan_to_num(Z, nan=0.0), -5, 5)

    def fit(self, panel, **kwargs) -> None:
        close = panel["close"]
        F = self._features(panel)
        T, N, m = F.shape
        fwd = (close.shift(-self.horizon) / close - 1.0)
        y = cs_z(fwd).to_numpy(float)
        valid = np.isfinite(y) & (np.arange(T)[:, None] >= self.window - 1)
        X = F[valid]
        yv = y[valid]
        lam = self.ridge * max(len(yv), 1) * 1e-3
        self.beta_ = np.linalg.solve(X.T @ X + lam * np.eye(m), X.T @ yv)
        self.n_train_ = int(len(yv))
        d = len(self.channels) + 1
        names = list(self.FEATURE_CHANNELS[:1]) + [c for c in self.FEATURE_CHANNELS[1:] if c in self.channels]
        self.feature_names_ = [f"S1[{a}]" for a in names] + [f"S2[{a},{b}]" for a in names for b in names]
        scores = pd.DataFrame(F @ self.beta_, index=close.index, columns=close.columns)
        ics = cs_z(scores).corrwith(fwd, axis=1, method="spearman").dropna()
        self.ic_ = float(ics.mean()) if len(ics) else 0.0

    def predict(self, panel, as_of=None, **kwargs):
        if self.beta_ is None:
            self.fit(panel)
        hist = truncate(panel, as_of)
        close = hist["close"]
        F = self._features(hist)
        raw = pd.Series(F[-1] @ self.beta_, index=close.columns)
        z = ((raw - raw.mean()) / (raw.std() if raw.std() > 0 else 1.0)).clip(-5, 5)
        sigma = _idio_vol(hist) * np.sqrt(self.horizon)
        er = self.ic_ * sigma * z * 1e4
        conf = np.clip(abs(self.ic_) / 0.1, 0, 1) * (z.abs() / 2.0).clip(upper=1.0)
        expo = {n: pd.Series(F[-1][:, i], index=close.columns) for i, n in enumerate(self.feature_names_)}
        meta = {"ic": self.ic_, "half_life_bars": 3.0, "window": float(self.window)}
        return make_signal_batch(self.model_id, close.index[-1], z, er, conf, self.horizon * self.bar_us, expo, meta, close.shape[1],
                                 notes=f"signature depth=2 window={self.window}")

    def score_panel(self, panel) -> pd.DataFrame:
        """全样本一次性出分（窗口特征只用历史，与逐日 predict 等价）。"""
        if self.beta_ is None:
            self.fit(panel)
        close = panel["close"]
        raw = pd.DataFrame(self._features(panel) @ self.beta_, index=close.index, columns=close.columns)
        raw.iloc[: self.window - 1] = np.nan
        return cs_z(raw)

    def describe(self) -> Dict[str, object]:
        return {
            "model_id": self.model_id, "library": "autoalgo", "family": "alpha.signature", "window": self.window, "ridge": self.ridge,
            "channels": list(self.channels), "normalize": self.normalize, "horizon": self.horizon, "n_train": self.n_train_,
            "in_sample_rank_ic": self.ic_, "beta": {n: float(b) for n, b in zip(self.feature_names_, self.beta_)} if self.beta_ is not None else {},
        }


# ============================================================================ 3. 跨学科算子 GP 挖掘
BASE_OPERATORS = ("add", "sub", "mul", "div", "neg", "abs", "log", "sign", "delay", "delta", "ts_pct", "ts_mean", "ts_std", "ts_zscore",
                  "ts_rank", "ts_corr", "ts_ema", "cs_rank", "cs_zscore", "cs_demean")


class CrossOperatorMinedAlpha:
    """在"基础算子 + 跨学科算子"上做遗传规划因子挖掘，再合成为 AlphaModel。

    机制
      1. ``components.operators.register_all`` 把 6 个 crossquant 算子注入 factormine 算子表；
      2. ``factormine.mine_factors``（GP 方法，种群 = max(20, budget)，代数 = max(2, budget // 20)）在训练段挖掘，
         样本外过滤后进入因子库；
      3. ``factormine.AlphaModel(library, method)`` 估计合成权重，``signal`` 给出分数与期望收益；
      4. ``describe`` 统计因子库中使用了跨学科算子的比例，作为"跨学科贡献度"。

    参数
      budget        计算预算（≈ 候选评估次数 / 10）
      seed          随机种子（GP 可复现）
      method        合成方法
      extended_only 为 True 时算子表只保留基础算子 + 扩展算子（默认），False 则用 factormine 全部算子
      seeds         种子表达式（可含扩展算子），保证池不为空
    """

    DEFAULT_SEEDS = (
        "neg(cs_rank(ts_kalman_resid(close, 0.1)))",
        "cs_rank(ts_levy_area(close, volume, 20))",
        "neg(cs_rank(ts_pct(close, 5)))",
        "mul(neg(cs_rank(ts_pct(close, 5))), cs_rank(ts_vr(returns, 60)))",
    )

    def __init__(self, budget: int = 40, seed: int = 0, method: str = "max_ir", horizon: int = 1, extended_only: bool = True,
                 pool_capacity: int = 12, max_corr: float = 0.7, min_ic: float = 0.01, seeds: Optional[Sequence[str]] = None,
                 windows: Sequence[int] = (5, 10, 20, 40, 60), bar_us: int = US_PER_TRADING_DAY, model_id: str = "autoalgo.cross_operator_mined_alpha"):
        self.budget, self.seed, self.method, self.horizon = int(budget), int(seed), method, int(horizon)
        self.extended_only, self.pool_capacity, self.max_corr, self.min_ic = bool(extended_only), int(pool_capacity), float(max_corr), float(min_ic)
        self.seeds = tuple(seeds) if seeds is not None else self.DEFAULT_SEEDS
        self.windows = tuple(int(w) for w in windows)
        self.bar_us, self.model_id = int(bar_us), model_id
        self.library = None
        self._model = None
        self.report_: Optional[pd.DataFrame] = None

    def fit(self, panel, **kwargs) -> None:
        require("factormine")
        import factormine as fm
        from factormine import FactorLibrary, FactorRecord, GPConfig, MiningConfig, mine_factors

        from .operators import EXTENDED_OPERATORS, register_all

        register_all()
        ops = list(BASE_OPERATORS) + list(EXTENDED_OPERATORS) if self.extended_only else None
        b = max(self.budget, 1)
        cfg = MiningConfig(methods=("gp",), horizon=self.horizon, pool_capacity=self.pool_capacity, max_corr=self.max_corr, min_ic=self.min_ic,
                           gp=GPConfig(population=max(20, b), generations=max(2, b // 20), seed=self.seed, operators=ops, windows=self.windows, submit_top=8))
        lib = FactorLibrary()
        lib, self.report_ = mine_factors(panel, cfg, library=lib, seeds=self.seeds, log=lambda s: None)
        if len(lib) == 0:  # 全部未通过样本外过滤：退回种子表达式，保证模型可用
            lib.extend([FactorRecord(name=f"seed_{i:02d}", expression=e, source="seed", rationale="兜底种子因子") for i, e in enumerate(self.seeds)])
        self.library = lib
        self._model = fm.AlphaModel(lib, method=self.method, horizon=self.horizon).fit(panel)

    def predict(self, panel, as_of=None, **kwargs):
        if self._model is None:
            self.fit(panel)
        from factormine import standardize

        hist = truncate(panel, as_of)
        sig = self._model.signal(hist)
        z = sig.scores
        er = sig.expected_return * 1e4
        conf = np.clip(abs(sig.ic) / 0.1, 0, 1) * (z.abs() / 2.0).clip(upper=1.0)
        facs = self._model._factors(hist)
        expo = {n: standardize(f.iloc[-60:]).iloc[-1] for n, f in facs.items()}
        hl = float(sig.half_life) if np.isfinite(sig.half_life) else 3.0
        meta = {"ic": float(sig.ic), "half_life_bars": hl, "urgency": float(sig.urgency)}
        return make_signal_batch(self.model_id, sig.timestamp, z, er, conf, self.horizon * self.bar_us, expo, meta, hist["close"].shape[1],
                                 notes=f"{len(self.library)} 个 GP 因子，跨学科算子占比 {self.extended_share():.0%}")

    def score_panel(self, panel) -> pd.DataFrame:
        """全样本一次性出分（factormine.AlphaModel.scores：时序算子只用历史，截面标准化逐行进行）。"""
        if self._model is None:
            self.fit(panel)
        return self._model.scores(panel)

    def extended_share(self) -> float:
        """因子库中含跨学科算子的因子比例。"""
        if not self.library or len(self.library) == 0:
            return 0.0
        from .operators import EXTENDED_OPERATORS

        n = sum(any(op + "(" in r.expression for op in EXTENDED_OPERATORS) for r in self.library.records.values())
        return n / len(self.library)

    def describe(self) -> Dict[str, object]:
        m = self._model
        return {
            "model_id": self.model_id, "library": "autoalgo", "family": "alpha.cross_operator_mined", "budget": self.budget, "seed": self.seed,
            "method": self.method, "horizon": self.horizon, "extended_only": self.extended_only,
            "n_factors": 0 if self.library is None else len(self.library),
            "factors": {} if self.library is None else {n: r.expression for n, r in self.library.records.items()},
            "weights": {} if m is None else {k: float(v) for k, v in m.weights.items()},
            "in_sample_rank_ic": None if m is None else float(m.ic_), "extended_share": self.extended_share(),
        }
