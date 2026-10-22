"""与全栈交易系统（quantstack）的契约适配层。

实现 ``quantstack.contracts.protocols`` 中的两个协议（结构化子类型，无需继承）：
  * ``ContractAlphaModel``  —— AlphaModel：fit / predict(as_of) -> SignalBatch / describe
  * ``ContractFactorMiner`` —— FactorMiner：mine(panel, budget) -> [FactorSpec] / evaluate -> FactorReport

若环境中能 import ``quantstack.contracts``，返回的就是那边的 dataclass；否则返回同字段的 dict，
保证 factormine 本身不依赖集成层。工厂函数 ``make_alpha_model`` / ``make_miner`` 在包根导出。
"""
from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .combine import standardize
from .data import Panel, forward_returns
from .evaluate import evaluate_factor
from .expression import Evaluator, parse
from .gp import GPConfig
from .interface import AlphaModel, _half_life
from .evaluate import ic_decay
from .library import FactorLibrary, FactorRecord
from .llm import LLMClient, LLMConfig
from .pipeline import MiningConfig, mine_factors
from .rl import RLConfig

try:  # 可选依赖：集成层契约
    from quantstack.contracts import AlphaSignal as _QSignal
    from quantstack.contracts import FactorReport as _QReport
    from quantstack.contracts import FactorSpec as _QSpec
    from quantstack.contracts import SignalBatch as _QBatch
except Exception:  # pragma: no cover
    _QSignal = _QReport = _QSpec = _QBatch = None

US_PER_TRADING_DAY = 23_400_000_000  # 6.5 小时
_FAMILY = {"gp": "gp", "rl": "rl", "llm": "llm", "neural": "dl", "dl": "dl", "seed": "handcrafted", "manual": "handcrafted"}


def _mk(cls, **kw):
    return cls(**kw) if cls is not None else kw


def _ts_us(ts) -> int:
    return int(pd.Timestamp(ts).value // 1000)


def _truncate(panel: Panel, as_of) -> Panel:
    """只保留 as_of 及之前的数据，从结构上杜绝未来函数。"""
    if as_of is None:
        return panel
    as_of = pd.Timestamp(as_of) if not isinstance(as_of, (int, np.integer)) else pd.Timestamp(int(as_of) * 1000)
    return {k: v.loc[:as_of] for k, v in panel.items()}


def record_to_spec(rec: FactorRecord, horizon: int = 1):
    return _mk(
        _QSpec, name=rec.name, expression=rec.expression, family=_FAMILY.get(rec.source, "handcrafted"),
        params={}, source_library="factormine", horizon=horizon,
        created_at=_dt.datetime.fromtimestamp(rec.created_at, _dt.timezone.utc).isoformat(),
        notes=rec.rationale,
    )


class ContractAlphaModel:
    """quantstack AlphaModel 协议实现。

    参数
      library       : FactorLibrary 或 JSON 路径；为空时 fit() 会先自动挖掘（mine_budget 控成本）
      method        : 合成方法 equal / ic / icir / max_ir / ridge
      horizon       : 预测期（bar 数）
      bar_us        : 一个 bar 的微秒数（日频默认 23_400_000_000）
    """

    def __init__(self, library: Optional[FactorLibrary | str] = None, method: str = "max_ir", horizon: int = 1,
                 bar_us: int = US_PER_TRADING_DAY, model_id: Optional[str] = None, mine_budget: int = 100,
                 seed: int = 0, llm: Optional[LLMClient] = None, factor_names: Optional[List[str]] = None):
        self.library = library if isinstance(library, FactorLibrary) else FactorLibrary(library)
        self.method, self.horizon, self.bar_us = method, horizon, bar_us
        self.model_id = model_id or f"factormine.{method}.h{horizon}"
        self.mine_budget, self.seed, self.llm = mine_budget, seed, llm
        self.factor_names = factor_names
        self._model: Optional[AlphaModel] = None

    def fit(self, panel: Panel, **kwargs) -> None:
        if len(self.library) == 0:
            miner = ContractFactorMiner(seed=self.seed, llm=self.llm, horizon=self.horizon, library=self.library)
            miner.mine(panel, budget=kwargs.get("budget", self.mine_budget))
        self._model = AlphaModel(self.library, self.factor_names, self.method, self.horizon).fit(panel)

    def predict(self, panel: Panel, as_of=None, **kwargs):
        if self._model is None:
            raise RuntimeError("请先调用 fit()")
        hist = _truncate(panel, as_of)
        sig = self._model.signal(hist)
        ts = _ts_us(sig.timestamp)
        facs = self._model._factors(hist)
        expo = {n: standardize(f.iloc[-60:]).iloc[-1] for n, f in facs.items()}  # 各因子当期横截面 z
        base_conf = float(np.clip(abs(sig.ic) / 0.1, 0.0, 1.0))  # |IC|=0.1 视为满置信
        signals = []
        for sym, z in sig.scores.dropna().items():
            er = sig.expected_return.get(sym, np.nan)
            signals.append(_mk(
                _QSignal, symbol=str(sym), ts_us=ts, horizon_us=int(self.horizon * self.bar_us),
                expected_return_bps=float(er * 1e4) if np.isfinite(er) else 0.0,
                confidence=base_conf * float(min(abs(z) / 2.0, 1.0)),
                score=float(z), source=self.model_id,
                factor_exposures={n: float(e.get(sym, np.nan)) for n, e in expo.items() if np.isfinite(e.get(sym, np.nan))},
                metadata={"half_life_bars": float(sig.half_life), "urgency": float(sig.urgency), "ic": float(sig.ic)},
            ))
        return _mk(_QBatch, ts_us=ts, signals=signals, model_id=self.model_id, universe_size=len(signals),
                   notes=f"half_life={sig.half_life:.2f} bars, urgency={sig.urgency:.2f}")

    def describe(self) -> Dict[str, object]:
        m = self._model
        return {
            "model_id": self.model_id, "library": "factormine", "method": self.method, "horizon": self.horizon,
            "n_factors": len(self.library),
            "factors": {n: r.expression for n, r in self.library.records.items()},
            "weights": {} if m is None else {k: float(v) for k, v in m.weights.items()},
            "in_sample_rank_ic": None if m is None else m.ic_,
            "half_life_bars": None if m is None else m.half_life_,
        }


class ContractFactorMiner:
    """quantstack FactorMiner 协议实现。

    ``budget`` 是粗粒度的计算量旋钮（约等于“候选评估次数 / 10”）：
      GP 种群 = max(20, budget)、代数 = max(2, budget // 20)；RL 轮数 = max(2, budget // 10)；
      LLM 轮数 = max(1, budget // 50)（仅当提供了 llm 客户端）。budget=100 在 500×100 日频面板上约 1~2 分钟。
    """

    def __init__(self, methods: Sequence[str] = ("gp", "rl", "llm"), seed: int = 0, llm: Optional[LLMClient] = None,
                 horizon: int = 1, train_ratio: float = 0.7, library: Optional[FactorLibrary] = None):
        self.methods, self.seed, self.llm = tuple(methods), seed, llm
        self.horizon, self.train_ratio = horizon, train_ratio
        self.library = library if library is not None else FactorLibrary()
        self.last_report: Optional[pd.DataFrame] = None

    def mine(self, panel: Panel, budget: int = 100, seeds: Sequence[str] = (), **kwargs) -> list:
        b = max(int(budget), 1)
        cfg = MiningConfig(
            methods=self.methods, horizon=self.horizon, train_ratio=self.train_ratio,
            gp=GPConfig(population=max(20, b), generations=max(2, b // 20), seed=self.seed),
            rl=RLConfig(iterations=max(2, b // 10), batch_size=16, seed=self.seed),
            llm=LLMConfig(rounds=max(1, b // 50)),
        )
        before = set(self.library.records)
        _, self.last_report = mine_factors(panel, cfg, llm=self.llm, library=self.library, seeds=seeds,
                                           log=kwargs.get("log", lambda s: None))
        return [record_to_spec(r, self.horizon) for n, r in self.library.records.items() if n not in before]

    def evaluate(self, spec, panel: Panel, sample: str = "oos", **kwargs):
        expr = spec["expression"] if isinstance(spec, dict) else spec.expression
        horizon = (spec.get("horizon", 1) if isinstance(spec, dict) else spec.horizon) or 1
        f = Evaluator(panel)(parse(expr))
        fwd = forward_returns(panel, horizon)
        T = len(f)
        cut = int(T * kwargs.get("train_ratio", self.train_ratio))
        rows = slice(0, cut - horizon) if sample == "is" else slice(cut, T - horizon)
        if sample == "full":
            rows = slice(0, T - horizon)
        fr, yr = f.iloc[rows], fwd.iloc[rows]
        rep = evaluate_factor(fr, yr, horizon=horizon)
        decay = ic_decay(fr, panel["close"].iloc[rows], horizons=(1, 2, 3, 5, 10, 20), marginal=True).abs()
        return _mk(
            _QReport, spec=spec, ic_mean=rep.ic, ic_ir=float(rep.icir / np.sqrt(252 / horizon)), rank_ic_mean=rep.rank_ic,  # ic_ir 为单期 IR（未年化）
            turnover=rep.turnover, long_short_sharpe=rep.ls_sharpe, decay_half_life=_half_life(decay),
            coverage=rep.coverage, period_start=str(fr.index[0]), period_end=str(fr.index[-1]),
            sample="is" if sample == "is" else "oos",
            extra={"ic_ir_annualized": rep.icir, "rank_icir_annualized": rep.rank_icir, "ic_t": rep.ic_t, "ls_ann_return": rep.ls_ann_return,
                   "monotonicity": rep.monotonicity, "n_periods": float(rep.n_periods)},
        )


def make_alpha_model(**cfg) -> ContractAlphaModel:
    return ContractAlphaModel(**cfg)


def make_miner(**cfg) -> ContractFactorMiner:
    return ContractFactorMiner(**cfg)
