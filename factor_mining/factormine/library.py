"""因子池（挖掘过程中的协同因子集合）与因子库（持久化存储）。

* ``FactorPool``  —— 挖掘器共享的在线因子池：去相关准入、容量淘汰、合成 IC 计算。
  遗传规划、强化学习、LLM 智能体都往同一个池子里投递候选，便于混合使用。
* ``FactorLibrary`` —— 落盘的因子库（JSON），记录表达式、来源算法、样本内外指标，
  是对接挂单/路由模块与全栈 agent 的“因子仓库”。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .combine import combine, fit_weights
from .evaluate import factor_correlation, ic_series
from .expression import Evaluator, Expr, parse


@dataclass
class FactorRecord:
    name: str
    expression: str
    source: str  # gp / rl / llm / neural / manual
    metrics_train: Dict[str, float] = field(default_factory=dict)
    metrics_valid: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""  # LLM 生成时的经济学假设
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


class FactorPool:
    """容量受限、相关性受控的因子池（AlphaGen 中的 alpha pool 思想）。

    准入规则：与池内任一因子的横截面秩相关绝对值 ≤ max_corr；
    池满时，新因子得分须高于最弱者才替换。
    """

    def __init__(
        self,
        evaluator: Evaluator,
        fwd: pd.DataFrame,
        train_rows: slice,
        capacity: int = 20,
        max_corr: float = 0.7,
        min_ic: float = 0.01,
    ):
        self.ev = evaluator
        self.fwd = fwd
        self.rows = train_rows
        self.capacity = capacity
        self.max_corr = max_corr
        self.min_ic = min_ic
        self.exprs: List[Expr] = []
        self.values: List[pd.DataFrame] = []
        self.ics: List[float] = []
        self.sources: List[str] = []

    def __len__(self) -> int:
        return len(self.exprs)

    def _train(self, f: pd.DataFrame) -> pd.DataFrame:
        return f.iloc[self.rows]

    def score(self, e: Expr) -> tuple[float, Optional[pd.DataFrame]]:
        try:
            v = self.ev(e)
        except Exception:
            return 0.0, None
        if self._train(v).notna().to_numpy().mean() < 0.5:
            return 0.0, None
        ics = ic_series(self._train(v), self._train(self.fwd))
        if ics.notna().mean() < 0.5:  # 常数/退化因子：多数期横截面无差异
            return 0.0, None
        ic = ics.mean()
        return (float(ic) if np.isfinite(ic) else 0.0), v

    def max_corr_with_pool(self, v: pd.DataFrame) -> float:
        if not self.values:
            return 0.0
        tv = self._train(v)
        cs = [abs(factor_correlation(tv, self._train(p))) for p in self.values]
        cs = [c for c in cs if np.isfinite(c)]
        return max(cs) if cs else 0.0

    def try_add(self, e: Expr, source: str = "manual") -> bool:
        if any(str(e) == str(x) for x in self.exprs):
            return False
        ic, v = self.score(e)
        if v is None or abs(ic) < self.min_ic:
            return False
        if self.max_corr_with_pool(v) > self.max_corr:
            return False
        if len(self.exprs) >= self.capacity:
            worst = int(np.argmin(np.abs(self.ics)))
            if abs(ic) <= abs(self.ics[worst]):
                return False
            for lst in (self.exprs, self.values, self.ics, self.sources):
                lst.pop(worst)
        self.exprs.append(e)
        self.values.append(v)
        self.ics.append(ic)
        self.sources.append(source)
        return True

    def as_factors(self) -> Dict[str, pd.DataFrame]:
        return {str(e): v for e, v in zip(self.exprs, self.values)}

    def combined_ic(self, extra: Optional[pd.DataFrame] = None, method: str = "ridge") -> float:
        """池（可附加一个候选）线性合成后的训练段 RankIC——RL 奖励即基于它的增量。"""
        facs = {f"f{i}": self._train(v) for i, v in enumerate(self.values)}
        if extra is not None:
            facs["cand"] = self._train(extra)
        if not facs:
            return 0.0
        fwd = self._train(self.fwd)
        w = fit_weights(facs, fwd, method=method)
        ic = ic_series(combine(facs, w), fwd).mean()
        return float(ic) if np.isfinite(ic) else 0.0

    def to_records(self, prefix: str = "alpha") -> List[FactorRecord]:
        return [
            FactorRecord(name=f"{prefix}_{i:03d}", expression=str(e), source=s, metrics_train={"rank_ic": ic})
            for i, (e, ic, s) in enumerate(zip(self.exprs, self.ics, self.sources))
        ]


class FactorLibrary:
    """JSON 持久化的因子库。

    用法::

        lib = FactorLibrary("alphas.json")
        lib.add(record); lib.save()
        values = lib.compute(panel)          # {name: DataFrame}
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self.records: Dict[str, FactorRecord] = {}
        if self.path and self.path.exists():
            self.load()

    def add(self, rec: FactorRecord, overwrite: bool = False) -> None:
        parse(rec.expression)  # 入库前校验
        if rec.name in self.records and not overwrite:
            raise KeyError(f"因子 {rec.name} 已存在")
        self.records[rec.name] = rec

    def extend(self, recs: List[FactorRecord]) -> None:
        existing = {r.expression for r in self.records.values()}
        for r in recs:
            if r.expression in existing:
                continue
            name = r.name
            i = 1
            while name in self.records:
                name = f"{r.name}_{i}"
                i += 1
            r.name = name
            self.add(r)
            existing.add(r.expression)

    def __len__(self) -> int:
        return len(self.records)

    def save(self, path: Optional[str | Path] = None) -> Path:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("未指定保存路径")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([asdict(r) for r in self.records.values()], ensure_ascii=False, indent=2))
        return p

    def load(self) -> None:
        for d in json.loads(self.path.read_text()):
            self.records[d["name"]] = FactorRecord(**d)

    def compute(self, panel, names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        ev = Evaluator(panel)
        names = names or list(self.records)
        return {n: ev(parse(self.records[n].expression)) for n in names}

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.records.values():
            d = {"name": r.name, "expression": r.expression, "source": r.source}
            d.update({f"train_{k}": v for k, v in r.metrics_train.items()})
            d.update({f"valid_{k}": v for k, v in r.metrics_valid.items()})
            rows.append(d)
        return pd.DataFrame(rows)
