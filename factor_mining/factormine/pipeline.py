"""一站式挖掘流水线：GP / RL / LLM 共享同一个因子池，最后做样本外检验并写入因子库。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import pandas as pd

from .data import Panel, forward_returns
from .evaluate import evaluate_factor
from .expression import Evaluator
from .gp import GPConfig, GPMiner
from .library import FactorLibrary, FactorPool
from .llm import LLMAlphaAgent, LLMClient, LLMConfig
from .rl import RLConfig, RLMiner


@dataclass
class MiningConfig:
    methods: Sequence[str] = ("gp", "rl", "llm")
    horizon: int = 1
    train_ratio: float = 0.7
    pool_capacity: int = 20
    max_corr: float = 0.7
    min_ic: float = 0.015
    min_valid_ic: float = 0.005  # 样本外 |RankIC| 低于此值不入库
    gp: GPConfig = field(default_factory=GPConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def mine_factors(
    panel: Panel,
    config: MiningConfig = MiningConfig(),
    llm: Optional[LLMClient] = None,
    library: Optional[FactorLibrary] = None,
    seeds: Sequence[str] = (),
    log: Callable[[str], None] = print,
) -> tuple[FactorLibrary, pd.DataFrame]:
    """返回 (因子库, 样本内外指标表)。

    因子在完整面板上计算（滚动窗口有足够预热），但适应度只用前 train_ratio 段，
    样本外检验只用剩余段，避免信息泄漏。
    """
    T = len(panel["close"])
    cut = int(T * config.train_ratio)
    train_rows, valid_rows = slice(0, cut - config.horizon), slice(cut, T - config.horizon)
    ev = Evaluator(panel)
    fwd = forward_returns(panel, config.horizon)
    pool = FactorPool(ev, fwd, train_rows, config.pool_capacity, config.max_corr, config.min_ic)
    for s in seeds:
        from .expression import parse
        pool.try_add(parse(s), source="seed")

    for m in config.methods:
        if m == "gp":
            GPMiner(pool, config.gp, log=log).run(seeds)
        elif m == "rl":
            RLMiner(pool, config=config.rl, log=log).run()
        elif m == "llm":
            if llm is None:
                log("[LLM] 未提供 LLM 客户端，跳过")
                continue
            LLMAlphaAgent(pool, llm, config.llm, log=log).run()
        else:
            raise ValueError(f"未知方法 {m}")

    library = library if library is not None else FactorLibrary()
    rows = []
    recs = pool.to_records(prefix="alpha")
    for rec, v in zip(recs, pool.values):
        tr = evaluate_factor(v.iloc[train_rows], fwd.iloc[train_rows], horizon=config.horizon)
        va = evaluate_factor(v.iloc[valid_rows], fwd.iloc[valid_rows], horizon=config.horizon)
        rec.metrics_train, rec.metrics_valid = tr.to_dict(), va.to_dict()
        # 样本外方向必须与样本内一致且有一定强度
        passed = abs(va.rank_ic) >= config.min_valid_ic and (va.rank_ic * tr.rank_ic > 0)
        rows.append({"expression": rec.expression, "source": rec.source,
                     "train_rank_ic": tr.rank_ic, "valid_rank_ic": va.rank_ic,
                     "valid_icir": va.rank_icir, "turnover": va.turnover, "passed": passed})
        if passed:
            library.extend([rec])
    report = pd.DataFrame(rows).sort_values("valid_rank_ic", key=abs, ascending=False) if rows else pd.DataFrame()
    log(f"[完成] 池内 {len(pool)} 个因子，样本外通过 {int(report['passed'].sum()) if len(report) else 0} 个，因子库共 {len(library)} 个")
    return library, report
