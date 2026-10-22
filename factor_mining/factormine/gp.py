"""遗传规划（Genetic Programming）公式化因子挖掘。

思路综合自：
  * gplearn / 经典 GP 符号回归；
  * AutoAlpha（Zhang et al., 2020）：分层进化、用与已有因子的相关性做多样性约束、热启动；
  * AlphaEvolve（Cui et al., SIGMOD 2021）：在公式中引入参数与跨资产算子。

适应度 = |训练段 RankIC| − 复杂度惩罚 − 与因子池的最大相关 × 惩罚系数。
每代把优秀个体投递进 ``FactorPool``，池内因子之间相关性受控，从而产出“协同”的因子集合。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .expression import Evaluator, Expr, parse
from .library import FactorPool
from .operators import DEFAULT_WINDOWS, OPERATORS

_CONST_CHOICES = {
    "signed_power": (0.5, 2.0, 3.0),
    "add_c": (-1.0, 1.0),
    "mul_c": (-1.0, 0.5, 2.0),
    "cs_winsorize": (3.0,),
}


@dataclass
class GPConfig:
    population: int = 200
    generations: int = 10
    max_depth: int = 4
    tournament: int = 5
    p_crossover: float = 0.6
    p_mutation: float = 0.3
    elitism: int = 10
    parsimony: float = 0.0005  # 每个节点的惩罚
    corr_penalty: float = 0.05
    fields: Sequence[str] = ("open", "high", "low", "close", "volume", "vwap", "returns")
    windows: Sequence[int] = DEFAULT_WINDOWS
    operators: Optional[Sequence[str]] = None  # None = 全部
    submit_top: int = 10  # 每代尝试入池的个体数
    seed: int = 0


class GPMiner:
    def __init__(self, pool: FactorPool, config: GPConfig = GPConfig(), log: Callable[[str], None] = print):
        self.pool = pool
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.ops = [OPERATORS[o] for o in (config.operators or OPERATORS)]
        self.log = log
        self._fit_cache: dict = {}
        self.history: List[dict] = []

    # ------------------------------------------------------------ 随机生成
    def _arg(self, ch: str, op: str, depth: int) -> Expr:
        if ch == "w":
            return Expr.const(self.rng.choice(self.cfg.windows))
        if ch == "c":
            return Expr.const(self.rng.choice(_CONST_CHOICES.get(op, (1.0,))))
        return self.random_tree(depth - 1)

    def random_tree(self, depth: Optional[int] = None, full: bool = False) -> Expr:
        depth = self.cfg.max_depth if depth is None else depth
        if depth <= 1 or (not full and self.rng.random() < 0.3):
            return Expr.feat(self.rng.choice(self.cfg.fields))
        spec = self.rng.choice(self.ops)
        return Expr.call(spec.name, *[self._arg(ch, spec.name, depth) for ch in spec.arity])

    def _factor_nodes(self, e: Expr):
        """可被替换的“因子值”节点（排除窗口/常数参数位）。"""
        out = []
        for node, parent, i in e.nodes():
            if parent is None or OPERATORS[parent.op].arity[i] == "x":
                out.append((node, parent, i))
        return out

    # ------------------------------------------------------------ 遗传算子
    def crossover(self, a: Expr, b: Expr) -> Expr:
        child = a.copy()
        targets = self._factor_nodes(child)
        donor, _, _ = self.rng.choice(self._factor_nodes(b))
        node, parent, i = self.rng.choice(targets)
        if parent is None:
            child = donor.copy()
        else:
            parent.children[i] = donor.copy()
        return child if child.depth() <= self.cfg.max_depth + 1 else a.copy()

    def mutate(self, a: Expr) -> Expr:
        child = a.copy()
        kind = self.rng.random()
        nodes = child.nodes()
        if kind < 0.4:  # 子树替换
            node, parent, i = self.rng.choice(self._factor_nodes(child))
            new = self.random_tree(max(2, self.cfg.max_depth - node.depth() + 1))
            if parent is None:
                return new
            parent.children[i] = new
        elif kind < 0.7:  # 窗口/常数扰动
            params = [(n, p, i) for n, p, i in nodes if p is not None and OPERATORS[p.op].arity[i] in "wc"]
            if not params:
                return self.mutate(a) if self.rng.random() < 0.5 else child
            n, p, i = self.rng.choice(params)
            ch = OPERATORS[p.op].arity[i]
            n.value = self.rng.choice(self.cfg.windows) if ch == "w" else self.rng.choice(_CONST_CHOICES.get(p.op, (1.0,)))
        elif kind < 0.9:  # 同签名算子替换
            inner = [n for n, _, _ in nodes if not n.is_leaf]
            if inner:
                n = self.rng.choice(inner)
                same = [s.name for s in self.ops if s.arity == OPERATORS[n.op].arity]
                n.op = self.rng.choice(same)
        else:  # 外包一层横截面/时序算子
            spec = self.rng.choice([s for s in self.ops if s.arity in ("x", "xw")])
            args = [child] + ([Expr.const(self.rng.choice(self.cfg.windows))] if spec.arity == "xw" else [])
            child = Expr.call(spec.name, *args)
        return child if child.depth() <= self.cfg.max_depth + 1 else a.copy()

    # ------------------------------------------------------------ 适应度
    def fitness(self, e: Expr) -> float:
        key = str(e)
        if key in self._fit_cache:
            return self._fit_cache[key]
        ic, v = self.pool.score(e)
        if v is None:
            f = -1.0
        else:
            corr = self.pool.max_corr_with_pool(v) if len(self.pool) else 0.0
            f = abs(ic) - self.cfg.parsimony * e.size() - self.cfg.corr_penalty * corr
        f = f if np.isfinite(f) else -1.0
        self._fit_cache[key] = f
        return f

    def _select(self, pop: List[Expr], fits: List[float]) -> Expr:
        idx = self.rng.sample(range(len(pop)), min(self.cfg.tournament, len(pop)))
        return pop[max(idx, key=lambda i: fits[i])]

    # ------------------------------------------------------------ 主循环
    def run(self, seeds: Sequence[str | Expr] = ()) -> FactorPool:
        """seeds：热启动表达式（例如经典 Alpha101 或 LLM 给出的候选）。"""
        pop = [parse(s) if isinstance(s, str) else s for s in seeds]
        while len(pop) < self.cfg.population:
            pop.append(self.random_tree(full=self.rng.random() < 0.5))
        for g in range(self.cfg.generations):
            fits = [self.fitness(e) for e in pop]
            order = np.argsort(fits)[::-1]
            added = 0
            for k in order[: self.cfg.submit_top]:
                added += self.pool.try_add(pop[k], source="gp")
            # 池子变化后相关惩罚失效，清缓存
            if added:
                self._fit_cache.clear()
            best = pop[order[0]]
            self.history.append({"gen": g, "best_fit": fits[order[0]], "best": str(best), "pool": len(self.pool)})
            self.log(f"[GP] 第 {g} 代  最优适应度={fits[order[0]]:.4f}  入池+{added}  池大小={len(self.pool)}  {best}")
            nxt = [pop[k].copy() for k in order[: self.cfg.elitism]]
            seen = {str(e) for e in nxt}
            while len(nxt) < self.cfg.population:
                r = self.rng.random()
                if r < self.cfg.p_crossover:
                    c = self.crossover(self._select(pop, fits), self._select(pop, fits))
                elif r < self.cfg.p_crossover + self.cfg.p_mutation:
                    c = self.mutate(self._select(pop, fits))
                else:
                    c = self.random_tree()
                if str(c) not in seen:
                    seen.add(str(c))
                    nxt.append(c)
            pop = nxt
        return self.pool
