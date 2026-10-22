"""生成管线：对每个模板族 提议参数 → 评估 → 选优 → 写出代码 + 中文可解释性文档 → 更新 generated/ 索引。

    from autoalgo.pipeline import GenerationConfig, generate
    results = generate(panel, GenerationConfig(budget="quick", n_candidates=3, proposer="random", seed=0))

每个结果的 ``paths`` 指向 generated/<algo_id>/ 下的四个文件。全过程由 ``seed`` 决定（LLM 提议器除外，其输出会被完整记录在血缘中）。
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ._paths import GENERATED_DIR
from .blueprint import GENERATOR_VERSION, AlgorithmBlueprint, EvaluationReport
from .codegen import emit
from .evaluate import evaluate, signed_score
from .families import FAMILIES, FamilySpec, families_for
from .search import make_proposer, params_key


@dataclass
class GenerationConfig:
    families: Optional[List[str]] = None          # 模板族名列表，默认全部
    protocols: Optional[List[str]] = None         # 或按协议筛选
    n_candidates: int = 3                         # 默认参数之外的候选数
    budget: str = "quick"                         # 评估预算 quick / standard / full
    seed: int = 0
    proposer: str = "random"                      # random / evolution / claude / mock
    model: str = "claude-fable-5-1"
    effort: str = "max"
    out_dir: Optional[str] = None                 # 默认 auto_algorithms/generated
    train_ratio: float = 0.7
    algo_ids: Dict[str, str] = field(default_factory=dict)   # family -> algo_id（默认用模板族的 default_algo_id）
    emit_failed: bool = True                      # 未通过评估的最优候选也写出（md 中标明）
    rounds: int = 2                               # 提议轮数（evolution / claude 按轮迭代）

    def resolved_families(self) -> List[FamilySpec]:
        if self.families:
            return [FAMILIES[f] for f in self.families]
        return families_for(self.protocols)


@dataclass
class GenerationResult:
    blueprint: AlgorithmBlueprint
    report: EvaluationReport
    paths: Dict[str, str]
    candidates: List[Dict[str, Any]]
    wall_s: float

    def summary(self) -> str:
        r = self.report
        base = r.baseline_primary()
        return (f"[{r.protocol}] {self.blueprint.algo_id}: {r.objective}={r.primary():.4g}" + (f" (基线 {base:.4g})" if base is not None else "") +
                f" {'通过' if r.passed else '未通过'} · {len(self.candidates)} 个候选 · {self.wall_s:.0f}s")


def default_panel(seed: int = 0, n_days: int = 400, n_assets: int = 60):
    from ._paths import require

    require("quantstack")
    from quantstack.sim.market import make_synthetic_panel

    return make_synthetic_panel(n_days, n_assets, seed=seed)


def _blueprint(spec: FamilySpec, params: Dict[str, Any], algo_id: str, lineage: Dict[str, Any]) -> AlgorithmBlueprint:
    return AlgorithmBlueprint(
        algo_id=algo_id, name_zh=spec.name_zh, protocol=spec.protocol, family=spec.family, params=dict(params), components=list(spec.components),
        hypothesis=spec.hypothesis, mechanism=list(spec.mechanism), assumptions=list(spec.assumptions), risks=list(spec.risks), param_doc=dict(spec.param_doc),
        search_space={k: list(v) for k, v in spec.search_space.items()}, lineage=lineage, registry_kind=f"autoalgo.{algo_id}",
    )


def generate_family(spec: FamilySpec, panel, config: GenerationConfig, proposer=None, log: Callable[[str], None] = print) -> GenerationResult:
    t0 = time.time()
    proposer = proposer or make_proposer(config.proposer, seed=config.seed, **({"model": config.model, "effort": config.effort} if config.proposer == "claude" else {}))
    algo_id = config.algo_ids.get(spec.family, spec.default_algo_id)
    history: List = []
    candidates: List[Dict[str, Any]] = []
    reports: Dict[str, EvaluationReport] = {}

    def _eval(params: Dict[str, Any], source: str):
        key = params_key(params)
        if key in reports:
            return
        log(f"  · 评估候选 #{len(candidates) + 1}（{source}）{json.dumps(params, ensure_ascii=False, default=str)[:160]}")
        rep = evaluate(spec, params, panel, budget=config.budget, seed=config.seed, train_ratio=config.train_ratio)
        score = signed_score(rep)
        history.append((params, score))
        reports[key] = rep
        candidates.append({"params": params, "source": source, "objective": rep.objective, "value": rep.primary(), "score": score, "passed": rep.passed,
                           "rationale": getattr(proposer, "rationales", {}).get(key, ""), "wall_s": round(rep.wall_s, 1)})
        log(f"    → {rep.objective}={rep.primary():.4g}  {'通过' if rep.passed else '未通过'}  ({rep.wall_s:.0f}s)")

    log(f"[{spec.protocol}] 模板族 {spec.family}（{spec.name_zh}）：默认参数 + {config.n_candidates} 个候选，预算 {config.budget}")
    _eval(dict(spec.default_params), "default")
    remaining = int(config.n_candidates)
    rounds = max(1, int(config.rounds)) if getattr(proposer, "name", "") in ("evolution", "claude", "mock") else 1
    per_round = int(math.ceil(remaining / rounds)) if remaining > 0 else 0
    for _ in range(rounds):
        if remaining <= 0:
            break
        k = min(per_round, remaining)
        for params in proposer.propose(spec, k, history):
            _eval(params, getattr(proposer, "effective_name", getattr(proposer, "name", "proposer")))
        remaining -= k
    best_params, best_score = max(history, key=lambda t: t[1])
    best_rep = reports[params_key(best_params)]
    lineage = {
        "generator": "autoalgo.pipeline.generate", "generator_version": GENERATOR_VERSION, "proposer": getattr(proposer, "effective_name", getattr(proposer, "name", "")),
        "model": config.model if config.proposer == "claude" else "", "effort": config.effort if config.proposer == "claude" else "", "llm_error": getattr(proposer, "last_error", ""),
        "seed": config.seed, "budget": config.budget, "train_ratio": config.train_ratio, "n_candidates": len(candidates),
        "panel": {"n_days": int(len(panel["close"])), "n_assets": int(panel["close"].shape[1]), "start": str(panel["close"].index[0].date()), "end": str(panel["close"].index[-1].date())},
        "candidates": candidates, "selected_score": best_score,
    }
    bp = _blueprint(spec, best_params, algo_id, lineage)
    best_rep.algo_id = algo_id
    if not best_rep.passed and not config.emit_failed:
        log(f"  最优候选未通过评估且 emit_failed=False，跳过写出 {algo_id}")
        return GenerationResult(bp, best_rep, {}, candidates, time.time() - t0)
    paths = emit(bp, spec, best_rep, out_dir=config.out_dir or GENERATED_DIR)
    res = GenerationResult(bp, best_rep, paths, candidates, time.time() - t0)
    log("  " + res.summary())
    return res


def generate(panel=None, config: Optional[GenerationConfig] = None, log: Callable[[str], None] = print) -> List[GenerationResult]:
    config = config or GenerationConfig()
    panel = panel if panel is not None else default_panel(config.seed)
    out_dir = config.out_dir or GENERATED_DIR
    os.makedirs(out_dir, exist_ok=True)
    proposer = make_proposer(config.proposer, seed=config.seed, **({"model": config.model, "effort": config.effort} if config.proposer == "claude" else {}))
    results: List[GenerationResult] = []
    t0 = time.time()
    for spec in config.resolved_families():
        results.append(generate_family(spec, panel, config, proposer=proposer, log=log))
    runlog = {"config": asdict(config), "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "wall_s": round(time.time() - t0, 1),
              "results": [{"algo_id": r.blueprint.algo_id, "family": r.blueprint.family, "protocol": r.blueprint.protocol, "objective": r.report.objective, "value": r.report.primary(),
                           "baseline": r.report.baseline_primary(), "passed": r.report.passed, "n_candidates": len(r.candidates), "wall_s": round(r.wall_s, 1)} for r in results]}
    with open(os.path.join(out_dir, "last_run.json"), "w", encoding="utf-8") as f:
        json.dump(runlog, f, ensure_ascii=False, indent=2, default=str)
    log(f"完成：{len(results)} 个算法，总耗时 {time.time() - t0:.0f}s，产物目录 {out_dir}")
    return results
