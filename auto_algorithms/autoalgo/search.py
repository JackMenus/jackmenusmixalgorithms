"""参数搜索：在模板族的搜索空间内提议候选参数。

三种提议器（都实现 ``propose(spec, n, history) -> List[params]``）：
  * ``RandomProposer``        均匀 / 对数均匀随机采样（可复现）；
  * ``EvolutionaryProposer``  从历史中按目标选最优的若干个做变异（每次改 1–2 个维度），不足时随机补齐；
  * ``autoalgo.llm.ClaudeProposer`` / ``MockProposer`` 在 llm 模块中，接口相同。

``history`` 是 ``[(params, score)]``（score 已按"越大越好"方向调整）。
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .families import FamilySpec

History = List[Tuple[Dict[str, Any], float]]


def sample_dim(dim: Tuple, rng: np.random.Generator):
    kind = dim[0]
    if kind == "int":
        return int(rng.integers(int(dim[1]), int(dim[2]) + 1))
    if kind == "float":
        return float(rng.uniform(float(dim[1]), float(dim[2])))
    if kind == "logfloat":
        return float(math.exp(rng.uniform(math.log(float(dim[1])), math.log(float(dim[2])))))
    if kind == "choice":
        opts = list(dim[1])
        return copy.deepcopy(opts[int(rng.integers(len(opts)))])
    raise ValueError(f"未知搜索维度 {dim!r}")


def clip_to_space(params: Dict[str, Any], spec: FamilySpec) -> Dict[str, Any]:
    """把外部（如 LLM）提议的参数裁剪到搜索空间内；未知键保留原值，非法 choice 回退默认。"""
    out = dict(spec.default_params)
    for k, v in params.items():
        dim = spec.search_space.get(k)
        if dim is None:
            out[k] = v
            continue
        kind = dim[0]
        try:
            if kind == "int":
                out[k] = int(np.clip(int(round(float(v))), int(dim[1]), int(dim[2])))
            elif kind in ("float", "logfloat"):
                out[k] = float(np.clip(float(v), float(dim[1]), float(dim[2])))
            elif kind == "choice":
                opts = list(dim[1])
                out[k] = v if v in opts else _nearest_choice(v, opts, spec.default_params.get(k))
        except (TypeError, ValueError):
            out[k] = spec.default_params.get(k, v)
    return out


def _nearest_choice(v, opts, default):
    try:
        fv = float(v)
        nums = [o for o in opts if isinstance(o, (int, float)) and not isinstance(o, bool)]
        if nums:
            return min(nums, key=lambda o: abs(float(o) - fv))
    except (TypeError, ValueError):
        pass
    return default if default in opts else opts[0]


def params_key(params: Dict[str, Any]) -> str:
    import json

    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


class RandomProposer:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(self, spec: FamilySpec, n: int, history: Optional[History] = None) -> List[Dict[str, Any]]:
        seen = {params_key(p) for p, _ in (history or [])}
        out: List[Dict[str, Any]] = []
        tries = 0
        while len(out) < n and tries < 50 * max(n, 1):
            tries += 1
            cand = {**spec.default_params, **{k: sample_dim(d, self.rng) for k, d in spec.search_space.items()}}
            key = params_key(cand)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        return out


class EvolutionaryProposer:
    """(1+λ) 风格：取历史前 ``elite`` 个做变异；每个子代改 1–2 个维度。"""

    name = "evolution"

    def __init__(self, seed: int = 0, elite: int = 2, n_mut: int = 2):
        self.rng = np.random.default_rng(seed)
        self.elite, self.n_mut = int(elite), int(n_mut)
        self._random = RandomProposer(seed + 1)

    def mutate(self, spec: FamilySpec, params: Dict[str, Any]) -> Dict[str, Any]:
        keys = list(spec.search_space)
        if not keys:
            return dict(params)
        child = copy.deepcopy(params)
        for k in self.rng.choice(keys, size=min(len(keys), int(self.rng.integers(1, self.n_mut + 1))), replace=False):
            dim = spec.search_space[k]
            if dim[0] in ("float", "logfloat") and k in child and isinstance(child[k], (int, float)):
                lo, hi = float(dim[1]), float(dim[2])
                if dim[0] == "logfloat":
                    v = math.exp(math.log(float(child[k])) + self.rng.normal(0, 0.3 * (math.log(hi) - math.log(lo))))
                else:
                    v = float(child[k]) + self.rng.normal(0, 0.2 * (hi - lo))
                child[k] = float(np.clip(v, lo, hi))
            else:
                child[k] = sample_dim(dim, self.rng)
        return child

    def propose(self, spec: FamilySpec, n: int, history: Optional[History] = None) -> List[Dict[str, Any]]:
        history = [(p, s) for p, s in (history or []) if s is not None and np.isfinite(s)]
        if not history:
            return self._random.propose(spec, n, history)
        ranked = sorted(history, key=lambda t: t[1], reverse=True)[: self.elite]
        seen = {params_key(p) for p, _ in history}
        out: List[Dict[str, Any]] = []
        tries = 0
        while len(out) < n and tries < 50 * max(n, 1):
            tries += 1
            parent = ranked[tries % len(ranked)][0]
            child = self.mutate(spec, parent)
            key = params_key(child)
            if key in seen:
                continue
            seen.add(key)
            out.append(child)
        if len(out) < n:
            out += self._random.propose(spec, n - len(out), history + [(p, 0.0) for p in out])
        return out


PROPOSERS = {"random": RandomProposer, "evolution": EvolutionaryProposer}


def make_proposer(name: str = "random", seed: int = 0, **kw):
    if name in PROPOSERS:
        return PROPOSERS[name](seed=seed, **kw)
    if name in ("claude", "llm", "mock"):
        from .llm import ClaudeProposer, MockProposer

        return MockProposer(seed=seed, **kw) if name == "mock" else ClaudeProposer(seed=seed, **kw)
    raise KeyError(f"未知提议器 {name!r}，可选：{sorted(PROPOSERS) + ['claude', 'mock']}")
