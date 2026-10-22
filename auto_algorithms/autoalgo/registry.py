"""注册表：按 kind 名实例化生成的算法，并向 quantstack 暴露 ``make_*`` 工厂。

kind 命名
  * ``autoalgo.<algo_id>``          generated/ 里的生成算法（使用搜索出的参数）
  * ``autoalgo.family.<family>``    模板族默认参数（无需先运行生成管线）
  * ``auto``                        该协议下第一个评估通过的生成算法，没有则取模板族默认
也接受不带前缀的 ``<algo_id>`` / ``<family>``。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from ._paths import GENERATED_DIR
from .codegen import load_index
from .families import FAMILIES, FamilySpec

_PROTOCOL_FAMILY_DEFAULT = {
    "AlphaModel": "alpha.regime_gated", "PortfolioConstructor": "portfolio.denoised_riskparity", "ExecutionStrategy": "execution.kalman_pid",
    "StrategySelector": "selector.bandit", "Router": "router.bayesian_venue",
}


def entries(out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    return load_index(out_dir or GENERATED_DIR)


def load_module(algo_id: str, out_dir: Optional[str] = None):
    out_dir = out_dir or GENERATED_DIR
    path = os.path.join(out_dir, algo_id, "algorithm.py")
    if not os.path.exists(path):
        raise KeyError(f"generated/ 中没有算法 {algo_id!r}（{path}）")
    name = f"autoalgo_generated.{algo_id}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _strip(kind: str) -> str:
    return kind[len("autoalgo."):] if kind.startswith("autoalgo.") else kind


def resolve(kind: str, protocol: Optional[str] = None, out_dir: Optional[str] = None):
    """返回 ("generated", algo_id) 或 ("family", FamilySpec)。"""
    k = _strip(kind)
    if k == "auto" or k == "":
        if protocol is None:
            raise ValueError("kind='auto' 需要指定协议")
        for e in entries(out_dir):
            if e["protocol"] == protocol and e.get("passed"):
                return "generated", e["algo_id"]
        for e in entries(out_dir):
            if e["protocol"] == protocol:
                return "generated", e["algo_id"]
        return "family", FAMILIES[_PROTOCOL_FAMILY_DEFAULT[protocol]]
    if k.startswith("family."):
        return "family", FAMILIES[k[len("family."):]]
    if k in FAMILIES:
        return "family", FAMILIES[k]
    ids = {e["algo_id"] for e in entries(out_dir)}
    if k in ids:
        return "generated", k
    for spec in FAMILIES.values():
        if spec.default_algo_id == k:
            return "family", spec
    raise KeyError(f"未知 kind {kind!r}；可选：{available_kinds(out_dir)}")


def available_kinds(out_dir: Optional[str] = None) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for e in entries(out_dir):
        out.setdefault(e["protocol"], []).append(f"autoalgo.{e['algo_id']}")
    for f in FAMILIES.values():
        out.setdefault(f.protocol, []).append(f"autoalgo.family.{f.family}")
    return out


def make(kind: str, protocol: Optional[str] = None, out_dir: Optional[str] = None, **kw):
    """通用实例化。exec_factory 族接受 base / factory_cfg；router 族接受 base / venues；其余参数作为覆盖。"""
    how, obj = resolve(kind, protocol, out_dir)
    if how == "generated":
        return load_module(obj, out_dir).make(**kw)
    spec: FamilySpec = obj
    ctx = {k: kw.pop(k) for k in ("base", "venues", "factory_cfg") if k in kw}
    return spec.build(kw, **ctx)


# ----------------------------------------------------------------------------- quantstack 工厂
def make_alpha_model(kind: str = "auto", out_dir: Optional[str] = None, **cfg):
    return make(kind, "AlphaModel", out_dir, **cfg)


def make_portfolio(kind: str = "auto", out_dir: Optional[str] = None, **cfg):
    return make(kind, "PortfolioConstructor", out_dir, **cfg)


def make_execution_factory(kind: str = "auto", base=None, out_dir: Optional[str] = None, factory_cfg: Optional[dict] = None, **cfg):
    return make(kind, "ExecutionStrategy", out_dir, base=base, factory_cfg=factory_cfg or {}, **cfg)


def make_router(kind: str = "auto", venues: Sequence[str] = (), base=None, out_dir: Optional[str] = None, **cfg):
    return make(kind, "Router", out_dir, base=base, venues=list(venues), **cfg)


def make_strategy_selector(kind: str = "auto", out_dir: Optional[str] = None, **cfg):
    return make(kind, "StrategySelector", out_dir, **cfg)


def registry_entries(out_dir: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{协议名: {kind: {"algo_id"|"family", "passed", "name_zh", "path"}}}``，供 quantstack.plugin_registry / agent 展示。"""
    reg: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in entries(out_dir):
        reg.setdefault(e["protocol"], {})[f"autoalgo.{e['algo_id']}"] = {"algo_id": e["algo_id"], "name_zh": e.get("name_zh", ""), "passed": e.get("passed", False), "path": e.get("path", "")}
    for f in FAMILIES.values():
        reg.setdefault(f.protocol, {})[f"autoalgo.family.{f.family}"] = {"family": f.family, "name_zh": f.name_zh, "passed": None, "path": f.import_path}
    return reg
