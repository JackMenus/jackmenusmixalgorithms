"""代码生成：把一份蓝图 + 评估报告落盘为 ``generated/<algo_id>/`` 目录，并维护 ``generated/index.json``。

目录内容
  algorithm.py     可直接 import 的模块：``make(**overrides)`` 返回算法实例；``BLUEPRINT`` / ``PARAMS`` / ``PROTOCOL`` / ``REGISTRY_KIND`` 元数据
  <algo_id>.md     中文可解释性文档（explain.render_markdown）
  blueprint.json   蓝图
  evaluation.json  评估报告
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from ._paths import GENERATED_DIR
from .blueprint import AlgorithmBlueprint, EvaluationReport
from .explain import render_index_markdown, render_markdown
from .families import FamilySpec

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{2,60}$")


def validate_algo_id(algo_id: str) -> str:
    if not _ID_RE.match(algo_id):
        raise ValueError(f"algo_id 须为小写字母/数字/下划线，且以字母开头：{algo_id!r}")
    return algo_id


def _make_body(spec: FamilySpec) -> str:
    mod, name = spec.import_path.split(":")
    if spec.build_kind == "exec_factory":
        return (f"    from {mod} import {name}\n\n"
                f"    params = {{**PARAMS, **overrides}}\n"
                f"    return {name}(base=base, params=params, **(factory_cfg or {{}}))\n")
    if spec.build_kind == "router":
        return (f"    from {mod} import {name}\n\n"
                f"    if base is None:\n"
                f"        from quantstack.core.registry import make_router as _mk\n\n"
                f"        base = _mk('auto', list(venues or ()))\n"
                f"    return {name}(base, list(venues or getattr(base, 'venues', [])), **{{**PARAMS, **overrides}})\n")
    return (f"    from {mod} import {name}\n\n"
            f"    return {name}(**{{**PARAMS, **overrides}})\n")


def _signature(spec: FamilySpec) -> str:
    if spec.build_kind == "exec_factory":
        return "def make(base=None, factory_cfg=None, **overrides):"
    if spec.build_kind == "router":
        return "def make(base=None, venues=None, **overrides):"
    return "def make(**overrides):"


def render_algorithm_py(bp: AlgorithmBlueprint, spec: FamilySpec, rep: Optional[EvaluationReport]) -> str:
    summary = ""
    if rep is not None:
        prim = rep.primary()
        base = rep.baseline_primary()
        summary = f"评估：{rep.objective} = {prim:.4g}" + (f"（基线 {base:.4g}）" if base is not None else "") + f"，{'通过' if rep.passed else '未通过'}。"
    doc = (f'"""{bp.name_zh}（autoalgo 自主生成，algo_id = {bp.algo_id}）\n\n'
           f"协议：{bp.protocol}；模板族：{bp.family}；注册名：{bp.registry_kind}\n"
           f"{bp.hypothesis}\n\n{summary}\n\n"
           f"说明文档：{bp.algo_id}.md（同目录）。本文件由 autoalgo.codegen 生成，参数来自生成管线的搜索结果；\n"
           f"手工修改请改 PARAMS 或在 make(**overrides) 传入覆盖值。\n"
           f'"""\n')
    lines = [doc, "from __future__ import annotations", "", "import json", "import os", "", f"ALGO_ID = {bp.algo_id!r}", f"PROTOCOL = {bp.protocol!r}",
             f"FAMILY = {bp.family!r}", f"REGISTRY_KIND = {bp.registry_kind!r}", f"NAME_ZH = {bp.name_zh!r}", f"GENERATOR_VERSION = {bp.generator_version!r}", "",
             "PARAMS = " + json.dumps(bp.params, ensure_ascii=False, indent=4, default=str).replace("true", "True").replace("false", "False").replace("null", "None"), "",
             "BLUEPRINT = " + json.dumps(bp.to_dict(), ensure_ascii=False, indent=4, default=str).replace("true", "True").replace("false", "False").replace("null", "None"), "",
             "_HERE = os.path.dirname(os.path.abspath(__file__))", "", "",
             "def _ensure_siblings():", "    \"\"\"确保 autoalgo 及其兄弟库可导入（独立脚本直接 import 本文件时也能工作）。\"\"\"",
             "    import sys", "", "    lib_root = os.path.dirname(os.path.dirname(_HERE))", "    if lib_root not in sys.path:", "        sys.path.insert(0, lib_root)",
             "    from autoalgo._paths import ensure_siblings", "", "    ensure_siblings()", "", "",
             _signature(spec), f'    """实例化 {bp.name_zh}；overrides 覆盖 PARAMS。"""', "    _ensure_siblings()", _make_body(spec), "",
             "def evaluation():", "    \"\"\"读取同目录的 evaluation.json（若存在）。\"\"\"", "    p = os.path.join(_HERE, 'evaluation.json')",
             "    if not os.path.exists(p):", "        return None", "    with open(p, encoding='utf-8') as f:", "        return json.load(f)", ""]
    return "\n".join(lines)


def usage_code(bp: AlgorithmBlueprint, spec: FamilySpec) -> str:
    kind = bp.registry_kind
    if spec.protocol == "AlphaModel":
        return (f"from autoalgo.registry import make_alpha_model\nmodel = make_alpha_model({kind!r})          # 或 quantstack: make_alpha_model(kind={kind!r})\n"
                f"model.fit(panel)\nbatch = model.predict(panel, as_of=panel['close'].index[-1])\n\n# 全栈流水线\nfrom quantstack.core.registry import build_sim_pipeline\n"
                f"pipe = build_sim_pipeline(panel, kinds={{'alpha': {kind!r}}})")
    if spec.protocol == "PortfolioConstructor":
        return (f"from autoalgo.registry import make_portfolio\npc = make_portfolio({kind!r})\npc.fit(panel)                     # 喂入历史收益用于协方差\n"
                f"targets = pc.construct(signals, portfolio_snapshot, market, limits)\n\n# 全栈流水线\npipe = build_sim_pipeline(panel, kinds={{'portfolio': {kind!r}}}); pipe.portfolio.fit(panel)")
    if spec.protocol == "ExecutionStrategy":
        return (f"from autoalgo.registry import make_execution_factory\nfactory = make_execution_factory({kind!r})   # 含挂单库全部策略 + kalman_pid\n"
                f"strategy = factory.create('kalman_pid', exec_order)\n\n# 全栈流水线：kinds={{'execution': {kind!r}}}，母单 strategy_hint='kalman_pid' 或由选择器选择")
    if spec.protocol == "StrategySelector":
        return (f"from autoalgo.registry import make_strategy_selector\nsel = make_strategy_selector({kind!r})\nsel.attach(pipeline)              # 设为 strategy_selector 并订阅 order_done\n\n"
                f"# 全栈流水线：build_sim_pipeline(panel, kinds={{'selector': {kind!r}}})")
    if spec.protocol == "Router":
        return (f"from autoalgo.registry import make_router\nrouter = make_router({kind!r}, venues=['NYSE', 'NASDAQ', 'ARCA', 'IEX'])   # 叠加在 latency_routing 之上\n\n"
                f"# 全栈流水线：build_sim_pipeline(panel, kinds={{'router': {kind!r}}})")
    return f"from autoalgo.registry import make\nobj = make({bp.algo_id!r})"


def emit(bp: AlgorithmBlueprint, spec: FamilySpec, rep: Optional[EvaluationReport], out_dir: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    out_dir = out_dir or GENERATED_DIR
    validate_algo_id(bp.algo_id)
    d = os.path.join(out_dir, bp.algo_id)
    os.makedirs(d, exist_ok=True)
    extra = {"formula": list(spec.formula), "usage_code": usage_code(bp, spec),
             "integration_note": f"quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind={bp.registry_kind!r}` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；"
                                 f"也可用 `autoalgo.registry.make({bp.algo_id!r})` 直接实例化。",
             "repro_cmd": f"python -m autoalgo generate --families {bp.family} --seed {bp.lineage.get('seed', 0)} --budget {bp.lineage.get('budget', 'quick')}",
             **(extra or {})}
    paths = {
        "algorithm": os.path.join(d, "algorithm.py"),
        "markdown": os.path.join(d, f"{bp.algo_id}.md"),
        "blueprint": os.path.join(d, "blueprint.json"),
        "evaluation": os.path.join(d, "evaluation.json"),
    }
    with open(paths["algorithm"], "w", encoding="utf-8") as f:
        f.write(render_algorithm_py(bp, spec, rep))
    with open(paths["markdown"], "w", encoding="utf-8") as f:
        f.write(render_markdown(bp, rep, extra))
    with open(paths["blueprint"], "w", encoding="utf-8") as f:
        f.write(bp.to_json())
    if rep is not None:
        with open(paths["evaluation"], "w", encoding="utf-8") as f:
            f.write(rep.to_json())
    update_index(out_dir, bp, rep)
    return paths


def rerender(out_dir: Optional[str] = None, algo_ids: Optional[List[str]] = None) -> List[str]:
    """不重新评估，仅由已保存的 blueprint.json / evaluation.json 重新生成 algorithm.py、中文说明与索引（模板改动后使用）。"""
    from .families import FAMILIES

    out_dir = out_dir or GENERATED_DIR
    done: List[str] = []
    for e in load_index(out_dir):
        if algo_ids and e["algo_id"] not in algo_ids:
            continue
        d = os.path.join(out_dir, e["algo_id"])
        with open(os.path.join(d, "blueprint.json"), encoding="utf-8") as f:
            bp = AlgorithmBlueprint.from_dict(json.load(f))
        rep = None
        ev = os.path.join(d, "evaluation.json")
        if os.path.exists(ev):
            with open(ev, encoding="utf-8") as f:
                rep = EvaluationReport.from_dict(json.load(f))
        spec = FAMILIES.get(bp.family)
        if spec is None:
            continue
        emit(bp, spec, rep, out_dir=out_dir)
        done.append(bp.algo_id)
    return done


def load_index(out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    out_dir = out_dir or GENERATED_DIR
    p = os.path.join(out_dir, "index.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def update_index(out_dir: str, bp: AlgorithmBlueprint, rep: Optional[EvaluationReport]) -> List[Dict[str, Any]]:
    entries = [e for e in load_index(out_dir) if e.get("algo_id") != bp.algo_id]
    entries.append({
        "algo_id": bp.algo_id, "name_zh": bp.name_zh, "protocol": bp.protocol, "family": bp.family, "registry_kind": bp.registry_kind,
        "objective": rep.objective if rep else "", "primary": rep.primary() if rep else None, "baseline": rep.baseline_primary() if rep else None,
        "higher_is_better": rep.higher_is_better if rep else True, "passed": bool(rep.passed) if rep else False, "created_at": bp.created_at,
        "libraries": bp.libraries, "path": f"{bp.algo_id}/algorithm.py", "markdown": f"{bp.algo_id}/{bp.algo_id}.md",
    })
    entries.sort(key=lambda e: (e["protocol"], e["algo_id"]))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2, default=float)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(render_index_markdown(entries))
    return entries
