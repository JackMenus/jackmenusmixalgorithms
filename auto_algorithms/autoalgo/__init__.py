"""autoalgo —— 自主生成算法库。

在五个既有库（factormine / order_placement / latency_routing / crossquant / quantstack）之上，
按 quantstack 协议拼装新算法，自动评估、生成代码与中文可解释性文档，并放入 ``generated/``。

模块一览
  _paths      兄弟库定位与按需导入
  blueprint   AlgorithmBlueprint / EvaluationReport 数据结构
  catalog     组件目录（可用积木）
  components  新算法实现（regime / operators / alpha / portfolio / execution / selector / routing）
  families    模板族：构造方式 + 搜索空间 + 可解释性文本骨架
  search      参数提议器（随机 / 进化）
  llm         Claude 提议器（Fable 5.1 最高思考强度）与离线替身
  evaluate    评估台（各协议层 vs 基线）
  explain     中文 Markdown 渲染
  codegen     生成 generated/<algo_id>/{algorithm.py, <algo_id>.md, blueprint.json, evaluation.json} 与 index.json
  pipeline    generate(panel, GenerationConfig)
  registry    make_* 工厂与 kind 解析（供 quantstack.core.registry 调用）
  cli         python -m autoalgo generate | list | show | evaluate | families | components

``import autoalgo`` 本身很轻（不导入 numpy/pandas 之外的重依赖），子模块按需加载。
"""
from __future__ import annotations

import importlib as _importlib

from ._paths import GENERATED_DIR, LIB_ROOT, PROJECT_ROOT, SIBLING_LIBS, available, ensure_siblings, require  # noqa: F401
from .blueprint import GENERATOR_VERSION, PROTOCOLS, AlgorithmBlueprint, ComponentRef, EvaluationReport  # noqa: F401

__version__ = "0.1.0"

_LAZY = {
    "FAMILIES": ("families", "FAMILIES"), "FamilySpec": ("families", "FamilySpec"),
    "GenerationConfig": ("pipeline", "GenerationConfig"), "generate": ("pipeline", "generate"), "generate_family": ("pipeline", "generate_family"), "default_panel": ("pipeline", "default_panel"),
    "evaluate": ("evaluate", "evaluate"), "render_markdown": ("explain", "render_markdown"), "catalog": ("catalog", "catalog"),
    "make": ("registry", "make"), "make_alpha_model": ("registry", "make_alpha_model"), "make_portfolio": ("registry", "make_portfolio"),
    "make_execution_factory": ("registry", "make_execution_factory"), "make_router": ("registry", "make_router"), "make_strategy_selector": ("registry", "make_strategy_selector"),
    "registry_entries": ("registry", "registry_entries"), "available_kinds": ("registry", "available_kinds"),
}
_SUBMODULES = ("blueprint", "catalog", "components", "families", "search", "llm", "evaluate", "explain", "codegen", "pipeline", "registry", "cli")


def __getattr__(name: str):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(_importlib.import_module(f"{__name__}.{mod}"), attr)
    if name in _SUBMODULES:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_SUBMODULES))
