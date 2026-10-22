"""兄弟库定位与按需导入。

与 ``quantstack.core.registry`` 保持同一约定：默认在本目录（``auto_algorithms/``）的上一级目录寻找兄弟库，
可用环境变量 ``QUANT_PROJECT_ROOT`` 覆盖。导入结果按进程缓存。
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_ROOT = os.path.dirname(HERE)                       # auto_algorithms/
_DEFAULT_ROOT = os.path.dirname(LIB_ROOT)              # 仓库根 / 共享目录
PROJECT_ROOT = os.environ.get("QUANT_PROJECT_ROOT", _DEFAULT_ROOT)

SIBLING_LIBS: Dict[str, str] = {
    "quantstack": os.path.join(PROJECT_ROOT, "trading_system"),
    "factormine": os.path.join(PROJECT_ROOT, "factor_mining"),
    "order_placement": os.path.join(PROJECT_ROOT, "quant", "order_placement"),
    "latency_routing": os.path.join(PROJECT_ROOT, "routing_latency"),
    "crossquant": os.path.join(PROJECT_ROOT, "physics_math_control"),
}

GENERATED_DIR = os.environ.get("AUTOALGO_GENERATED_DIR", os.path.join(LIB_ROOT, "generated"))

_STATUS: Dict[str, bool] = {}


def ensure_siblings(names: Optional[Iterable[str]] = None, refresh: bool = False) -> Dict[str, bool]:
    """把兄弟库目录加入 ``sys.path`` 并尝试导入，返回 ``{库名: 是否可用}``。"""
    wanted = list(names) if names is not None else list(SIBLING_LIBS)
    out: Dict[str, bool] = {}
    for name in wanted:
        path = SIBLING_LIBS.get(name)
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
        if refresh or name not in _STATUS:
            try:
                importlib.import_module(name)
                _STATUS[name] = True
            except Exception:
                _STATUS[name] = False
        out[name] = _STATUS[name]
    return out


def available(name: str) -> bool:
    return bool(ensure_siblings([name]).get(name))


def require(*names: str) -> None:
    missing = [n for n in names if not available(n)]
    if missing:
        raise ImportError(f"需要兄弟库 {missing}（路径：{[SIBLING_LIBS[m] for m in missing]}）")
