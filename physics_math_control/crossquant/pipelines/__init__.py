"""端到端流水线：配对交易、组合风险、市场状态仪表盘。"""
import importlib as _importlib

# 子模块按需加载（PEP 562）：首次访问 ``crossquant.pipelines.<模块>`` 时才导入，避免一次性加载全部 scipy 子模块。
_SUBMODULES = ('pairs_trading', 'portfolio_risk', 'regime')
__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
