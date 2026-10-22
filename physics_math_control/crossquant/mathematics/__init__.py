"""数学模块：最优传输、路径签名、信息几何、SDE/QMC、小波、分数阶差分、Koopman/DMD、TDA、极值理论、稀疏恢复、Kelly、谱方法、排队论、Copula、动力系统、流形学习。"""
import importlib as _importlib

# 子模块按需加载（PEP 562）：首次访问 ``crossquant.mathematics.<模块>`` 时才导入，避免一次性加载全部 scipy 子模块。
_SUBMODULES = ('optimal_transport', 'signatures', 'information_geometry', 'sde', 'wavelets', 'fracdiff', 'timeseries', 'koopman', 'tda', 'extreme_value', 'sparse', 'kelly', 'spectral', 'queueing', 'copulas', 'dynamical_systems', 'manifold')
__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
