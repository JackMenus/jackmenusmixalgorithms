"""物理模块：OU/Langevin、Hawkes、随机矩阵、Hurst/DFA、熵、网络/HRP、模拟退火、首达时间、粗糙波动率、MCMC、PDE、Ising。"""
import importlib as _importlib

# 子模块按需加载（PEP 562）：首次访问 ``crossquant.physics.<模块>`` 时才导入，避免一次性加载全部 scipy 子模块。
_SUBMODULES = ('ou_process', 'hawkes', 'random_matrix', 'hurst', 'entropy', 'networks', 'annealing', 'first_passage', 'rough_volatility', 'mcmc', 'pde', 'ising')
__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
