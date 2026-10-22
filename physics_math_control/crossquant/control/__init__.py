"""控制论模块：滤波、LQR/MPC、HJB、最优执行、做市、变点、老虎机、最优停止、PID、RLS。"""
import importlib as _importlib

# 子模块按需加载（PEP 562）：首次访问 ``crossquant.control.<模块>`` 时才导入，避免一次性加载全部 scipy 子模块。
_SUBMODULES = ('kalman', 'particle_filter', 'lqr', 'mpc', 'hjb', 'execution', 'market_making', 'changepoint', 'bandits', 'optimal_stopping', 'pid', 'rls')
__all__ = list(_SUBMODULES)


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBMODULES))
