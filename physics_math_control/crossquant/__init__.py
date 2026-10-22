"""
crossquant —— 跨学科量化算法库
==================================

收集 **物理学 / 数学 / 控制论** 中可以借鉴到量化交易的算法，每个模块都包含：

1. 学科来源（这个算法最初解决什么问题）
2. 量化映射（它在交易中对应什么问题）
3. 核心公式与实现（仅依赖 numpy / scipy）
4. 参考文献（经典 + 前沿）

子包
----
- ``crossquant.control``      控制论：滤波、LQR/MPC、HJB 随机控制、最优执行、做市、变点、多臂老虎机、最优停止
- ``crossquant.physics``      物理：OU/Langevin、Hawkes、随机矩阵、Hurst/DFA、熵与传递熵、网络/HRP、模拟退火、首达时间、粗糙波动率、MCMC、PDE、Ising
- ``crossquant.mathematics``  数学：最优传输、路径签名、信息几何、SDE/蒙特卡洛、小波、分数阶差分、Koopman/DMD、拓扑数据分析、极值理论、稀疏恢复、Kelly、谱方法、排队论、Copula、动力系统、流形学习
- ``crossquant.pipelines``    端到端流水线：配对交易、组合风险、市场状态仪表盘
- ``crossquant.adapters``     对接全栈交易系统（quantstack）的 StateEstimator / Controller / ImpactModel 适配器

使用约定
--------
- 所有函数接受 ``numpy.ndarray``（必要时自动转换），返回 ``numpy`` 数组或 ``dict``；
- 随机数统一使用 ``numpy.random.Generator``，通过 ``seed`` 或 ``rng`` 参数控制；
- 时间序列默认按时间升序排列，形状 ``(T,)`` 或 ``(T, n)``。
"""

import importlib as _importlib

# 子包按需加载（PEP 562）：``import crossquant`` 本身很轻，首次访问 ``crossquant.physics`` 等时才导入，
# 便于全栈系统的注册表快速启动；``from crossquant.control import kalman`` 等写法不受影响。
_SUBPACKAGES = ("control", "physics", "mathematics", "pipelines")


def __getattr__(name: str):
    if name in _SUBPACKAGES:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SUBPACKAGES))


from . import adapters  # noqa: E402,F401  —— 适配器只依赖 control 下的少数轻量模块
from .adapters import (  # noqa: F401  —— 顶层再导出：使注册名 "crossquant.XXX" 同时就是合法导入路径
    AlmgrenChrissImpactModel,
    KalmanTrendEstimator,
    LQRTrackingController,
    OUSpreadEstimator,
    PIDParticipationController,
    SquareRootImpactModel,
    registry_entries,
)

__version__ = "0.1.0"
__all__ = [
    "control", "physics", "mathematics", "pipelines", "adapters", "__version__",
    "KalmanTrendEstimator", "OUSpreadEstimator", "PIDParticipationController", "LQRTrackingController",
    "SquareRootImpactModel", "AlmgrenChrissImpactModel", "registry_entries",
]
