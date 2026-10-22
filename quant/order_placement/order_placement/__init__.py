"""挂单策略算法库（order_placement）。

子模块：
- interfaces      跨库数据契约与协议
- microstructure  LOB 微结构特征、成交强度估计
- market_making   做市模型（Avellaneda-Stoikov、GLFT、自适应 GLFT）
- execution       最优执行（Almgren-Chriss、Obizhaeva-Wang、传播子模型、TWAP/VWAP/POV）与执行器
- queue           队列位置、成交概率、限价/市价决策、Cont-Kukanov 多交易所拆分
- rl              强化学习执行环境与基线
- simulation      轻量仿真器
- engine          集成门面 OrderPlacementEngine
"""
from .interfaces import *  # noqa: F401,F403
from .engine import OrderPlacementEngine  # noqa: F401
from .market_making.base import MarketMakerConfig  # noqa: F401
from .market_making.models import AvellanedaStoikov, GLFT, AdaptiveGLFT  # noqa: F401
from .execution.executor import ScheduledExecutor, ExecutorConfig  # noqa: F401

__version__ = "0.1.0"


def make_execution_factory(**cfg):
    """quantstack 集成层入口（需要 /mnt/project-files/trading_system 在 sys.path 中）。"""
    from .adapters.quantstack_exec import make_execution_factory as _mk
    return _mk(**cfg)


def make_impact_model(**cfg):
    from .adapters.quantstack_exec import SqrtLawImpact
    return SqrtLawImpact(**cfg)
