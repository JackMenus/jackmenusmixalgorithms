"""跨库协议（Protocol）。每个算法库只需实现其中一组方法即可被流水线与 agent 接入。

实现方不需要继承这些类（结构化子类型），但建议在库的测试里加一句
``assert isinstance(obj, quantstack.contracts.ExecutionStrategy)`` 做自检。

数据流与协议的对应关系::

    Panel ──AlphaModel.predict──▶ SignalBatch ──PortfolioConstructor.construct──▶ [TargetPosition]
          ──OMS 差分 + RiskGate──▶ [ExecutionOrder] ──ExecutionStrategy.on_market──▶ [OrderSlice]
          ──Router.route──▶ RoutingDecision{[VenueOrder]} ──VenueGateway.submit──▶ 交易所
          ◀──VenueGateway.poll── [OrderEvent/Fill] ──▶ Router.on_event / ExecutionStrategy.on_fill / OMS
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from .market import MarketSnapshot, Panel
from .orders import CancelRequest, ExecutionOrder, Fill, OrderEvent, OrderSlice, RoutingDecision, VenueOrder
from .portfolio import PortfolioSnapshot, TargetPosition
from .risk import RiskLimits
from .signals import FactorReport, FactorSpec, SignalBatch
from .tca import VenueStats


# ----------------------------------------------------------------------------- 因子挖掘库
@runtime_checkable
class FactorMiner(Protocol):
    """因子挖掘器：给定面板数据，在预算内产出候选因子并评估。"""

    def mine(self, panel: Panel, budget: int = 100, **kwargs) -> List[FactorSpec]: ...

    def evaluate(self, spec: FactorSpec, panel: Panel, **kwargs) -> FactorReport: ...


@runtime_checkable
class AlphaModel(Protocol):
    """alpha 模型：把（一个或多个）因子组合成对每个标的的收益预测。

    ``predict`` 只能使用 ``as_of`` 及之前的数据（无未来函数），返回 ``as_of`` 时刻的信号。
    """

    model_id: str

    def fit(self, panel: Panel, **kwargs) -> None: ...

    def predict(self, panel: Panel, as_of=None, **kwargs) -> SignalBatch: ...

    def describe(self) -> Dict[str, object]: ...


# ----------------------------------------------------------------------------- 组合层（集成层自带，可被跨学科库替换）
@runtime_checkable
class PortfolioConstructor(Protocol):
    def construct(
        self,
        signals: SignalBatch,
        portfolio: PortfolioSnapshot,
        market: Dict[str, MarketSnapshot],
        limits: RiskLimits,
    ) -> List[TargetPosition]: ...


# ----------------------------------------------------------------------------- 挂单策略库
@runtime_checkable
class ExecutionStrategy(Protocol):
    """一个母单的执行算法实例（事件驱动，一次只服务一张母单）。

    生命周期：``start`` -> 多次 ``on_market`` / ``on_fill`` / ``on_event`` -> ``is_complete``。
    每个回调返回新的子单切片（可为空）；需要撤单时返回 ``CancelRequest``。
    """

    name: str

    def start(self, order: ExecutionOrder, snapshot: MarketSnapshot) -> List[OrderSlice]: ...

    def on_market(self, snapshot: MarketSnapshot) -> List[OrderSlice]: ...

    def on_fill(self, fill: Fill, snapshot: MarketSnapshot) -> List[OrderSlice]: ...

    def on_event(self, event: OrderEvent, snapshot: MarketSnapshot) -> List[OrderSlice]: ...

    def cancels(self, snapshot: MarketSnapshot) -> List[CancelRequest]: ...

    def is_complete(self) -> bool: ...

    def state(self) -> Dict[str, float]: ...


@runtime_checkable
class ExecutionStrategyFactory(Protocol):
    """按名称创建执行策略；``available`` 列出支持的策略名与参数说明，供 agent 选择。"""

    def create(self, name: str, order: ExecutionOrder, **params) -> ExecutionStrategy: ...

    def available(self) -> Dict[str, str]: ...


@runtime_checkable
class ImpactModel(Protocol):
    """市场冲击/成本模型（跨学科库或挂单库提供），用于执行计划与 agent 的成本-alpha 权衡。"""

    def expected_cost_bps(self, quantity: int, snapshot: MarketSnapshot, duration_us: int) -> float: ...


# ----------------------------------------------------------------------------- 路由延迟优化库
@runtime_checkable
class LatencyModel(Protocol):
    """与 ``latency_routing.interfaces.LatencyModelProtocol`` 相同。"""

    def observe(self, venue: str, latency_us: float) -> None: ...

    def quantile(self, venue: str, q: float) -> float: ...

    def sample(self, venue: str, n: int) -> Sequence[float]: ...


@runtime_checkable
class Router(Protocol):
    """智能路由器：把一张子单切片拆到若干场所，并决定发送时刻。"""

    def route(self, order_slice: OrderSlice, snapshot: MarketSnapshot) -> RoutingDecision: ...

    def on_event(self, event: OrderEvent) -> None: ...

    def venue_stats(self) -> List[VenueStats]: ...


# ----------------------------------------------------------------------------- 网关 / 行情（集成层自带：模拟、纸面、实盘）
@runtime_checkable
class VenueGateway(Protocol):
    def submit(self, order: VenueOrder) -> None: ...

    def cancel(self, request: CancelRequest) -> None: ...

    def poll(self, ts_us: int) -> List[OrderEvent]: ...


@runtime_checkable
class MarketDataFeed(Protocol):
    def snapshot(self, symbol: str, ts_us: int) -> MarketSnapshot: ...

    def symbols(self) -> List[str]: ...


# ----------------------------------------------------------------------------- 跨学科库（物理/数学/控制）可选钩子
@runtime_checkable
class StateEstimator(Protocol):
    """状态估计器（Kalman / 粒子滤波 等）：对价格、价差、延迟等隐状态做滤波。"""

    def update(self, observation: float, ts_us: int) -> float: ...

    def predict(self, horizon_us: int) -> float: ...


@runtime_checkable
class Controller(Protocol):
    """控制器（PID / LQR / MPC 等）：根据跟踪误差输出控制量，如执行进度的参与率调节。"""

    def step(self, error: float, dt_us: int) -> float: ...
