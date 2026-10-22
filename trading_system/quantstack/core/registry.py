"""插件注册与自动发现：按名称或 "auto" 选择各层实现（兄弟库优先，参考实现兜底）。

兄弟库（与 ``trading_system/`` 同级的目录下，可用 ``QUANT_PROJECT_ROOT`` 覆盖）：

- ``factormine``       因子挖掘库            → AlphaModel / FactorMiner
- ``order_placement``  挂单策略库            → ExecutionStrategyFactory / ImpactModel
- ``latency_routing``  路由延迟优化库        → Router（经 ``adapters.latency_routing_adapter`` 适配）
- ``crossquant``       物理/数学/控制跨学科库 → StateEstimator / Controller / ImpactModel（结构化子类型，直接登记）
- ``autoalgo``         自主生成算法库（auto_algorithms/）→ 任意协议的生成算法，kind 形如 ``autoalgo.<algo_id>`` /
                       ``autoalgo.family.<family>``；``autoalgo.auto`` 取该协议下第一个评估通过的生成算法
"""
from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, Iterable, Optional, Sequence

from ..contracts import US_PER_SEC, RiskLimits
from .clock import SimClock
from .oms import OrderManager
from .pipeline import PipelineConfig, TradingPipeline
from .risk_gate import PreTradeRiskGate
from .tca import TCAEngine

# 默认取 trading_system/ 的上一级目录（仓库根或共享目录），可用 QUANT_PROJECT_ROOT 覆盖
_DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PROJECT_ROOT = os.environ.get("QUANT_PROJECT_ROOT", _DEFAULT_ROOT)
SIBLING_LIBS: Dict[str, str] = {
    "factormine": os.path.join(PROJECT_ROOT, "factor_mining"),
    "order_placement": os.path.join(PROJECT_ROOT, "quant", "order_placement"),
    "latency_routing": os.path.join(PROJECT_ROOT, "routing_latency"),
    "crossquant": os.path.join(PROJECT_ROOT, "physics_math_control"),
    "autoalgo": os.path.join(PROJECT_ROOT, "auto_algorithms"),
}

# crossquant 各协议在 kind="auto" / "crossquant" 时的默认实现
CROSSQUANT_DEFAULTS: Dict[str, str] = {
    "StateEstimator": "crossquant.KalmanTrendEstimator",
    "Controller": "crossquant.PIDParticipationController",
    "ImpactModel": "crossquant.SquareRootImpactModel",
}

_STATUS: Dict[str, bool] = {}


def discover(names: Optional[Iterable[str]] = None, extra_paths: Optional[Dict[str, str]] = None, refresh: bool = False) -> Dict[str, bool]:
    """把兄弟库目录加入 sys.path 并尝试导入；返回 {库名: 是否可用}。

    只导入 ``names`` 里的库（默认全部），结果按进程缓存：各 ``make_*`` 工厂只触发自己需要的库，
    避免无关的大依赖（如 crossquant 的 scipy）拖慢启动。``refresh=True`` 重新尝试导入。
    """
    if isinstance(names, dict):  # 兼容旧签名 discover(extra_paths)
        extra_paths, names = names, None
    paths = {**SIBLING_LIBS, **(extra_paths or {})}
    wanted = list(names) if names is not None else list(paths)
    status: Dict[str, bool] = {}
    for name in wanted:
        path = paths.get(name)
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
        if refresh or name not in _STATUS:
            try:
                importlib.import_module(name)
                _STATUS[name] = True
            except Exception:
                _STATUS[name] = False
        status[name] = _STATUS[name]
    return status


def _available(name: str) -> bool:
    return bool(discover([name]).get(name))


def _is_autoalgo(kind: str) -> bool:
    return isinstance(kind, str) and (kind == "autoalgo" or kind.startswith("autoalgo."))


def _autoalgo():
    if not _available("autoalgo"):
        raise ImportError(f"autoalgo 不可用（路径 {SIBLING_LIBS['autoalgo']}）")
    from autoalgo import registry as _r

    return _r


def _autoalgo_kind(kind: str) -> str:
    return "auto" if kind == "autoalgo" else kind


# ----------------------------------------------------------------------------- 因子挖掘库
def make_alpha_model(kind: str = "auto", **cfg):
    if _is_autoalgo(kind):
        return _autoalgo().make_alpha_model(_autoalgo_kind(kind), **cfg)
    if kind in ("auto", "factormine") and _available("factormine"):
        import factormine as fm

        return fm.make_alpha_model(**cfg)
    if kind == "factormine":
        raise ImportError("factormine 不可用")
    from ..reference.alpha import ReversalVolumeAlpha

    return ReversalVolumeAlpha(**{k: v for k, v in cfg.items() if k in ("lookback", "vol_window", "horizon", "horizon_us")})


def make_factor_miner(kind: str = "auto", **cfg):
    if kind in ("auto", "factormine") and _available("factormine"):
        import factormine as fm

        return fm.make_miner(**cfg)
    return None


# ----------------------------------------------------------------------------- 挂单策略库
def make_execution_factory(kind: str = "auto", **cfg):
    if _is_autoalgo(kind):  # 组合式工厂：自研策略 + 挂单库全部策略
        return _autoalgo().make_execution_factory(_autoalgo_kind(kind), **cfg)
    if kind in ("auto", "order_placement") and _available("order_placement"):
        import order_placement as op

        return op.make_execution_factory(**cfg)
    if kind == "order_placement":
        raise ImportError("order_placement 不可用")
    from ..reference.execution import ReferenceExecutionFactory

    return ReferenceExecutionFactory()


# ----------------------------------------------------------------------------- 路由延迟优化库
def make_router(kind: str = "auto", venues: Sequence[str] = (), **cfg):
    if _is_autoalgo(kind):  # 叠加层：底层默认为 latency_routing 适配器（不可用时为参考路由）
        return _autoalgo().make_router(_autoalgo_kind(kind), venues=list(venues), base=cfg.pop("base", None) or make_router("auto", venues), **cfg)
    if kind in ("auto", "latency_routing") and _available("latency_routing"):
        from ..adapters.latency_routing_adapter import LatencyRoutingAdapter

        return LatencyRoutingAdapter.create(list(venues), **cfg)
    if kind == "latency_routing":
        raise ImportError("latency_routing 不可用")
    from ..reference.routing import CostLatencyRouter

    return CostLatencyRouter(list(venues), **{k: v for k, v in cfg.items() if k in ("latency_model", "sync_enabled", "oversize_cap", "stale_penalty_bps")})


# ----------------------------------------------------------------------------- 跨学科库（物理 / 数学 / 控制）
def crossquant_entries() -> Dict[str, Dict[str, type]]:
    """crossquant 登记的实现 ``{协议名: {登记名: 类}}``；库不可用（缺 scipy 或目录不存在）时为空字典。"""
    if not _available("crossquant"):
        return {}
    import crossquant

    return crossquant.registry_entries()


def _make_crossquant(protocol: str, kind: str, **cfg):
    entries = crossquant_entries().get(protocol, {})
    if not entries:
        raise ImportError(f"crossquant 不可用（需要 numpy + scipy，路径 {SIBLING_LIBS['crossquant']}）")
    name = CROSSQUANT_DEFAULTS[protocol] if kind in ("auto", "crossquant") else kind
    if name not in entries:
        raise KeyError(f"crossquant 没有 {protocol} 实现 {name!r}；可选: {sorted(entries)}")
    return entries[name](**cfg)


def make_impact_model(kind: str = "auto", **cfg):
    """冲击/成本模型。

    kind: ``auto``（挂单库 → crossquant 平方根律 → None）| ``order_placement`` | ``crossquant``（平方根律）
    | ``crossquant.SquareRootImpactModel`` | ``crossquant.AlmgrenChrissImpactModel``；其他取值返回 None。
    """
    if kind in ("auto", "order_placement") and _available("order_placement"):
        import order_placement as op

        return op.make_impact_model(**cfg)
    if kind == "order_placement":
        raise ImportError("order_placement 不可用")
    if kind == "auto":
        try:
            return _make_crossquant("ImpactModel", "auto", **cfg)
        except ImportError:
            return None
    if kind.startswith("crossquant"):
        return _make_crossquant("ImpactModel", kind, **cfg)
    return None


def make_state_estimator(kind: str = "auto", **cfg):
    """状态估计器（公允价 / 价差 / 延迟的隐状态滤波与外推）。

    kind: ``auto`` / ``crossquant``（Kalman 局部线性趋势）| ``crossquant.KalmanTrendEstimator`` | ``crossquant.OUSpreadEstimator``。
    ``auto`` 在 crossquant 不可用时返回 None，显式名称不可用时抛 ImportError / KeyError。
    """
    if kind == "auto" or kind == "crossquant" or kind.startswith("crossquant."):
        try:
            return _make_crossquant("StateEstimator", kind, **cfg)
        except ImportError:
            if kind == "auto":
                return None
            raise
    return None


def make_controller(kind: str = "auto", **cfg):
    """控制器（执行进度跟踪 → 参与率调节）。

    kind: ``auto`` / ``crossquant``（抗饱和 PID）| ``crossquant.PIDParticipationController`` | ``crossquant.LQRTrackingController``。
    """
    if kind == "auto" or kind == "crossquant" or kind.startswith("crossquant."):
        try:
            return _make_crossquant("Controller", kind, **cfg)
        except ImportError:
            if kind == "auto":
                return None
            raise
    return None


# ----------------------------------------------------------------------------- 集成层自带
def make_portfolio(kind: str = "reference", **cfg):
    if _is_autoalgo(kind):
        return _autoalgo().make_portfolio(_autoalgo_kind(kind), **cfg)
    from ..reference.portfolio import LongShortConstructor

    return LongShortConstructor(**cfg)


def make_strategy_selector(kind: str = "reference", **cfg):
    """执行策略选择器：``reference``（规则 choose_strategy）或 ``autoalgo.<algo_id>`` / ``autoalgo.auto``（上下文老虎机等生成算法）。

    返回的对象是 ``(order, snapshot, available) -> (name, params)`` 的可调用对象；若有 ``attach(pipeline)`` 方法，
    ``build_sim_pipeline`` 会调用它以订阅 ``order_done`` 回填奖励。
    """
    if _is_autoalgo(kind):
        return _autoalgo().make_strategy_selector(_autoalgo_kind(kind), **cfg)
    from ..reference.execution import choose_strategy

    return choose_strategy


def plugin_registry() -> Dict[str, Dict[str, str]]:
    """全部可用实现一览 ``{协议名: {登记名: 来源目录}}``（会导入所有兄弟库，用于 CLI / agent 展示）。"""
    ref = "quantstack.reference"
    reg: Dict[str, Dict[str, str]] = {
        "AlphaModel": {"reference.ReversalVolumeAlpha": ref},
        "PortfolioConstructor": {"reference.LongShortConstructor": ref},
        "ExecutionStrategyFactory": {"reference.ReferenceExecutionFactory": ref},
        "Router": {"reference.CostLatencyRouter": ref},
        "ImpactModel": {},
        "StateEstimator": {},
        "Controller": {},
    }
    st = discover()
    if st.get("factormine"):
        reg["AlphaModel"]["factormine.make_alpha_model"] = SIBLING_LIBS["factormine"]
        reg["FactorMiner"] = {"factormine.make_miner": SIBLING_LIBS["factormine"]}
    if st.get("order_placement"):
        reg["ExecutionStrategyFactory"]["order_placement.make_execution_factory"] = SIBLING_LIBS["order_placement"]
        reg["ImpactModel"]["order_placement.make_impact_model"] = SIBLING_LIBS["order_placement"]
    if st.get("latency_routing"):
        reg["Router"]["latency_routing.make_router"] = SIBLING_LIBS["latency_routing"]
    for proto, impls in crossquant_entries().items():
        reg.setdefault(proto, {}).update({name: SIBLING_LIBS["crossquant"] for name in impls})
    if st.get("autoalgo"):
        for proto, impls in _autoalgo().registry_entries().items():
            key = "ExecutionStrategyFactory" if proto == "ExecutionStrategy" else proto  # 生成的执行策略以组合式工厂的形式接入
            reg.setdefault(key, {}).update({name: SIBLING_LIBS["autoalgo"] for name in impls})
    return {k: v for k, v in reg.items() if v}


def build_sim_pipeline(
    panel,
    as_of=None,
    *,
    kinds: Optional[Dict[str, str]] = None,
    seed: int = 0,
    initial_cash: float = 10_000_000.0,
    limits: Optional[RiskLimits] = None,
    config: Optional[PipelineConfig] = None,
    venues=None,
    portfolio_cfg: Optional[dict] = None,
    alpha_cfg: Optional[dict] = None,
    exec_cfg: Optional[dict] = None,
    router_cfg: Optional[dict] = None,
    start_us: int = 0,
    selector_cfg: Optional[dict] = None,
) -> TradingPipeline:
    """用合成日内市场 + 撮合模拟器搭一条完整流水线（演示 / 测试 / agent 训练场）。

    ``kinds`` 可含 ``alpha / execution / router / portfolio / selector``；``portfolio`` 默认 ``reference``，``selector`` 默认 ``reference``
    （规则选择器）。自主生成算法库的实现用 ``autoalgo.<algo_id>``（或 ``autoalgo.auto``）指定，例如
    ``kinds={"alpha": "autoalgo.regime_gated_alpha", "execution": "autoalgo.kalman_pid_execution", "selector": "autoalgo.bandit_strategy_selector"}``。
    """
    from ..sim.market import DEFAULT_VENUES, SyntheticMarket, panel_stats
    from ..sim.venue import VenueSimulator

    kinds = {"alpha": "auto", "execution": "auto", "router": "auto", "portfolio": "reference", "selector": "reference", **(kinds or {})}
    venues = list(venues or DEFAULT_VENUES)
    st = panel_stats(panel, as_of)
    market = SyntheticMarket(st["price"], st["volatility"], st["adv"], venues=venues, start_us=start_us, seed=seed)
    gateway = VenueSimulator(market, seed=seed + 1)
    clock = SimClock(start_us)
    alpha = make_alpha_model(kinds["alpha"], **(alpha_cfg or {}))
    exec_factory = make_execution_factory(kinds["execution"], **(exec_cfg or {}))
    router = make_router(kinds["router"], [v.name for v in venues], **(router_cfg or {}))
    portfolio = make_portfolio(kinds["portfolio"], **(portfolio_cfg or {}))
    selector = make_strategy_selector(kinds["selector"], **(selector_cfg or {}))
    pipe = TradingPipeline(
        alpha_model=alpha,
        portfolio=portfolio,
        execution_factory=exec_factory,
        router=router,
        gateway=gateway,
        feed=market,
        clock=clock,
        oms=OrderManager(initial_cash=initial_cash),
        risk_gate=PreTradeRiskGate(limits or RiskLimits()),
        tca=TCAEngine(),
        config=config or PipelineConfig(),
        strategy_selector=selector,
    )
    if hasattr(selector, "attach"):
        selector.attach(pipe)
    if hasattr(portfolio, "fit") and kinds["portfolio"] != "reference":
        portfolio.fit(panel, as_of)
    return pipe
