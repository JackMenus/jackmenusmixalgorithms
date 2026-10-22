"""latency_routing —— 路由与延迟优化算法库。

模块：
- interfaces    : 与因子库 / 挂单库 / Agent 对接的数据结构与协议
- latency       : 延迟建模（在线分位数、混合分布、时钟偏差估计）
- sor           : 智能订单路由（Cont-Kukanov 拆分、陈旧报价、同步到达、组合路由器）
- infra         : 低延迟基础设施参考实现（SPSC 无锁队列；C++ 版见 ../cpp）
- sim           : 延迟套利对抗仿真
- agent_adapter : 给集成 Agent 的 JSON 接口与 LLM 工具 Schema
"""
from .interfaces import ChildOrder, ParentOrder, RoutingDecision, VenueQuote, VenueStats, RouterProtocol, LatencyModelProtocol
from .latency.model import VenueLatencyModel, P2Quantile, estimate_clock_offset
from .sor.router import LatencyAwareRouter, RouterConfig, make_router, sweep_allocate
from .sor.cont_kukanov import solve_cont_kukanov
from .sor.sync_send import plan_synchronized_send
from .agent_adapter import RoutingService, TOOL_SCHEMA

__version__ = "0.1.0"
