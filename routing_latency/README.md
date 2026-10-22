# latency_routing —— 路由与延迟优化算法库

全栈交易系统中「因子挖掘 → 挂单策略 → **路由与延迟优化**」的最后一环：
把母单拆成多场所子单，决定吃单/挂单比例、每个场所的数量，以及为了对抗延迟套利而设定的发送时间。

- 依赖：Python ≥ 3.10、numpy（测试需要 pytest）。C++ 部分只需 C++17 编译器。
- 时间单位一律为**微秒**，`side = +1` 买 / `-1` 卖。
- 理论背景见 [docs/研究综述.md](docs/研究综述.md)。

## 目录

```
routing_latency/
├── latency_routing/
│   ├── interfaces.py          # 对接契约：VenueQuote / ParentOrder / ChildOrder / RoutingDecision + 协议
│   ├── latency/model.py       # 延迟建模：P² 在线分位数、对数正态+突刺混合、时钟偏差估计
│   ├── sor/
│   │   ├── cont_kukanov.py    # Cont-Kukanov 多场所最优挂单拆分
│   │   ├── stale_quote.py     # 报价陈旧/被抢单风险
│   │   ├── sync_send.py       # 同步到达调度（THOR 类，随机延迟下的鲁棒版本）
│   │   └── router.py          # 组合路由器 LatencyAwareRouter
│   ├── infra/ring_buffer.py   # SPSC 无锁环形队列（参考实现）
│   ├── sim/latency_arb_sim.py # 延迟套利对抗仿真
│   └── agent_adapter.py       # 给集成 Agent 的 JSON 接口与 LLM 工具 Schema
├── cpp/spsc_ring.hpp          # 生产用 C++17 无锁队列 + spsc_bench.cpp 基准
├── examples/demo.py           # 端到端演示
├── tests/test_all.py          # 11 个测试
└── docs/研究综述.md
```

## 快速开始

```bash
cd routing_latency
pip install numpy pytest
python3 -m pytest -q tests        # 11 passed
python3 examples/demo.py
g++ -O3 -std=c++17 -pthread cpp/spsc_bench.cpp -o spsc_bench && ./spsc_bench
```

```python
from latency_routing import RoutingService, ParentOrder, VenueQuote

svc = RoutingService()
svc.observe_latency("SSE", 180.0)   # 用下单→确认回报持续喂入单程延迟
decision = svc.route(
    ParentOrder("600000", side=+1, quantity=1000, urgency=0.7, alpha_bps=3.0),
    [VenueQuote("SSE", 9.99, 10.00, 500, 800, taker_fee=0.0003, maker_fee=0.0)],
)
for c in decision.children:
    print(c.venue, c.order_type, c.quantity, c.price, c.send_delay_us)
```

## 路由器逻辑（`LatencyAwareRouter.route`）

1. **有效 urgency** = 挂单策略给出的 `urgency` + 因子 alpha 加成（alpha 超过穿越成本的部分，最多 +0.5）。
2. **吃单部分**（数量 = 母单 × urgency）：`sweep_allocate` 按含费有效价格排序，按 `E[exp(-λL)]` 折算陈旧报价后的可成交量；多场所时用 `plan_synchronized_send` 计算每个场所的 `send_delay_us`，让子单在对手反应窗口内同时到达。
3. **挂单部分**（剩余 + 吃单预计吃不到的部分）：Cont-Kukanov 在 NBBO 场所之间拆分，可能同时给出少量市价单。
4. 输出 `RoutingDecision`：子单列表、预期成本（bps，相对中间价）、预期成交率、诊断信息。

### 重要约定
- **超额挂单**：Cont-Kukanov 在成交不确定时会让挂单总量大于目标量（见 `diagnostics.passive_posted_qty` 与 `passive_target_qty`）。下游执行层**必须在母单完成时撤销剩余挂单**。
- `oversize_cap` 默认 1.0（关闭）。打开后 IOC 会探测隐藏流动性，但多个场所叠加可能导致超成交。
- `send_delay_us` 是相对于决策时刻的延时，由执行层（或 FPGA/网卡定时发送）负责精确执行。
- 默认的流出量样本 `default_outflow_samples` 只是占位，实盘应换成历史回放或 Hawkes 模拟。

## 与其他两个库及 Agent 的对接

| 方向 | 字段 | 说明 |
|---|---|---|
| 因子挖掘库 → 本库 | `ParentOrder.alpha_bps`, `alpha_horizon_us` | 按交易方向为正表示有利 |
| 挂单策略库 → 本库 | `ParentOrder.quantity / urgency / limit_price` | urgency ∈ [0,1] |
| 行情 → 本库 | `VenueQuote` 列表 | 含费用、返佣、报价刷新强度 |
| 执行回报 → 本库 | `RoutingService.observe_latency(venue, us)` | 持续更新延迟模型 |
| 本库 → 执行层/挂单策略库 | `RoutingDecision.to_dict()` | JSON 安全 |

集成 Agent 只需依赖 `RouterProtocol`，或者使用 `RoutingService.route_json(dict) -> dict`。`RouterProtocol` 包含三个方法：
- `route(order, quotes) -> RoutingDecision`
- `on_event(event) -> None`：接收回报。fill/partial 事件用 `fill.latency_us / 2` 作为单程延迟喂入延迟模型，并更新场所统计；reject 事件计入拒单数。参数是鸭子类型，兼容 `quantstack.contracts.orders.OrderEvent`。
- `venue_stats() -> list[VenueStats]`：返回每个场所的成交数、拒单数、成交率、滑点、延迟 P50/P99 和 maker 占比。

工厂函数为 `make_router(venues, latency_prior_us={...}, **RouterConfig字段)`，返回 `LatencyAwareRouter`。

`send_delay_us` 的时间基准是 `route()` 返回、执行层开始派发的时刻 t0。同一次决策的所有子单共享 t0，执行层应在 t0 + send_delay_us 时发出。
`agent_adapter.TOOL_SCHEMA` 可以直接注册为 LLM 工具（Claude tool use / MCP）。
LLM 只做编排和参数选择（例如根据市况调 `reaction_window_us`），**不在下单热路径上**，热路径全部是确定性数值计算。

## 验证情况
- 11 个单元/集成测试全部通过：P² 分位数与 numpy 误差 < 3%；时钟偏差估计误差 < 10µs；Cont-Kukanov 成本不高于均匀拆分，且符合论文「λ_u < h+f 不用市价单」的结论；同步发送在仿真中比立即发送成交率高 20 个百分点以上。
- `demo.py`（合成数据，对手反应窗口 150µs）：同步发送平均成交率约 92%，立即发送约 11%。
- C++ 队列：本机约 5 ns/条（云容器，未绑核）。
- **尚未用真实交易所数据回测**。所有参数（费用、刷新强度、延迟）都是示例值。
