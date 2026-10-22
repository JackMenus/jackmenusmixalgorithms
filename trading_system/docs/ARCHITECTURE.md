# 全栈量化交易系统架构（quantstack）

> 本文描述集成层的设计：如何把因子挖掘库、挂单策略库、路由延迟优化库（以及可选的跨学科算法库）
> 串成一条**同一份代码可回测、可模拟、可实盘**的流水线，并由 LLM agent 在明确的决策点上介入。
> 接口细节见 [INTERFACES.md](INTERFACES.md)，模型路由见 [MODEL_POLICY.md](MODEL_POLICY.md)。

## 1. 设计目标

1. **松耦合**：四个库并行开发、互不 import，只共享 `quantstack.contracts`（纯 dataclass + Protocol）。
2. **确定性优先**：账本（OMS）、事前风控、TCA、事件循环全是确定性代码；LLM 只产出结构化决策，且每个决策都被校验。
3. **闭环反馈**：成交/拒单/延迟沿原路回传：路由层更新场所统计与延迟模型，执行层更新剩余量，OMS 更新持仓，TCA 产出成本，agent 据此复盘调参。
4. **可解释、可回放**：所有层间交互发布到事件总线并写入日志；agent 的每个决策（模型、思考强度、输入、输出、是否被采纳）写入 `decisions.jsonl`。
5. **模型分级**：强推理阶段用 Claude Fable 5.1 `effort=max`，其余用 Opus 5.5 / Fable 5.1 自适应阶梯。

## 2. 分层

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  quantstack.agent  TradingAgent                                              │
│   决策点：数据质量 · 研究评审* · 执行规划 · 交易后复盘 → 深度复盘* · 事故响应* · 日报   │
│   ModelPolicy（* = Fable 5.1 effort=max；其余 Opus 5.5 → Fable 5.1 自适应阶梯）   │
│   ToolBox：只读查询 + 有边界调参 + 熔断            ScriptedBackend：离线确定性基线   │
├──────────────────────────────────────────────────────────────────────────────┤
│  quantstack.core  TradingPipeline（事件循环） · EventBus · Clock · Registry       │
│   OMS（账本/差分/状态机） · PreTradeRiskGate（母单/子单/盈亏） · TCAEngine          │
├────────────┬───────────────┬────────────────┬────────────────┬────────────────┤
│ 因子挖掘库   │ 组合层          │ 挂单策略库        │ 路由延迟库        │ 网关/行情        │
│ AlphaModel │ Portfolio-    │ Execution-     │ Router         │ VenueGateway   │
│ FactorMiner│ Constructor   │ Strategy(+Factory)│ LatencyModel │ MarketDataFeed │
│ factormine │ reference /   │ order_placement│ latency_routing│ sim / paper /  │
│            │ 跨学科库优化器    │ / reference    │ / reference    │ live           │
├────────────┴───────────────┴────────────────┴────────────────┴────────────────┤
│  quantstack.contracts  Panel · SignalBatch/AlphaSignal · TargetPosition ·       │
│   ExecutionOrder · OrderSlice · VenueOrder · RoutingDecision · Fill/OrderEvent · │
│   RiskLimits/RiskVerdict/Incident · TCAReport/VenueStats · 协议（Protocol）       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 3. 数据流（一次每日周期）

| 步骤 | 输入 → 输出 | 实现者 | agent 介入 |
|---|---|---|---|
| 0 数据质量 | Panel → DataQualityReport | 确定性检查（缺失/陈旧/跳变） | 有问题时自适应解释 |
| 1 信号 | Panel → SignalBatch | 因子挖掘库 `AlphaModel.predict`（只用 as_of 之前数据） | 研究评审*（部署/下线因子） |
| 2 目标持仓 | SignalBatch + 持仓 + 行情 → [TargetPosition] | 组合层（多空、单票上限、成本门槛、紧急度） | — |
| 3 母单 | 目标 − 持仓 → [ExecutionOrder]，经风控 | OMS 差分 + PreTradeRiskGate | — |
| 4 执行规划 | 母单特征 → {order_id: 策略, 参数, 紧急度} | agent（自适应），缺省 `choose_strategy` | 执行规划 |
| 5 事件循环 | 每步：回报 → 策略决策 → 子单风控 → 路由 → 网关 | TradingPipeline | （实时工具可查询） |
| 6 复盘 | TCA + 场所统计 + 事件 → PostTradeReview | agent（自适应）→ 升级深度复盘* | 调参（有边界） |
| 7 事故 | kill switch → IncidentResponse | agent（强推理）；恢复交易必须人工 | 事故响应* |
| 8 日报 | 周期摘要 → DailyReport | agent（自适应） | — |

## 4. 事件循环（`TradingPipeline.step`）

```
now = clock.now_us()
1. 处理网关回报 gateway.poll(now)：
   fill    → OMS 记账 · TCA · Router.on_event · Strategy.on_fill · 切片满额则撤掉该切片剩余场所订单
   reject/cancel/expire → 释放该场所订单的在途量 · Strategy.on_event（被流水线主动撤掉的单不重复通知）
2. 对每张未完成母单：
   snapshot = feed.snapshot(symbol, now)；首次 → Strategy.start，否则 Strategy.on_market
   新切片 → OMS 在途上限裁剪 → RiskGate.check_slice（价格带/频率/熔断）→ Router.route
         → 场所订单号改写为 "{slice_id}-{k}" → 未足额路由的余量立即释放 → gateway.submit
   Strategy.cancels() → 按场所订单或整切片撤单
   过期切片（ttl）→ 撤单 + 通知策略 expire
   完成或超过窗口+宽限 → 收尾：撤掉在途、置状态、TCA 结束
3. RiskGate.check_pnl（当日亏损上限）→ 触发 Incident 则全部收尾并熔断
clock.advance(step)
```

同一循环在 `SimClock + VenueSimulator` 下回放，在 `WallClock + 真实网关` 下实盘（网关需实现 `VenueGateway`/`MarketDataFeed`）。

## 5. 执行簿记与安全机制

| 机制 | 位置 | 作用 |
|---|---|---|
| 在途量上限 | `pipeline._dispatch` | 其他切片在途 + 新切片 ≤ 母单剩余；缩量时以 `cancel/oms:in_flight_cap_resize` 通知策略 |
| 路由超额裁剪 | `pipeline._dispatch` | 路由层可超额挂单（Cont-Kukanov），但总量不超过母单剩余 |
| 未路由余量释放 | `pipeline._dispatch` | 路由只覆盖可见量时，余量以 `cancel/unrouted` 回给策略，避免幽灵在途量 |
| 切片 TTL | `pipeline._expire_slices` | 被动切片超时撤单并通知策略重切 |
| 切片满额撤单 | `pipeline._handle_event` | 切片成交量达到目标后撤掉其余场所订单（超额挂单的收尾） |
| 撤单竞态 | `VenueSimulator._on_cancel` | 撤单先于订单到达时标记「到达即撤」 |
| 事前风控 | `PreTradeRiskGate` | 单笔数量/名义、持仓/总/净敞口、做空、参与率、价格带（乌龙指）、子单频率 |
| 熔断 | `PreTradeRiskGate.check_pnl` / `trip` | 当日亏损上限；一旦触发拒绝一切新单；agent 只能触发不能解除 |
| 调参边界 | `RiskLimits.tunable_bounds` | agent 只能在区间内改 `price_band_bps / max_participation / max_order_notional` |

## 6. 模拟器（`quantstack.sim`）

* **日频面板**：植入短期反转与量价背离两类信号（与因子库合成数据同构），供 alpha 层与因子挖掘使用。
* **日内行情**：公允价几何布朗运动（按日波动率缩放）；每个场所独立维护报价缓存，只在「更新事件」（泊松强度 `quote_update_rate`）、
  公允价移动超过半个 tick、或本方可见量被吃空时刷新 —— 两次更新之间报价是**陈旧**的，这正是延迟套利与 IOC 拒单的来源。
* **延迟**：每个场所单程延迟为对数正态 + 尾部突刺（IEX 额外 350µs speed bump 体现在中位数里）；订单在 `created_us + send_delay_us + 延迟` 到达，按**到达时刻**的报价撮合；回报再经一次延迟；`Fill.latency_us` = 传输 + 回报延迟（不含排队时间）。
* **撮合**：IOC 价格不利即拒单、否则吃可见量（含隐藏流动性系数）并撤余量；MARKET 逐档扫到保护价；POST_ONLY 穿越即拒；
  挂单进入队列（队列前方量 = 到达时本方可见量），由对手价穿越或触价成交流（泊松）消耗。
* **冲击**：只有主动成交施加冲击；临时冲击线性（`η·σ·q/ADV`，半衰期衰减），永久冲击线性（`γ·σ·q/ADV`）。
  被动成交不计冲击（对手方是价格推动者）；不用逐笔平方根律，以免「切得越细冲击越大」的偏差。

## 7. agent 层设计要点

* **决策点而非自由漫游**：agent 只在第 3 节的决策点被调用，输入是结构化上下文（JSON），输出是 pydantic 决策对象（`output_format`）。
* **三级保障**：模型输出 → 结构化校验（策略名必须可用、参数必须数值、风控参数必须在区间）→ 失败/低置信沿阶梯升级重试 → 仍失败回退 `ScriptedBackend` 基线。
* **工具**：10 个只读工具（组合、母单、策略列表、TCA、场所统计、事件、风控、行情、冲击估计）+ 2 个写工具（有边界调参、熔断）。模型无法通过工具下单。
* **日志**：`decisions.jsonl` 记录任务、模型、effort、尝试次数、用量、工具调用、决策与采纳情况；事件总线记录 `agent_decision` / `agent_param_change`。
* **研究流程**：`research_campaign` = 因子假设*（Fable 5.1 max）→ `FactorMiner.mine` → `evaluate` → 研究评审*；只有评审过的因子才能进入部署集合。

## 8. 与三库的接入

`quantstack.core.registry.discover(names)` 把 `/mnt/project-files/{factor_mining, quant/order_placement, routing_latency, physics_math_control}` 加入 `sys.path` 并按需导入（结果缓存，各工厂只导入自己需要的库）；
`make_alpha_model / make_execution_factory / make_router / make_impact_model` 以 `kind="auto"` 优先选兄弟库、否则参考实现；`make_state_estimator / make_controller` 取跨学科库 crossquant 的 Kalman / OU 估计器与 PID / LQR 控制器（`crossquant.registry_entries()` 一次性登记，`plugin_registry()` 列出四个库的全部可用实现，`python -m quantstack.agent.cli plugins` 打印）。

| 库 | 入口 | 适配 |
|---|---|---|
| factormine | `make_alpha_model(**cfg)` / `make_miner(**cfg)` | 直接返回契约对象，无需适配 |
| order_placement | `make_execution_factory(**cfg)` / `make_impact_model()` | 直接实现契约协议 |
| latency_routing | `make_router(venues, **cfg)` | `adapters/latency_routing_adapter.py`：OrderSlice↔ParentOrder、ChildOrder→VenueOrder、VenueStats 同名互转 |

## 9. 运行模式

| 模式 | 时钟 | 行情 | 网关 | 说明 |
|---|---|---|---|---|
| 模拟/回测 | `SimClock` | `SyntheticMarket`（或历史行情回放器） | `VenueSimulator` | 当前实现 |
| 纸面 | `WallClock` | 实时行情适配器 | 纸面撮合（可复用 `VenueSimulator` 的撮合规则） | 待实现 |
| 实盘 | `WallClock` | 实时行情适配器 | 券商/交易所 API 适配器（实现 `VenueGateway`） | 待实现 |

## 10. 已知限制与下一步

1. 单条路径的 TCA 噪声很大（同一策略不同种子 IS 可差 20bps 以上）；agent 的策略/场所评价应基于多种子、多日滚动统计，建议新增 `evaluate.py` 跑 N 个种子取分布。
2. 组合层是参考实现（多空分数加权），跨学科库若提供均值-方差/风险平价/鲁棒优化器可直接替换 `PortfolioConstructor`。
3. 模拟器是 L1（最优报价）级别；L2/L3 队列模型、隐藏单、暗池与做市商库存可在 `sim/venue.py` 扩展。
4. 实盘网关与行情适配器未实现；接口已定义。
5. 真实 Claude 调用在本环境未做联网验证（无 API 凭据），`ClaudeBackend` 按 Anthropic SDK 2.x 文档编写，首次联网运行请先跑 `examples/run_agent_live.py --dry-run`。
