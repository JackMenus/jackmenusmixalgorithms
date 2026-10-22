# 全栈交易系统 · 跨库接口契约（v1.0）

> 本文件是因子挖掘库 / 挂单策略库 / 路由延迟优化库 / 跨学科算法库 与集成层（`quantstack`）之间的**唯一契约**。
> 代码定义在 `quantstack/contracts/`，本文是对它的说明；二者冲突时以代码为准。
> 契约对象全部是 `dataclass`，自带 `to_dict()/from_dict()/to_json()/from_json()`，可跨进程/跨语言传递。

## 0. 全局约定

| 项目 | 约定 |
|---|---|
| 时间戳 | `ts_us`：自 epoch 起的**微秒**整数（与路由库一致）。日频面板用 pandas `DatetimeIndex`。 |
| 方向 | `side = +1` 买，`-1` 卖。 |
| 价格 / 数量 | 价格 `float`；数量**整数**股/张。 |
| 费用 | 每股费用，正数为收费，负数为返佣。 |
| 收益单位 | 基点 bps（1e-4）。`alpha_bps` 以 `side` 方向为正表示有利。 |
| 日频面板 | `Panel = dict[str, pd.DataFrame]`，index=时间、columns=资产、字段小写 `open/high/low/close/volume/vwap/returns` —— 与 `factormine.data.Panel` 完全相同。 |
| 依赖方向 | 三个算法库 **不互相 import**，也不 import `quantstack.core`；只依赖 `quantstack.contracts`（或保持自己的等价 dataclass，由 `quantstack/adapters` 做转换）。 |
| 随机性 | 所有带随机性的算法接受 `seed` 参数，便于回放。 |

## 1. 数据流总览

```
Panel ──AlphaModel.predict()──▶ SignalBatch{[AlphaSignal]}
      ──PortfolioConstructor.construct()──▶ [TargetPosition]
      ──OMS 差分 + PreTradeRiskGate──▶ [ExecutionOrder]                 (母单)
      ──ExecutionStrategy.start()/on_market()/on_fill()──▶ [OrderSlice]  (子单切片)
      ──Router.route()──▶ RoutingDecision{[VenueOrder]}                (场所订单)
      ──VenueGateway.submit()──▶ 交易所/模拟器
      ◀──VenueGateway.poll()── [OrderEvent{Fill}]
            └─▶ Router.on_event()  └─▶ ExecutionStrategy.on_fill()  └─▶ OMS/TCA/Agent
```

每一层只认识紧邻上下游的契约对象。LLM agent 在 **研究 / 执行规划 / 复盘 / 事故** 四个决策点介入，所有决策经确定性风控闸门校验。

## 2. 因子挖掘库 → 需要实现

### 2.1 `FactorMiner`
```python
class FactorMiner(Protocol):
    def mine(self, panel: Panel, budget: int = 100, **kwargs) -> list[FactorSpec]: ...
    def evaluate(self, spec: FactorSpec, panel: Panel, **kwargs) -> FactorReport: ...
```
* `FactorSpec(name, expression, family, params, source_library, horizon, created_at, notes)`
  `family ∈ {gp, rl, llm, dl, handcrafted}`；`expression` 用你们的表达式语法即可，集成层不解析，只透传与存档。
* `FactorReport(spec, ic_mean, ic_ir, rank_ic_mean, turnover, long_short_sharpe, decay_half_life, coverage, period_start, period_end, sample="is"/"oos", extra)`

### 2.2 `AlphaModel`（**流水线真正消费的接口**）
```python
class AlphaModel(Protocol):
    model_id: str
    def fit(self, panel: Panel, **kwargs) -> None: ...
    def predict(self, panel: Panel, as_of=None, **kwargs) -> SignalBatch: ...
    def describe(self) -> dict: ...
```
* `predict` **只能用 `as_of` 及之前的数据**，返回 `as_of` 时刻（默认面板最后一行）的 `SignalBatch`。
* `AlphaSignal(symbol, ts_us, horizon_us, expected_return_bps, confidence∈[0,1], score, source, factor_exposures, metadata)`
  * `score`：横截面标准化分数（z-score 或 rank），组合层按它排序/加权；
  * `expected_return_bps`：预测期期望超额收益（组合层据此和交易成本做权衡）；
  * `horizon_us`：预测有效期，日频可填 `23_400_000_000`（6.5 小时）或 `86_400_000_000`。
* 建议在包根暴露工厂：`factormine.make_alpha_model(**cfg) -> AlphaModel`、`factormine.make_miner(**cfg) -> FactorMiner`。若不方便，告诉集成线程类名，由 `quantstack/adapters/factormine_adapter.py` 包一层。

## 3. 挂单策略库 → 需要实现

### 3.1 `ExecutionStrategy`（一个实例服务一张母单，事件驱动）
```python
class ExecutionStrategy(Protocol):
    name: str
    def start(self, order: ExecutionOrder, snapshot: MarketSnapshot) -> list[OrderSlice]: ...
    def on_market(self, snapshot: MarketSnapshot) -> list[OrderSlice]: ...
    def on_fill(self, fill: Fill, snapshot: MarketSnapshot) -> list[OrderSlice]: ...
    def on_event(self, event: OrderEvent, snapshot: MarketSnapshot) -> list[OrderSlice]: ...
    def cancels(self, snapshot: MarketSnapshot) -> list[CancelRequest]: ...
    def is_complete(self) -> bool: ...
    def state(self) -> dict[str, float]: ...   # 进度、剩余量、当前参与率等，供 agent 观察
```
* 输入母单 `ExecutionOrder(order_id, symbol, side, quantity, start_us, end_us, benchmark, urgency, limit_price, max_participation, alpha_bps, alpha_horizon_us, strategy_hint, params, ...)`
* 输入行情 `MarketSnapshot(symbol, ts_us, quotes=[VenueQuote...], last_price, adv, volatility, session_volume)`，带 `nbbo()/mid/spread_bps/displayed_size(side)` 辅助方法。
* 输出子单 `OrderSlice(slice_id, parent_id, symbol, side, quantity, order_type, limit_price, urgency, alpha_bps, alpha_horizon_us, ttl_us, created_us, display_quantity, tag)`
  * `order_type ∈ {LIMIT, MARKET, IOC, POST_ONLY, PEG}`；`urgency` 0=完全被动 1=立即成交，路由层据此在吃单/挂单、费用/成交概率间权衡。
  * **执行层不选场所、不决定发送时刻**，那是路由层的事。
* 流水线调用频率：每个模拟/实盘时间步调用一次 `on_market`（默认 1 秒），成交到达时调用 `on_fill`。

### 3.2 `ExecutionStrategyFactory`
```python
class ExecutionStrategyFactory(Protocol):
    def create(self, name: str, order: ExecutionOrder, **params) -> ExecutionStrategy: ...
    def available(self) -> dict[str, str]:   # 名称 -> 一句话说明 + 关键参数，agent 用它做选择
```
建议策略名：`twap / vwap / pov / almgren_chriss / is_rl / market_making / adaptive_passive` 等。
建议在包根暴露 `make_execution_factory(**cfg) -> ExecutionStrategyFactory`。
可选：`ImpactModel.expected_cost_bps(quantity, snapshot, duration_us)`，agent 用于成本-alpha 权衡。

## 4. 路由延迟优化库 → 需要实现

> 你们现有的 `latency_routing.interfaces` 已基本对齐：`ParentOrder ≈ OrderSlice`（`order_id ↔ slice_id`）、`ChildOrder ≈ VenueOrder`、`VenueQuote` 相同（集成层多一个 `symbol` 字段）。集成层的 `adapters/latency_routing_adapter.py` 负责互转，**你们不必改现有类型**；只需保证下面的方法存在。

```python
class Router(Protocol):
    def route(self, order_slice: OrderSlice, snapshot: MarketSnapshot) -> RoutingDecision: ...
    def on_event(self, event: OrderEvent) -> None: ...        # 用成交/拒单/延迟更新场所统计与延迟模型
    def venue_stats(self) -> list[VenueStats]: ...

class LatencyModel(Protocol):                                # 与 LatencyModelProtocol 相同
    def observe(self, venue: str, latency_us: float) -> None: ...
    def quantile(self, venue: str, q: float) -> float: ...
    def sample(self, venue: str, n: int) -> Sequence[float]: ...
```
* `RoutingDecision(slice_id, venue_orders=[VenueOrder...], expected_cost_bps, expected_fill_ratio, diagnostics, rationale)`
* `VenueOrder(venue_order_id, slice_id, parent_id, venue, symbol, side, quantity, price, order_type, send_delay_us, expected_arrival_us, created_us, route_reason)`
* `OrderEvent(kind ∈ ack/reject/cancel/expire/fill/partial, venue_order_id, ts_us, venue, reason, fill: Fill|None)`，`Fill.latency_us` 给出该笔的往返延迟，请用它喂 `LatencyModel.observe`。
* `VenueStats(venue, fills, rejects, fill_ratio, avg_slippage_bps, latency_p50_us, latency_p99_us, maker_ratio)`
建议在包根暴露 `make_router(venues: list[str], **cfg) -> Router`。

## 5. 跨学科算法库（物理 / 数学 / 控制）→ 可选钩子

集成层预留三个协议，便于把滤波 / 控制 / 成本类算法直接插到执行与路由层：
```python
class StateEstimator(Protocol):    # Kalman / 粒子滤波：价格、价差、延迟的隐状态估计
    def update(self, observation: float, ts_us: int) -> float: ...
    def predict(self, horizon_us: int) -> float: ...
class Controller(Protocol):        # PID / LQR / MPC：执行进度跟踪 -> 参与率调节
    def step(self, error: float, dt_us: int) -> float: ...
class ImpactModel(Protocol):       # 冲击/成本模型（平方根律、Almgren-Chriss、传播子模型等）
    def expected_cost_bps(self, quantity: int, snapshot: MarketSnapshot, duration_us: int) -> float: ...
```
其他算法（如最优控制求解器、随机过程模拟、组合优化器）可以提供为普通函数/类，由集成层或兄弟库直接调用。

**已接入实现**（`/mnt/project-files/physics_math_control`，包 `crossquant`，仅依赖 numpy + scipy，2026-10-22 登记）：

| 协议 | 登记名（同时是导入路径） | 说明 |
|---|---|---|
| StateEstimator | `crossquant.KalmanTrendEstimator(level_var, slope_var, obs_var)` | 局部线性趋势 Kalman：`update` 返回滤波后的水平，`predict(h)` = level + slope·h，用于公允价 / 延迟外推 |
| StateEstimator | `crossquant.OUSpreadEstimator(forgetting, dt_sec)` | 在线 RLS 拟合 AR(1)/OU，`predict` 给出均值回复的条件期望，用于价差 |
| Controller | `crossquant.PIDParticipationController(kp, ki, kd, out_min, out_max)` | 抗积分饱和 PID：error = 目标完成比例 − 实际完成比例 → 参与率调整量 |
| Controller | `crossquant.LQRTrackingController(q, r, dt_ref_sec, out_max)` | 标量 LQ 最优反馈 u = −K·error（Riccati 解） |
| ImpactModel | `crossquant.SquareRootImpactModel(Y=0.5, horizon_volume=False)` | 半价差 + Y·σ·sqrt(Q/ADV) |
| ImpactModel | `crossquant.AlmgrenChrissImpactModel(lam, n_slices, perm_frac)` | Almgren–Chriss 期望成本（临时 + 永久冲击 + 价差） |

冲击模型只读 `snapshot` 的 `mid / spread_bps / adv / volatility` 属性，与 `MarketSnapshot` 一致。集成层入口：`registry.make_state_estimator(kind)`、`registry.make_controller(kind)`、`registry.make_impact_model(kind)`，`kind` 取上表登记名；`"auto"` 分别取 Kalman / PID / （挂单库不可用时）平方根律。`crossquant.registry_entries()` 返回 `{协议名: {登记名: 类}}`，`registry.crossquant_entries()` 原样转发，`registry.plugin_registry()` 列出四个库全部可用实现。

## 6. 集成层自带（无需各库实现）
`PortfolioConstructor`、`VenueGateway`（模拟 / 纸面 / 实盘）、`MarketDataFeed`、OMS、`PreTradeRiskGate`、TCA、事件总线、LLM agent。

## 7. 接入检查清单
1. `pip install -e` 后 `import quantstack.contracts` 可用（或保留自己的等价 dataclass + 告知集成线程）。
2. 在你们的测试里加 `assert isinstance(obj, quantstack.contracts.<Protocol>)`。
3. 在包根暴露 `make_*` 工厂（见各节），并在 README 写明名称与参数。
4. 把包的**导入名**与**共享目录路径**告知集成线程，`quantstack.core.registry` 会自动发现。
5. 所有随机算法接受 `seed`；所有耗时方法支持 `budget`/`max_iter` 这类预算参数，便于 agent 控制成本。

## 8. 联调后的补充约定（2026-10-22，三库均已确认；跨学科库 crossquant 同日登记）

| 约定 | 内容 |
|---|---|
| `alpha_bps` 语义 | `> 0` 表示价格预计朝 `side` 方向运动（买单看涨）。等待更贵 → 执行层加速；路由层据此权衡吃单/挂单。 |
| 场所订单号 | 流水线在路由后把每个 `VenueOrder.venue_order_id` 强制改写为 `"{slice_id}-{k}"`（k 从 1 起）。执行层可用 `venue_order_id == slice_id` 或 `startswith(slice_id + "-")` 定位切片。 |
| 整切片撤单 | 执行层 `cancels()` 可返回 `CancelRequest(venue_order_id=slice_id, reason="slice:<slice_id>")`，流水线展开为该切片下全部场所订单。 |
| 未路由余量 | 路由层只路由了切片的一部分（如 IOC 不超过可见量）时，流水线立即向执行层发 `OrderEvent(kind="cancel", venue_order_id=slice_id, reason="unrouted", remaining_quantity=余量)`，执行层只释放该余量。 |
| 在途量上限 | 流水线保证「其他切片在途 + 新切片 ≤ 母单剩余」；切片被缩量时发 `OrderEvent(kind="cancel", venue_order_id=slice_id, reason="oms:in_flight_cap_resize", remaining_quantity=缩减量)`。路由层超额挂单总量也被裁到母单剩余以内。 |
| 部分撤销 | 执行层处理 `cancel/reject/expire` 时，若 `0 < remaining_quantity < 在途量`，只扣减 `remaining_quantity`，不要整切片移除。 |
| 切片完成 | 切片成交量达到目标后，流水线撤掉该切片其余场所订单（配合路由层 Cont-Kukanov 的有意超额挂单）。 |
| `send_delay_us` 基准 | 相对 `route()` 返回、流水线开始派发的时刻 t0；同一决策的子单共享 t0；`expected_arrival_us` 同基准。 |
| `Fill` 必填 | 网关必须填写 `Fill.slice_id / parent_id / latency_us`；`latency_us` 为传输 + 回报延迟（不含排队时间），路由层以 `latency_us/2` 作单程样本。 |
| 路由层事件 | 路由层 `on_event` 直接接受 quantstack `OrderEvent`；`fill/partial` 更新延迟模型与场所统计，`reject` 计拒单，`ack/cancel/expire` 可忽略。 |
| 被动成交与冲击 | 模拟器只对主动（taker）成交施加冲击；被动成交不计冲击。比较策略成本请用多个种子取平均（单路径噪声可达 20bps）。 |
| 入口函数 | `factormine.make_alpha_model(library=None, method="max_ir", horizon=1, bar_us=23_400_000_000, mine_budget=100, seed=0, llm=None)`、`factormine.make_miner(...)`；`order_placement.make_execution_factory(**cfg)`（cfg: trading_seconds_per_day / bucket_s / refresh_s）、`order_placement.make_impact_model()`；`latency_routing.make_router(venues, latency_prior_us=None, default_latency_us=500, halflife=200, **cfg)`；`crossquant.registry_entries()`（见 §5 表）。 |
| 共享路径 | `/mnt/project-files/factor_mining`、`/mnt/project-files/quant/order_placement`、`/mnt/project-files/routing_latency`、`/mnt/project-files/physics_math_control`、`/mnt/project-files/trading_system`（本库）。 |

## 9. 自主生成算法库 autoalgo（2026-10-22 登记）

`/mnt/project-files/auto_algorithms`（包 `autoalgo`）在四个算法库之上按本文档的协议**拼装新算法**，并为每个算法自动产出评估报告与中文可解释性文档，放入 `auto_algorithms/generated/`。它只消费本文档的契约，不新增契约；新增的只有一个"执行策略选择器"的注入点（本来就是 `TradingPipeline.strategy_selector`）。

| 协议 | 生成算法（`generated/` 中的 algo_id） | 模板族 kind | 建立在 |
|---|---|---|---|
| AlphaModel | `regime_gated_alpha`（Hurst 状态门控多因子）、`signature_alpha`（路径签名因子）、`cross_operator_mined_alpha`（跨学科算子 GP 挖掘） | `autoalgo.family.alpha.*` | factormine + crossquant |
| PortfolioConstructor | `denoised_riskparity_portfolio`（RMT 去噪协方差 → 均值-方差 / HRP + 信号倾斜） | `autoalgo.family.portfolio.denoised_riskparity` | crossquant + 参考实现规则 |
| ExecutionStrategy（经组合式工厂） | `kalman_pid_execution`，策略名 `kalman_pid`（AC 轨迹 × Kalman 公允价 × PID/LQR 反馈） | `autoalgo.family.execution.kalman_pid` | order_placement 基类 + crossquant |
| StrategySelector | `bandit_strategy_selector`（LinUCB 上下文老虎机，奖励 = −TCA 实现偏差） | `autoalgo.family.selector.bandit` | crossquant + 参考规则冷启动 |
| Router | `bayesian_venue_router`（Thompson 采样 + BOCPD 退化监测叠加层） | `autoalgo.family.router.bayesian_venue` | latency_routing 适配器 + crossquant |

**kind 命名**：`autoalgo.<algo_id>` 用 `generated/` 中搜索出的参数；`autoalgo.family.<family>` 用模板族默认参数（不依赖生成结果）；`autoalgo.auto`（或 `autoalgo`）取该协议下第一个评估通过的生成算法。`registry.make_alpha_model / make_portfolio / make_execution_factory / make_router / make_strategy_selector` 都接受这些 kind，`build_sim_pipeline(kinds={"alpha": ..., "execution": ..., "router": ..., "portfolio": ..., "selector": ...})` 同样接受（`portfolio` / `selector` 为本次新增键，默认 `reference`）。

**补充约定**：

| 约定 | 内容 |
|---|---|
| 组合式执行工厂 | `autoalgo` 的执行工厂包裹挂单库工厂：`available()` = 挂单库 8 种策略 + `kalman_pid`；未知名字透传给挂单库。 |
| 选择器注入 | 选择器是可调用对象 `(order, snapshot, available) -> (name, params)`；若有 `attach(pipeline)`，`build_sim_pipeline` 会调用它（订阅 `order_done` 读取 TCA 回填奖励）。 |
| 组合构造器需要历史 | 自研组合构造器估协方差需要 `fit(panel, as_of)`；`build_sim_pipeline` 在 `kinds["portfolio"]` 非 `reference` 时自动调用一次；长期运行应在每次 `run_cycle` 前 `pipe.portfolio.fit(panel, as_of)`。 |
| 路由叠加层 | `make_router("autoalgo.*", venues, base=None)`：`base` 缺省为 `make_router("auto", venues)`（latency_routing 适配器）；叠加层只改吃单份额 / 退化场所，`on_event` 先透传给底层。 |
| 可解释性文档 | 每个生成算法附 `generated/<algo_id>/<algo_id>.md`（概述 / 协议层 / 假设 / 组件来源 / 流程与公式 / 参数 / 评估 vs 基线 / 风险 / 接入方式 / 复现与血缘），`evaluation.json` 给出通过与否；`passed=false` 的算法默认不应上线。 |
| 评估基线 | 因子层对比 `reference.ReversalVolumeAlpha`；组合层在同一信号（默认受控 IC 合成信号，多面板平均）下对比 `reference.LongShortConstructor`；执行层对比挂单库 `almgren_chriss` / `twap`；选择器对比 `reference.choose_strategy`；路由层对比底层 latency_routing。模拟器评估均为多种子成对比较。 |
| 入口 | `python -m autoalgo generate --budget quick|standard|full --proposer random|evolution|claude|mock`；`autoalgo.registry.make(kind)`；`autoalgo.registry_entries()` 被 `registry.plugin_registry()` 合并展示。 |
| 模型策略 | 参数 / 设计提议阶段用 `claude-fable-5-1` + `output_config.effort="max"`（`--proposer claude`，需 `anthropic` 与 API 凭据；不可用时退回随机并在血缘中记录）；评估与文档生成为确定性代码。 |
