# 接口契约（供集成 agent / 因子库 / 路由库使用）

所有类型定义在 `order_placement/interfaces.py`，字段均为基础类型，`to_dict()` 可直接 JSON 序列化。

## 1. 因子挖掘库 → 本库

实现 `AlphaProvider` 协议即可：

```python
class MyFactorLib:
    def latest(self, symbol: str, ts: float) -> AlphaSignal | None: ...
```

`AlphaSignal` 字段：

| 字段 | 含义 |
|---|---|
| `value` | 在 `horizon_s` 秒内的预期收益，**单位 bps**，正数看涨 |
| `horizon_s` | 预测期限（秒） |
| `confidence` | [0,1]，线性缩放信号 |

本库通过 `drift_per_second(price) = price·value·1e-4·confidence / horizon_s` 把它换算成价格漂移 μ：
- 做市：AS 的保留价加 μτ；GLFT 的双边偏移加 ∓μ/(γσ²)·c2；
- 执行：Almgren-Chriss 目标函数加漂移项；`ScheduledExecutor` 在 alpha 不利时提前一个区间。

## 2. 本库 → 路由延迟优化库

实现 `Router` 协议：

```python
class MyRouter:
    def route(self, orders: Sequence[ChildOrder]) -> list[str]: ...          # 发送，返回 order_id
    def latency_profile(self, venue: str | None = None) -> LatencyProfile: ...
```

`ChildOrder` 关键字段：`side`、`qty`、`price`（None=市价）、`order_type`（LIMIT/MARKET/IOC/POST_ONLY/CANCEL）、
`urgency`（0 被动 ~ 1 立即，路由层可据此选更快通道或吃单交易所）、`venue_hint`、`parent_id`、`order_id`。
CANCEL 单的 `order_id` 即被撤订单的 id。

## 3. 路由库 → 本库

- `LatencyProfile(venue, one_way_ms, jitter_ms, cancel_ack_ms)`：做市报价按 `latency_z·σ·√RTT` 加宽，RTT = 2·(单向 + 2·抖动)。
- `Fill`：成交回报，经 `OrderPlacementEngine.on_fill` 按 `order_id` 分发给对应策略。
- 多交易所拆分：`queue.cont_kukanov_allocation` 需要每个交易所的排队量、返佣与预期流出量（`VenueSpec`），可由路由库提供。

## 4. 集成门面

```python
engine = OrderPlacementEngine(router=MyRouter(), alpha_provider=MyFactorLib())
engine.add_strategy("mm-btc", "BTCUSDT", GLFT(MarketMakerConfig(...)))
engine.add_strategy("twap-eth", "ETHUSDT", ScheduledExecutor(parent, twap(qty, 20)))
engine.on_book(book)   # 每个行情快照调用，自动拉 alpha、取延迟、生成并发送子单
engine.on_fill(fill)   # 每笔成交回报调用
```

## 5. 新增策略

任何实现 `name`、`on_book(book, state, alpha, latency) -> list[ChildOrder]`、`on_fill(fill, state)` 的类都满足 `OrderPlacementStrategy` 协议，可直接注册进引擎。做市模型推荐继承 `BaseMarketMaker`，只实现 `_offsets(q, tau, sigma, mu) -> (δ_bid, δ_ask)`。

## 6. quantstack 集成层适配（`/mnt/project-files/trading_system`）

`order_placement/adapters/quantstack_exec.py` 直接实现 `quantstack.contracts.ExecutionStrategy` / `ExecutionStrategyFactory` / `ImpactModel`。

```python
import sys
sys.path += ["/mnt/project-files/trading_system", "/mnt/project-files/quant/order_placement"]
from order_placement import make_execution_factory, make_impact_model
factory = make_execution_factory(trading_seconds_per_day=23400)   # 加密市场 86400
strat = factory.create("almgren_chriss", exec_order, bucket_s=60)
impact = make_impact_model()   # SqrtLawImpact.expected_cost_bps(quantity, snapshot, duration_us)
```

`available()` 返回的策略名：`twap`、`vwap`、`pov`、`almgren_chriss`、`obizhaeva_wang`、`adaptive_passive`、`market_making`、`is_rl`。

约定：
- `alpha_bps > 0` 视为价格预计朝 `side` 方向移动（买单看涨），对执行意味着“等待更贵”，策略会加速。
- `cancels()` 返回的 `CancelRequest.venue_order_id` 填 **slice_id**，`reason="slice:<slice_id>"`，路由层需展开为该切片下的全部场所订单。
- `on_event` 用 `venue_order_id == slice_id` 或以 `slice_id + "-"` 开头来定位切片，请路由层生成场所订单号时保留该前缀。
- 被动切片为 `POST_ONLY` 且带 `ttl_us = refresh_s`；追单与扫尾为 `IOC`，价格均受母单 `limit_price` 保护。
- 可调参数（`ExecutionOrder.params` 或 `create(**params)`）：`bucket_s`、`refresh_s`、`catch_up_slices`、`impact_frac`、`alpha_urgency_bps`、`risk_aversion`（AC）、`resilience_per_s`（OW）、`volume_profile`（VWAP）、`gamma/A/k`（做市）、`rl_episodes`（RL）。
- `state()` 字段：`filled`、`remaining`、`progress`、`time_elapsed`、`schedule_target`、`inflight`、`avg_price`、`shortfall_bps`。
