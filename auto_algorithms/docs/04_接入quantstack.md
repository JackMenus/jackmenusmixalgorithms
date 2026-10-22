# 04 接入 quantstack

## 发现

`quantstack.core.registry.SIBLING_LIBS` 新增 `"autoalgo": <PROJECT_ROOT>/auto_algorithms`。`discover()` 导入 `autoalgo`（轻量，不触发 scipy）。
`plugin_registry()` 合并 `autoalgo.registry_entries()`：生成算法以 `autoalgo.<algo_id>` 登记，模板族以 `autoalgo.family.<family>` 登记；
`ExecutionStrategy` 协议的条目归入 `ExecutionStrategyFactory`（组合式工厂）。

## kind

| kind | 含义 |
|---|---|
| `autoalgo.<algo_id>` | `generated/index.json` 中的算法，使用搜索出的参数 |
| `autoalgo.family.<family>` | 模板族默认参数，不依赖 `generated/` |
| `autoalgo.auto` / `autoalgo` | 该协议下第一个 `passed=true` 的生成算法；没有则第一个生成算法；再没有则模板族默认 |

## 工厂

```python
from quantstack.core.registry import make_alpha_model, make_portfolio, make_execution_factory, make_router, make_strategy_selector
make_alpha_model("autoalgo.regime_gated_alpha")
make_portfolio("autoalgo.denoised_riskparity_portfolio", top_k=8)        # 覆盖参数
make_execution_factory("autoalgo.kalman_pid_execution")                   # 挂单库 8 策略 + kalman_pid
make_router("autoalgo.bayesian_venue_router", venues, base=None)          # base 缺省 = latency_routing 适配器
make_strategy_selector("autoalgo.bandit_strategy_selector")               # 可调用对象，带 attach(pipeline)
```

`build_sim_pipeline(panel, kinds={"alpha", "execution", "router", "portfolio", "selector"})`：后两个键为新增，默认 `reference`。
选择器若有 `attach` 会被自动挂到流水线（订阅 `order_done`）；非参考组合构造器会被自动 `fit(panel, as_of)` 一次。

## 流水线内的生命周期

- 因子层：`pipe.fit_alpha(panel, as_of)` → `predict(as_of)`；autoalgo 的三个模型都支持 `as_of` 截断。
- 组合层：每次 `run_cycle` 前调用 `pipe.portfolio.fit(panel, as_of)` 以刷新协方差窗口。
- 执行层：母单 `strategy_hint="kalman_pid"`，或由选择器选择；参数可放在 `plans[order_id]["params"]`。
- 路由层：叠加层的 `venue_stats()` 直接返回底层统计；`describe()` 给出各场所后验与退化概率，agent 可读。

## agent 侧

`quantstack.agent` 的工具层读取 `plugin_registry()` 展示可用实现；生成算法的 md 文档路径在 `index.json` 的 `markdown` 字段，
agent 可以把它作为"算法说明书"读入上下文后再决定是否启用。
