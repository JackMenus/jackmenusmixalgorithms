# 模型路由策略（Fable 5.1 最高推理 / Opus 5.5 自适应）

实现：`quantstack/agent/model_policy.py`；打印当前路由表：`python -m quantstack.agent.cli policy`。

## 1. 任务分级

| 任务类（TaskClass） | 级别 | 起始档位 | 上限 | 触发位置 |
|---|---|---|---|---|
| architecture / strategy_design | 强推理 | Fable 5.1 @ max | 同左 | 人工发起的设计任务 |
| factor_hypothesis / research_review | 强推理 | Fable 5.1 @ max | 同左 | `research_campaign`、`run_daily_cycle(factor_reports=...)` |
| post_mortem / risk_incident / regime_assessment | 强推理 | Fable 5.1 @ max | 同左 | 复盘 escalate、kill switch |
| execution_plan | 自适应 | Opus 5.5 @ low | Fable 5.1 @ xhigh | 每日周期步骤 4 |
| post_trade_review / routing_diagnostics / param_tuning | 自适应 | Opus 5.5 @ low | Fable 5.1 @ xhigh | 每日周期步骤 6 |
| data_quality / daily_report / log_summary / code_routine | 自适应 | Opus 5.5 @ low | Fable 5.1 @ xhigh | 步骤 0 / 8 |

强推理任务**不设重试阶梯**（已是最高档），只允许一次尝试 + 回退基线；自适应任务最多 3 次尝试。

## 2. 自适应档位

阶梯：`Opus 5.5@low → Opus 5.5@medium → Opus 5.5@high → Fable 5.1@high → Fable 5.1@xhigh`

起始档位由复杂度分数决定（`ComplexitySignals.score()`）：

```
score = 0.10·size + 0.15·entities + 0.30·stakes + 0.20·novelty + 0.25·anomaly
size     = 1 − exp(−上下文字符数 / 20000)
entities = 1 − exp(−涉及实体数 / 15)        # 母单/因子/场所数量
stakes   ∈ [0,1]   本次名义金额 / NAV 等资金权重
novelty  ∈ [0,1]   新标的 / 新市场状态
anomaly  ∈ [0,1]   TCA 异常、拒单率、事故
阈值：< 0.25 → 档 0；< 0.50 → 档 1；< 0.65 → 档 2；否则 档 3
```

升级规则：结构化输出不合法、后端出错、拒答、或决策 `confidence < 0.4` → `prior_failures += 1`，下一次尝试档位 +1。
全部失败 → 回退 `ScriptedBackend`（确定性基线），决策日志标记 `-> scripted fallback`。

## 3. API 参数映射（Anthropic SDK 2.x）

| 项目 | Fable 5.1 | Opus 5.5 |
|---|---|---|
| 模型 ID | `claude-fable-5-1` | `claude-opus-5-5` |
| 思考 | 始终开启，省略 `thinking`（等价 adaptive）；`budget_tokens`/`disabled` 均 400 | 同左；默认 effort 为 medium，必须显式设置 |
| 深度 | `output_config={"effort": "max"}`（强推理）/ `high`/`xhigh`（阶梯） | `output_config={"effort": "low"/"medium"/"high"}` |
| 结构化输出 | `output_format=<pydantic 模型>`（tool_runner / stream / parse 均支持） | 同左 |
| 工具 | `client.beta.messages.tool_runner`，`tool_choice` 只能 auto（强制调用返回 400），用提示词引导 | 同左 |
| 拒答回退 | `betas=["server-side-fallback-2026-07-01"], fallbacks="default"`（默认开启；可在 `ClaudeBackend(enable_fallbacks=False)` 关闭） | 同左 |
| 长输出 | `stream=True`（`get_final_message()`），`max_tokens` 64000，超时 30 分钟 | 非流式 `max_tokens` 16000 |
| 缓存 | 系统提示块加 `cache_control: ephemeral`（角色提示稳定、上下文放在 user 消息里） | 同左 |

`ModelSpec.request_kwargs()` 产出上述参数；`ClaudeBackend` 负责 betas/fallbacks/timeout。

## 4. 成本提示（官方 API 价格）

| 模型 | 输入 $/MTok | 输出 $/MTok |
|---|---|---|
| Claude Fable 5.1 | 10 | 50 |
| Claude Opus 5.5 | 4 | 20 |

强推理任务一次调用可能运行数分钟并消耗数万输出 token；因此只在表 1 的决策点触发，并把上下文压缩为 `CycleResult.brief()` 这样的摘要而非全部订单明细。
`decisions.jsonl` 中的 `usage` 字段可用于按任务统计成本。
