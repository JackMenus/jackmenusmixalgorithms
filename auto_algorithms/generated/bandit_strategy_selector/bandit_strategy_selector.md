# 上下文老虎机执行策略选择器（`bandit_strategy_selector`）

> 自主生成算法库（autoalgo）生成 · 协议 `StrategySelector`（执行规划） · 模板族 `selector.bandit` · 生成时间 2026-10-22T11:42:06+00:00 · 生成器 v0.1.0

## 1. 概述

没有一种执行策略在所有母单上最优：大单/紧急单适合前置轨迹，小单/低紧急度适合被动赚价差。参考规则是手写阈值，LinUCB 用母单特征作上下文、以实现偏差为奖励在线学习策略选择，长期应不劣于规则并随数据改进。

结论：**评估通过**。 后半程实现偏差 -1.60 bps vs 规则基线 0.54 bps；全程 3.26 vs 8.63；各臂被选次数 {'twap': 18, 'almgren_chriss': 8, 'pov': 10, 'adaptive_passive': 8, 'kalman_pid': 16}。

## 2. 所处协议层与接口

- 协议：`quantstack.core.pipeline.StrategySelector`（可调用对象）
- 接口：`(order, snapshot, available) -> (strategy_name, params)`，注入 `TradingPipeline.strategy_selector`
- 输入 / 输出：为每张母单选择执行策略与参数
- quantstack 注册名：`autoalgo.bandit_strategy_selector`

## 3. 设计动机与假设

- 同一流水线内母单特征与成本的关系平稳；成本以 TCA 实现偏差衡量。
- 候选策略名存在于执行工厂 available() 中。

## 4. 组件来源（建立在哪些库之上）

| 库 | 组件 | 在本算法中的作用 |
|---|---|---|
| `crossquant` | `crossquant.control.bandits.LinUCB` | 上下文线性老虎机 |
| `order_placement` | `order_placement.make_execution_factory` | 8 种基础执行策略工厂（被 AutoExecutionFactory 包裹） |
| `quantstack` | `quantstack.reference.execution.choose_strategy` | 确定性执行策略选择规则（冷启动与基线） |
| `quantstack` | `quantstack.core.tca.TCAEngine` | 实现偏差、成交率、maker 比例 |

涉及库：`crossquant`, `order_placement`, `quantstack`。

## 5. 算法流程与关键公式

1. 上下文 x = [1, log10(%ADV), urgency, tanh(alpha/10), 价差, 时长, 波动率, %ADV]。
2. 前 warmup 次与 ε 探索沿用 quantstack.reference.execution.choose_strategy，并用其结果训练对应臂（冷启动不劣于规则）。
3. 其余按 LinUCB 选臂：θ_a = A_a⁻¹ b_a，score = θ_aᵀx + α√(xᵀA_a⁻¹x)。
4. attach(pipeline) 订阅 order_done：reward = −clip(IS_bps, ±3s)/s（未成交按最差计），更新 A_a += xxᵀ, b_a += r·x。

关键公式：

- `a_t = argmax_a θ_aᵀx_t + α√(x_tᵀA_a⁻¹x_t)`
- `r = −clip(IS_bps, −3s, 3s)/s`

## 6. 参数

| 参数 | 取值 | 含义 | 搜索范围 |
|---|---|---|---|
| `arms` | [twap, almgren_chriss, pov, adaptive_passive, kalman_pid] | 候选策略 | — |
| `alpha` | 0.9195 | LinUCB 探索系数 | [0.2, 1.5] |
| `ridge` | 0.3 | 岭惩罚 | [0.3, 5]（对数均匀） |
| `reward_scale_bps` | 20 | 奖励归一化尺度 | {10, 20, 40} |
| `warmup` | 2 | 冷启动次数 | {1, 2, 4} |
| `epsilon` | 0.05 | ε-贪心探索概率 | [0, 0.15] |

## 7. 评估结果

- 主目标：`learned_rounds_shortfall_bps`（越小越好）；评估设置：{"rounds": 6, "orders_per_round": 5, "seeds": [1, 2], "duration_s": 900, "step_s": 1.0, "arm_counts": {"twap": 18, "almgren_chriss": 8, "pov": 10, "adaptive_passive": 8, "kalman_pid": 16}, "budget": "standard", "seed": 0}
- 相对第一基线的改善：+2.143（已按方向调整，正数 = 更好）

| 指标 | 说明 | 本算法 | reference.choose_strategy |
|---|---|---|---|
| `shortfall_bps` | 实现偏差均值（bps，越小越好） | 3.261 | 8.631 |
| `shortfall_median_bps` | 实现偏差中位数 | 3.083 | 5.278 |
| `shortfall_std_bps` | 实现偏差标准差 | 21.03 | 27.21 |
| `fill_ratio` | 成交率 | 1 | 1 |
| `maker_ratio` | 被动成交比例 | 0.7791 | 0.9106 |
| `n_orders` | 母单数 | 60 | 60 |
| `learned_rounds_shortfall_bps` | 学习期（后半程）实现偏差 | -1.603 | 0.5395 |
| `n_seeds` | 随机种子数 | 2 | 2 |

- 判定：通过。后半程实现偏差 -1.60 bps vs 规则基线 0.54 bps；全程 3.26 vs 8.63；各臂被选次数 {'twap': 18, 'almgren_chriss': 8, 'pov': 10, 'adaptive_passive': 8, 'kalman_pid': 16}。
- 通过条件：后半程平均实现偏差 ≤ 基线 + 1.0 bps。前 warmup 轮沿用基线规则，因此全程差异主要来自后半程。
- 评估耗时：17.2 秒

## 8. 已知风险与失效条件

- 奖励噪声极大（单路径 IS 噪声可达 20 bps），需要数十笔以上母单才能显著优于规则。
- 臂集合变化（新策略加入）需重新学习；线性模型不能表达特征交互。

## 9. 接入方式

```python
from autoalgo.registry import make_strategy_selector
sel = make_strategy_selector('autoalgo.bandit_strategy_selector')
sel.attach(pipeline)              # 设为 strategy_selector 并订阅 order_done

# 全栈流水线：build_sim_pipeline(panel, kinds={'selector': 'autoalgo.bandit_strategy_selector'})
```

quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind='autoalgo.bandit_strategy_selector'` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；也可用 `autoalgo.registry.make('bandit_strategy_selector')` 直接实例化。

## 10. 复现与血缘

- 生成器：autoalgo.pipeline.generate
- 生成器版本：0.1.0
- 提议器：evolution
- 随机种子：0
- 评估预算：standard
- 训练集比例：0.7
- 评估候选数：9
- 评估面板：{"n_days": 600, "n_assets": 60, "start": "2021-01-04", "end": "2023-04-21"}
- 选中候选得分：1.603

候选评估记录（共 9 个，按评估顺序；★ 为最终选中的参数）：

| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | default | （默认） | -1.439 | 是 | 32 |
| 2 | evolution | `ridge`=0.5223, `epsilon`=0.04595 | 1.874 | 否 | 18.1 |
| 3 | evolution | `alpha`=0.6299 | -1.439 | 是 | 19 |
| 4 | evolution | `alpha`=0.6299, `reward_scale_bps`=40, `epsilon`=0.01871 | 4.897 | 否 | 17.3 |
| 5 | evolution | `ridge`=0.9883, `epsilon`=0.08967 | -1.439 | 是 | 16.5 |
| 6 | evolution | `alpha`=0.6299, `warmup`=4 | 8.724 | 否 | 17.8 |
| 7 | evolution | ★ `alpha`=0.9195, `ridge`=0.3 | -1.603 | 是 | 17.2 |
| 8 | evolution | `alpha`=0.6995, `epsilon`=0.06114 | -1.439 | 是 | 14.8 |
| 9 | evolution | `alpha`=0.9195, `ridge`=0.3, `warmup`=1, `epsilon`=0 | 4.152 | 否 | 16.3 |

- 重新生成：`python -m autoalgo generate --families selector.bandit --seed 0 --budget standard`
- 只重排文档（不重新评估）：`python -m autoalgo rerender`
- 产物：`generated/bandit_strategy_selector/algorithm.py`、`bandit_strategy_selector.md`、`blueprint.json`、`evaluation.json`
