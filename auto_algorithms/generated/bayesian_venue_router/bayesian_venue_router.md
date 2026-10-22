# Thompson 采样 + 变点监测的场所路由叠加层（`bayesian_venue_router`）

> 自主生成算法库（autoalgo）生成 · 协议 `Router`（路由层） · 模板族 `router.bayesian_venue` · 生成时间 2026-10-22T12:03:18+00:00 · 生成器 v0.1.0

## 1. 概述

延迟感知路由器按费用与延迟分位数做静态最优拆分，但场所的真实成交质量（滑点、隐藏流动性、拒单）只能事后观察，并会随时间漂移。把每个场所当成一条臂，用高斯 Thompson 采样在'探索冷门场所'与'利用便宜场所'间权衡；再用 BOCPD 监测延迟分布突变，退化场所降权或改投。

结论：**评估通过**。 实现偏差 1.86 bps vs 底层路由 9.44 bps（逐种子胜率 100%）；成交率 0.994 vs 1.000；重分配切片 75 个，奖励样本 6368 个。

## 2. 所处协议层与接口

- 协议：`quantstack.contracts.Router`
- 接口：`route(slice, snapshot) -> RoutingDecision` / `on_event(event)` / `venue_stats()`
- 输入 / 输出：把子单切片拆到各交易场所
- quantstack 注册名：`autoalgo.bayesian_venue_router`

## 3. 设计动机与假设

- 场所质量差异在同一会话内持续足够长，使得几十笔成交就能学到。
- Fill.latency_us 与 fee 由网关正确填写（§8）。

## 4. 组件来源（建立在哪些库之上）

| 库 | 组件 | 在本算法中的作用 |
|---|---|---|
| `crossquant` | `crossquant.control.changepoint.bocpd` | 贝叶斯在线变点概率 |
| `crossquant` | `crossquant.control.bandits.ThompsonGaussian` | 高斯 Thompson 采样（场所选择） |
| `latency_routing` | `latency_routing.make_router（经 quantstack.adapters.latency_routing_adapter）` | 延迟感知 SOR：扫单 + Cont-Kukanov 拆分 + 同步到达 |

涉及库：`crossquant`, `latency_routing`。

## 5. 算法流程与关键公式

1. 底层路由器（latency_routing 适配器）给出基准分配 q_base。
2. 吃单切片：对每个场所从后验 N(m_v, v_v) 采样，平移为正后归一得 w_TS；最终份额 = (1−mix)·q_base/Q + mix·w_TS，裁剪到 [min_share, max_share] 与 1.3× 可见量。
3. 成交回报：奖励 = −(side·(成交价−路由时中间价)/中间价 + 费用)·1e4 bps，更新对应场所后验；拒单计 −reject_penalty。
4. 每个场所保留最近 latency_window 个回报延迟，log 标准化后 BOCPD；变点概率 > 阈值且延迟上行 → 退化：吃单份额 ×(1−penalty)，被动单改投后验均值最高的正常场所。

关键公式：

- `m_v | data ~ N((Σr/σ²)/(1/τ²+n/σ²), 1/(1/τ²+n/σ²))`
- `share_v = (1−mix)·q_v/Q + mix·w_TS,v`

## 6. 参数

| 参数 | 取值 | 含义 | 搜索范围 |
|---|---|---|---|
| `mix` | 0.5 | 向 Thompson 权重倾斜的比例 | [0.2, 0.9] |
| `obs_var_bps2` | 4.288 | 奖励观测方差 | [1, 16]（对数均匀） |
| `prior_var` | 25 | 先验方差 | {9, 25, 100} |
| `min_share` | 0.05 | 单场所最小份额 | — |
| `max_share` | 0.8 | 单场所最大份额 | {0.6, 0.8, 1} |
| `latency_window` | 40 | 延迟样本窗口 | {20, 40, 80} |
| `degrade_hazard` | 0.05 | BOCPD 先验变点概率 | — |
| `degrade_threshold` | 0.5 | 退化判定阈值 | [0.3, 0.8] |
| `degrade_penalty` | 0.6 | 退化场所份额惩罚 | [0.3, 1] |

## 7. 评估结果

- 主目标：`shortfall_bps`（越小越好）；评估设置：{"n_orders_per_seed": 5, "seeds": [1, 2, 3], "duration_s": 600, "step_s": 1.0, "mc_samples": 400, "execution": "almgren_chriss", "budget": "standard", "seed": 0}
- 相对第一基线的改善：+7.579（已按方向调整，正数 = 更好）

| 指标 | 说明 | 本算法 | latency_routing.make_router |
|---|---|---|---|
| `shortfall_bps` | 实现偏差均值（bps，越小越好） | 1.858 | 9.438 |
| `shortfall_median_bps` | 实现偏差中位数 | 3.455 | 10.45 |
| `shortfall_std_bps` | 实现偏差标准差 | 12.44 | 9.798 |
| `fill_ratio` | 成交率 | 0.9942 | 1 |
| `maker_ratio` | 被动成交比例 | 0.5305 | 0.4268 |
| `n_orders` | 母单数 | 15 | 15 |
| `n_seeds` | 随机种子数 | 3 | 3 |
| `wins_vs_baseline` | 逐种子 / 逐面板对比胜率 | 1 | — |
| `n_rebalanced` | 被重新分配的切片数 | 75 | — |
| `n_rewards` | 收到的奖励样本数 | 6368 | — |

- 判定：通过。实现偏差 1.86 bps vs 底层路由 9.44 bps（逐种子胜率 100%）；成交率 0.994 vs 1.000；重分配切片 75 个，奖励样本 6368 个。
- 通过条件：实现偏差 ≤ 基线 + 1.0 bps 且成交率 ≥ 基线 − 0.02。
- 评估耗时：126.8 秒

## 8. 已知风险与失效条件

- Thompson 采样会在早期把量导向噪声中偶然'便宜'的场所；prior_var 与 obs_var 控制探索速度。
- 份额重分配可能超过某场所可见量导致部分未成交（已限 1.3×），底层的同步到达计划在重分配后仍沿用。
- BOCPD 对延迟突刺敏感，latency_window 太短会误判。

## 9. 接入方式

```python
from autoalgo.registry import make_router
router = make_router('autoalgo.bayesian_venue_router', venues=['NYSE', 'NASDAQ', 'ARCA', 'IEX'])   # 叠加在 latency_routing 之上

# 全栈流水线：build_sim_pipeline(panel, kinds={'router': 'autoalgo.bayesian_venue_router'})
```

quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind='autoalgo.bayesian_venue_router'` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；也可用 `autoalgo.registry.make('bayesian_venue_router')` 直接实例化。

## 10. 复现与血缘

- 生成器：autoalgo.pipeline.generate
- 生成器版本：0.1.0
- 提议器：evolution
- 随机种子：0
- 评估预算：standard
- 训练集比例：0.7
- 评估候选数：9
- 评估面板：{"n_days": 600, "n_assets": 60, "start": "2021-01-04", "end": "2023-04-21"}
- 选中候选得分：-1.858

候选评估记录（共 9 个，按评估顺序；★ 为最终选中的参数）：

| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | default | （默认） | 3.546 | 是 | 216 |
| 2 | evolution | `mix`=0.6512 | 7.596 | 是 | 118.3 |
| 3 | evolution | ★ `obs_var_bps2`=4.288 | 1.858 | 是 | 126.8 |
| 4 | evolution | `degrade_threshold`=0.3515, `degrade_penalty`=0.5984 | 4.836 | 是 | 117.9 |
| 5 | evolution | `mix`=0.334, `obs_var_bps2`=4.288 | 6.942 | 是 | 132 |
| 6 | evolution | `obs_var_bps2`=4.288, `degrade_threshold`=0.6816, `degrade_penalty`=0.5669 | 2.2 | 是 | 139.2 |
| 7 | evolution | `degrade_threshold`=0.5087 | 3.546 | 是 | 142.6 |
| 8 | evolution | `obs_var_bps2`=4.288, `prior_var`=100, `max_share`=1, `degrade_threshold`=0.6816, `degrade_penalty`=0.5669 | 2.61 | 是 | 140 |
| 9 | evolution | `mix`=0.5067, `obs_var_bps2`=4.288 | 8.648 | 是 | 138.9 |

- 重新生成：`python -m autoalgo generate --families router.bayesian_venue --seed 0 --budget standard`
- 只重排文档（不重新评估）：`python -m autoalgo rerender`
- 产物：`generated/bayesian_venue_router/algorithm.py`、`bayesian_venue_router.md`、`blueprint.json`、`evaluation.json`
