# Kalman 公允价 + PID 反馈的 Almgren–Chriss 执行（`kalman_pid_execution`）

> 自主生成算法库（autoalgo）生成 · 协议 `ExecutionStrategy`（执行层） · 模板族 `execution.kalman_pid` · 生成时间 2026-10-22T11:39:17+00:00 · 生成器 v0.1.0

## 1. 概述

Almgren–Chriss 轨迹是开环的：被动成交偏慢时会持续落后，最后不得不大量吃单；它也不看盘中价格相对公允价的位置。用 Kalman 局部线性趋势模型滤波中间价得到平滑公允价：reversion 模式把中间价对公允价的偏离视为暂时性（自身冲击、报价噪声），偏贵时放慢并后撤一跳等价格回来、便宜时加速并改善一跳；momentum 模式则按斜率外推追趋势。两种用法谁对由评估决定。再用 PID/LQR 把'计划−实际完成比例'的误差闭环反馈到计划量上，防止持续落后或超前。

结论：**评估通过**。 实现偏差 -1.62 bps vs order_placement.almgren_chriss 3.21 bps（逐种子胜率 75%）；成交率 1.000，被动成交占比 0.43。

## 2. 所处协议层与接口

- 协议：`quantstack.contracts.ExecutionStrategy`
- 接口：`start / on_market / on_fill / on_event / cancels / is_complete / state`；经 `ExecutionStrategyFactory.create(name, order, **params)` 创建
- 输入 / 输出：把母单切成子单切片（IOC / POST_ONLY …）
- quantstack 注册名：`autoalgo.kalman_pid_execution`

## 3. 设计动机与假设

- 中间价对平滑公允价的偏离在 10–60 秒尺度上会部分回复（冲击衰减、报价噪声），或存在可外推的短期趋势。
- 快照包含 adv / volatility，用于 AC 参数与 Kalman 噪声尺度。

## 4. 组件来源（建立在哪些库之上）

| 库 | 组件 | 在本算法中的作用 |
|---|---|---|
| `crossquant` | `crossquant.KalmanTrendEstimator` | 局部线性趋势 Kalman：公允价滤波与外推 |
| `crossquant` | `crossquant.PIDParticipationController / LQRTrackingController` | 执行进度反馈控制 |
| `order_placement` | `order_placement.adapters.quantstack_exec.QuantstackExecution` | 执行策略基类：计划曲线、追单/被动挂单、在途簿记、§8 撤单约定 |
| `order_placement` | `order_placement.adapters.quantstack_exec.AlmgrenChrissExec.plan` | Almgren–Chriss 最优轨迹（urgency→λ，alpha→漂移） |
| `order_placement` | `order_placement.queue.models.queue_value` | 排队价值判断（挂近端还是改善一跳） |
| `order_placement` | `order_placement.make_execution_factory` | 8 种基础执行策略工厂（被 AutoExecutionFactory 包裹） |

涉及库：`crossquant`, `order_placement`。

## 5. 算法流程与关键公式

1. 继承 order_placement.QuantstackExecution，计划曲线取 AlmgrenChrissExec.plan（urgency→λ，alpha→漂移）。
2. 每个 on_market：Kalman 更新 (level, slope)。reversion：pred = −side·(mid − level)/mid；momentum：pred = side·(slope·h)/mid（bps，正 = 不利 / 应加速）。
3. urgency_target = AC 目标 + u·Q（u = 控制器对误差 e=(目标−已成交)/Q 的输出）± drift_gain·k·切片（k=|pred|/阈值，上限 drift_max_slices）。
4. passive_price：pred > improve_threshold → 近端改善一跳（不穿越对手价）；pred < −阈值且 backoff → 后撤一跳。
5. 追单 / 扫尾 / 撤单簿记全部沿用基类，满足 §8 全部约定。

关键公式：

- `x_{t+dt} = [[1, dt],[0, 1]] x_t + w，观测 mid = [1 0] x + v`
- `reversion: pred = −side·(mid − level)/mid·1e4；momentum: pred = side·slope·h/mid·1e4`
- `u_t = Kp e_t + Ki Σ e dt + Kd de/dt（限幅）`
- `target_t = AC(t) + u_t·Q + sign(pred)·g·min(|pred|/θ, 3)·slice`

## 6. 参数

| 参数 | 取值 | 含义 | 搜索范围 |
|---|---|---|---|
| `mode` | reversion | Kalman 信号用法：reversion / momentum | {reversion, momentum} |
| `controller` | lqr | pid / lqr | {pid, lqr} |
| `kp` | 0.5 | 比例增益 | [0.1, 1.5] |
| `ki` | 0.05 | 积分增益（每秒） | [0.002, 0.2]（对数均匀） |
| `ctrl_out_max` | 0.2 | 控制量上限（占母单比例） | {0.1, 0.2, 0.3, 0.5} |
| `kf_obs_mult` | 10 | 观测噪声倍数（越大公允价越平滑） | {1, 10, 50, 200} |
| `fair_horizon_s` | 30 | momentum 外推时长（秒） | {10, 30, 60} |
| `drift_threshold_bps` | 1 | 信号生效阈值（bps） | [0.3, 4] |
| `drift_gain` | 1 | 每倍阈值提前/推后的切片数（0 = 关闭节奏调节） | [0, 2] |
| `improve_threshold_bps` | 1.5 | 改善/后撤一跳的阈值（bps） | [0.5, 6] |
| `backoff` | 1 | 信号有利时是否后撤一跳 | {0, 1} |

## 7. 评估结果

- 主目标：`shortfall_bps`（越小越好）；评估设置：{"n_orders_per_seed": 4, "seeds": [1, 2, 3, 4], "duration_s": 900, "step_s": 1.0, "params": {"mode": "reversion", "controller": "lqr", "kp": 0.5, "ki": 0.05, "ctrl_out_max": 0.2, "kf_obs_mult": 10.0, "fair_horizon_s": 30.0, "drift_threshold_bps": 1.0, "drift_gain": 1.0, "improve_threshold_bps": 1.5, "backoff": 1.0}, "market": "quantstack 合成多场所模拟器（参考路由器）", "budget": "standard", "seed": 0}
- 相对第一基线的改善：+4.838（已按方向调整，正数 = 更好）

| 指标 | 说明 | 本算法 | order_placement.almgren_chriss | order_placement.twap |
|---|---|---|---|---|
| `shortfall_bps` | 实现偏差均值（bps，越小越好） | -1.625 | 3.214 | 3.865 |
| `shortfall_median_bps` | 实现偏差中位数 | -1.947 | 5.824 | 3.587 |
| `shortfall_std_bps` | 实现偏差标准差 | 13.27 | 9.043 | 24.22 |
| `fill_ratio` | 成交率 | 1 | 1 | 1 |
| `maker_ratio` | 被动成交比例 | 0.4347 | 0.8995 | 0.9906 |
| `n_orders` | 母单数 | 16 | 16 | 16 |
| `n_seeds` | 随机种子数 | 4 | — | — |
| `wins_vs_baseline` | 逐种子 / 逐面板对比胜率 | 0.75 | — | — |

- 判定：通过。实现偏差 -1.62 bps vs order_placement.almgren_chriss 3.21 bps（逐种子胜率 75%）；成交率 1.000，被动成交占比 0.43。
- 通过条件：平均实现偏差 ≤ 第一基线 + 1.0 bps 且成交率 ≥ 0.95。单路径噪声可达 20 bps，种子数越多结论越可靠。
- 评估耗时：2.7 秒

## 8. 已知风险与失效条件

- reversion 模式在真实趋势行情中会越贵越等、越便宜越追（被动等待的机会成本）；momentum 模式会把自身冲击当成趋势追涨杀跌。
- 阈值过低会频繁改价增加撤改量；PID 积分项在长时间无成交时累积，ctrl_out_max 限幅防止突然大量吃单。
- 模拟器只对主动成交施加冲击、被动成交近乎免费，因此更被动的配置在模拟器里天然占优；真实市场的逆向选择成本会改变最优参数。

## 9. 接入方式

```python
from autoalgo.registry import make_execution_factory
factory = make_execution_factory('autoalgo.kalman_pid_execution')   # 含挂单库全部策略 + kalman_pid
strategy = factory.create('kalman_pid', exec_order)

# 全栈流水线：kinds={'execution': 'autoalgo.kalman_pid_execution'}，母单 strategy_hint='kalman_pid' 或由选择器选择
```

quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind='autoalgo.kalman_pid_execution'` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；也可用 `autoalgo.registry.make('kalman_pid_execution')` 直接实例化。

## 10. 复现与血缘

- 生成器：autoalgo.pipeline.generate
- 生成器版本：0.1.0
- 提议器：evolution
- 随机种子：0
- 评估预算：standard
- 训练集比例：0.7
- 评估候选数：9
- 评估面板：{"n_days": 600, "n_assets": 60, "start": "2021-01-04", "end": "2023-04-21"}
- 选中候选得分：1.625

候选评估记录（共 9 个，按评估顺序；★ 为最终选中的参数）：

| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | default | （默认） | -0.2144 | 是 | 9.5 |
| 2 | evolution | `ki`=0.1001 | 3.777 | 是 | 3.8 |
| 3 | evolution | ★ `controller`=lqr | -1.625 | 是 | 2.7 |
| 4 | evolution | `kp`=0.7445, `ctrl_out_max`=0.1 | 4.93 | 否 | 2.7 |
| 5 | evolution | `controller`=lqr, `fair_horizon_s`=60 | -1.625 | 是 | 2.6 |
| 6 | evolution | `controller`=lqr, `kf_obs_mult`=50, `fair_horizon_s`=60 | 4.22 | 否 | 2.8 |
| 7 | evolution | `controller`=lqr, `kf_obs_mult`=200, `fair_horizon_s`=60 | 2.399 | 是 | 2.5 |
| 8 | evolution | `mode`=momentum, `controller`=lqr, `kp`=0.7033, `fair_horizon_s`=60 | 6.331 | 否 | 3.1 |
| 9 | evolution | `controller`=lqr, `kp`=0.7205, `ki`=0.1605 | -1.625 | 是 | 2.9 |

- 重新生成：`python -m autoalgo generate --families execution.kalman_pid --seed 0 --budget standard`
- 只重排文档（不重新评估）：`python -m autoalgo rerender`
- 产物：`generated/kalman_pid_execution/algorithm.py`、`kalman_pid_execution.md`、`blueprint.json`、`evaluation.json`
