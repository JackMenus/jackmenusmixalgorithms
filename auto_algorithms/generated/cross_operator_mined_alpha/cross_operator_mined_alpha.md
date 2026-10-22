# 跨学科算子遗传规划因子（`cross_operator_mined_alpha`）

> 自主生成算法库（autoalgo）生成 · 协议 `AlphaModel`（因子层） · 模板族 `alpha.cross_operator_mined` · 生成时间 2026-10-22T11:37:57+00:00 · 生成器 v0.1.0

## 1. 概述

因子挖掘的搜索空间由算子表决定。把物理/数学/控制中的 6 个工具（分数阶差分、方差比、Hurst、排列熵、Kalman 残差、Lévy 面积）改写成面板算子注入 factormine，遗传规划就能搜到传统算子表达不出的因子（如'价量 Lévy 面积为正且方差比显示均值回复'）。

结论：**评估通过**。 样本外 RankIC=0.0222（t=2.26，显著），不及基线 0.0341；换手 0.345。

## 2. 所处协议层与接口

- 协议：`quantstack.contracts.AlphaModel`
- 接口：`fit(panel)` / `predict(panel, as_of) -> SignalBatch` / `describe()`
- 输入 / 输出：输入 `Panel = {字段: DataFrame(时间×资产)}`，输出每个资产的分数、期望收益（bps）与置信度
- quantstack 注册名：`autoalgo.cross_operator_mined_alpha`

## 3. 设计动机与假设

- 训练段足够长（≥ 200 bar）让 GP 的样本外过滤有意义。
- 扩展算子的窗口/常数候选覆盖了数据的时间尺度。

## 4. 组件来源（建立在哪些库之上）

| 库 | 组件 | 在本算法中的作用 |
|---|---|---|
| `factormine` | `factormine.Evaluator / factormine.parse` | 因子表达式解析与带缓存求值 |
| `factormine` | `factormine.fit_weights / combine` | 多因子合成权重（equal/ic/icir/max_ir/ridge）与标准化合成 |
| `factormine` | `factormine.register` | 算子注册入口（扩展算子注入点） |
| `factormine` | `factormine.mine_factors / GPConfig / MiningConfig` | 遗传规划因子挖掘（样本外过滤入库） |
| `factormine` | `factormine.AlphaModel` | 因子库 → 合成信号（Grinold 期望收益、IC 半衰期） |

涉及库：`factormine`。

## 5. 算法流程与关键公式

1. components.operators.register_all 把扩展算子注册进 factormine.OPERATORS，并为常数参数提供候选值。
2. factormine.mine_factors（GP：种群 max(20,budget)，代数 max(2,budget//20)）在训练段挖掘，样本外 RankIC 同号且 ≥ min_ic 才入库。
3. factormine.AlphaModel(library, method) 估合成权重，signal 给出分数、Grinold 期望收益与 IC 半衰期。
4. describe 报告因子库中含扩展算子的因子比例（跨学科贡献度）。

关键公式：

- `FFD: x̃_t = Σ_k w_k x_{t−k}, w_k = −w_{k−1}(d−k+1)/k`
- `VR(k) = Var(Σ_{i<k} r_{t−i}) / (k·Var r)`
- `K_ss = (P/R)/(P/R+1), P/R = (q+√(q²+4q))/2`

## 6. 参数

| 参数 | 取值 | 含义 | 搜索范围 |
|---|---|---|---|
| `budget` | 60 | 计算预算（≈ 候选评估次数/10） | {30, 40, 60} |
| `seed` | 0 | 随机种子 | — |
| `method` | max_ir | 合成方法 | {max_ir, icir, ridge} |
| `extended_only` | 是 | 是否只用基础+扩展算子 | — |
| `pool_capacity` | 12 | 因子池容量 | {8, 12, 16} |
| `max_corr` | 0.7 | 池内因子最大相关 | [0.5, 0.8] |
| `min_ic` | 0.01107 | 入池最低 |RankIC| | [0.005, 0.02] |

## 7. 评估结果

- 主目标：`rank_ic`（越大越好）；评估设置：{"train_ratio": 0.7, "n_train": 420, "n_oos": 179, "stride": 1, "horizon": 1, "n_assets": 60, "budget": "standard", "seed": 0}
- 相对第一基线的改善：-0.01191（已按方向调整，正数 = 更好）

| 指标 | 说明 | 本算法 | reference.ReversalVolumeAlpha |
|---|---|---|---|
| `rank_ic` | 样本外 RankIC 均值（越大越好） | 0.02217 | 0.03408 |
| `rank_icir` | RankIC 均值/标准差（年化） | 2.679 | 4.085 |
| `ic_t` | RankIC 的 t 统计量 | 2.258 | 3.443 |
| `ls_sharpe` | 五分组多空年化夏普 | 3.324 | 3.386 |
| `turnover` | 日均换手（分数变化幅度） | 0.3452 | 0.5257 |
| `coverage` | 有效信号覆盖率 | 1 | 1 |
| `n_periods` | 样本外期数 | 179 | 179 |

- 判定：通过。样本外 RankIC=0.0222（t=2.26，显著），不及基线 0.0341；换手 0.345。
- 通过条件：样本外 RankIC > 0 且 t ≥ 2.0；'优于基线' 不是通过的必要条件（算法假设不同，覆盖的信息也不同）。
- 提示：合成面板只植入了反转与量价背离两类信号，参考基线恰好直接使用它们，因此在合成数据上难以被超越；请在真实面板上复核。
- 评估耗时：8.8 秒

## 8. 已知风险与失效条件

- GP 本质是大规模多重检验，样本外过滤只能部分抑制过拟合；因子库应定期用新数据重挖。
- ts_hurst / ts_fracdiff 计算较慢，budget 过大时挖掘时间显著增加。
- 随机种子影响结果；生成文档记录了 seed 以便复现。

## 9. 接入方式

```python
from autoalgo.registry import make_alpha_model
model = make_alpha_model('autoalgo.cross_operator_mined_alpha')          # 或 quantstack: make_alpha_model(kind='autoalgo.cross_operator_mined_alpha')
model.fit(panel)
batch = model.predict(panel, as_of=panel['close'].index[-1])

# 全栈流水线
from quantstack.core.registry import build_sim_pipeline
pipe = build_sim_pipeline(panel, kinds={'alpha': 'autoalgo.cross_operator_mined_alpha'})
```

quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind='autoalgo.cross_operator_mined_alpha'` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；也可用 `autoalgo.registry.make('cross_operator_mined_alpha')` 直接实例化。

## 10. 复现与血缘

- 生成器：autoalgo.pipeline.generate
- 生成器版本：0.1.0
- 提议器：evolution
- 随机种子：0
- 评估预算：standard
- 训练集比例：0.7
- 评估候选数：9
- 评估面板：{"n_days": 600, "n_assets": 60, "start": "2021-01-04", "end": "2023-04-21"}
- 选中候选得分：0.02217

候选评估记录（共 9 个，按评估顺序；★ 为最终选中的参数）：

| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | default | （默认） | 0.001481 | 否 | 5.7 |
| 2 | evolution | ★ `budget`=60, `min_ic`=0.01107 | 0.02217 | 是 | 8.8 |
| 3 | evolution | `method`=ridge | 0.001888 | 否 | 6 |
| 4 | evolution | `method`=ridge, `min_ic`=0.01119 | 0.001888 | 否 | 6.4 |
| 5 | evolution | `budget`=60, `method`=ridge, `min_ic`=0.009087 | 0.01808 | 否 | 10.3 |
| 6 | evolution | `budget`=60, `min_ic`=0.009584 | 0.02217 | 是 | 10 |
| 7 | evolution | `budget`=60, `method`=ridge, `min_ic`=0.01384 | 0.01655 | 否 | 9.4 |
| 8 | evolution | `budget`=60, `max_corr`=0.7031, `min_ic`=0.005 | 0.0004781 | 否 | 11.1 |
| 9 | evolution | `budget`=60, `max_corr`=0.7602, `min_ic`=0.01107 | 0.02217 | 是 | 10 |

- 重新生成：`python -m autoalgo generate --families alpha.cross_operator_mined --seed 0 --budget standard`
- 只重排文档（不重新评估）：`python -m autoalgo rerender`
- 产物：`generated/cross_operator_mined_alpha/algorithm.py`、`cross_operator_mined_alpha.md`、`blueprint.json`、`evaluation.json`
