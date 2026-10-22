# 市场状态门控多因子模型（`regime_gated_alpha`）

> 自主生成算法库（autoalgo）生成 · 协议 `AlphaModel`（因子层） · 模板族 `alpha.regime_gated` · 生成时间 2026-10-22T11:36:10+00:00 · 生成器 v0.1.0

## 1. 概述

同一组因子在不同市场状态下的有效性不同：均值回复状态（Hurst<0.5）里反转、量价背离因子更强，趋势状态（Hurst>0.5）里动量更强。用物理学中的 DFA-Hurst 指数识别状态、分状态估计合成权重，并在变点刚发生（BOCPD 概率高、状态标签不可靠）时收缩置信度。

结论：**评估通过**。 样本外 RankIC=0.0234（t=2.44，显著），不及基线 0.0341；换手 0.646。

## 2. 所处协议层与接口

- 协议：`quantstack.contracts.AlphaModel`
- 接口：`fit(panel)` / `predict(panel, as_of) -> SignalBatch` / `describe()`
- 输入 / 输出：输入 `Panel = {字段: DataFrame(时间×资产)}`，输出每个资产的分数、期望收益（bps）与置信度
- quantstack 注册名：`autoalgo.regime_gated_alpha`

## 3. 设计动机与假设

- 市场状态可由 Hurst 指数在 60–180 bar 窗口内识别，且状态持续时间长于窗口更新步长。
- 各状态内因子权重稳定，训练期每个状态有足够样本（min_regime_obs）。
- 面板为日频、横截面 ≥ 20 个资产，字段齐全（open/high/low/close/volume/returns）。

## 4. 组件来源（建立在哪些库之上）

| 库 | 组件 | 在本算法中的作用 |
|---|---|---|
| `factormine` | `factormine.Evaluator / factormine.parse` | 因子表达式解析与带缓存求值 |
| `factormine` | `factormine.fit_weights / combine` | 多因子合成权重（equal/ic/icir/max_ir/ridge）与标准化合成 |
| `crossquant` | `crossquant.physics.hurst.dfa` | DFA Hurst 指数：趋势 / 均值回复强度 |
| `crossquant` | `crossquant.control.changepoint.bocpd` | 贝叶斯在线变点概率 |

涉及库：`factormine`, `crossquant`。

## 5. 算法流程与关键公式

1. 用 factormine 求值 K 个因子表达式，并计算未来 h 期收益。
2. 等权市场收益 → 滚动 DFA Hurst（crossquant）→ 按阈值 lo/hi 标注 均值回复 / 中性 / 趋势。
3. 每个状态子样本上用 factormine.fit_weights(method) 估合成权重；样本不足回退全局权重。
4. 预测时按当前状态取权重合成分数 z，期望收益 E[r]=IC_state·σ_idio·z（Grinold）。
5. BOCPD 给出'最近发生变点'的概率 p，置信度乘 (1−shrink·p)。

关键公式：

- `H = DFA 斜率：F(s) ∝ s^H`
- `w_state = argmax IC_IR（Qian–Hua）: w ∝ Σ_IC⁻¹ μ_IC`
- `E[r_i] = IC_state · σ_i · z_i`
- `conf_i = clip(|IC|/0.1)·(1 − shrink·p_change)·min(|z_i|/2, 1)`

## 6. 参数

| 参数 | 取值 | 含义 | 搜索范围 |
|---|---|---|---|
| `hurst_window` | 180 | 滚动 Hurst 窗口（bar） | {60, 90, 120, 180} |
| `lo` | 0.45 | 均值回复阈值（H<lo） | [0.4, 0.49] |
| `hi` | 0.55 | 趋势阈值（H>hi） | [0.51, 0.6] |
| `hazard` | 0.008333 | BOCPD 先验变点概率 | [0.0025, 0.025]（对数均匀） |
| `method` | max_ir | 合成方法 | {max_ir, icir, ridge, equal} |
| `shrink_on_change` | 0.402 | 变点概率对置信度的收缩强度 | [0, 0.9] |
| `min_regime_obs` | 40 | 状态样本下限 | {30, 40, 60} |

## 7. 评估结果

- 主目标：`rank_ic`（越大越好）；评估设置：{"train_ratio": 0.7, "n_train": 420, "n_oos": 179, "stride": 1, "horizon": 1, "n_assets": 60, "budget": "standard", "seed": 0}
- 相对第一基线的改善：-0.0107（已按方向调整，正数 = 更好）

| 指标 | 说明 | 本算法 | reference.ReversalVolumeAlpha |
|---|---|---|---|
| `rank_ic` | 样本外 RankIC 均值（越大越好） | 0.02338 | 0.03408 |
| `rank_icir` | RankIC 均值/标准差（年化） | 2.895 | 4.085 |
| `ic_t` | RankIC 的 t 统计量 | 2.44 | 3.443 |
| `ls_sharpe` | 五分组多空年化夏普 | 1.911 | 3.386 |
| `turnover` | 日均换手（分数变化幅度） | 0.646 | 0.5257 |
| `coverage` | 有效信号覆盖率 | 1 | 1 |
| `n_periods` | 样本外期数 | 179 | 179 |

- 判定：通过。样本外 RankIC=0.0234（t=2.44，显著），不及基线 0.0341；换手 0.646。
- 通过条件：样本外 RankIC > 0 且 t ≥ 2.0；'优于基线' 不是通过的必要条件（算法假设不同，覆盖的信息也不同）。
- 提示：合成面板只植入了反转与量价背离两类信号，参考基线恰好直接使用它们，因此在合成数据上难以被超越；请在真实面板上复核。
- 评估耗时：3.7 秒

## 8. 已知风险与失效条件

- 状态识别滞后：Hurst 估计需要较长窗口，状态切换初期会用错权重（变点收缩只能部分缓解）。
- 分状态估计样本更少，权重估计噪声更大；过细的阈值（lo/hi 太靠近 0.5）会放大噪声。
- 合成数据上的改善未必迁移到真实市场；应在真实面板上重新运行生成管线复核。

## 9. 接入方式

```python
from autoalgo.registry import make_alpha_model
model = make_alpha_model('autoalgo.regime_gated_alpha')          # 或 quantstack: make_alpha_model(kind='autoalgo.regime_gated_alpha')
model.fit(panel)
batch = model.predict(panel, as_of=panel['close'].index[-1])

# 全栈流水线
from quantstack.core.registry import build_sim_pipeline
pipe = build_sim_pipeline(panel, kinds={'alpha': 'autoalgo.regime_gated_alpha'})
```

quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind='autoalgo.regime_gated_alpha'` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；也可用 `autoalgo.registry.make('regime_gated_alpha')` 直接实例化。

## 10. 复现与血缘

- 生成器：autoalgo.pipeline.generate
- 生成器版本：0.1.0
- 提议器：evolution
- 随机种子：0
- 评估预算：standard
- 训练集比例：0.7
- 评估候选数：9
- 评估面板：{"n_days": 600, "n_assets": 60, "start": "2021-01-04", "end": "2023-04-21"}
- 选中候选得分：0.02338

候选评估记录（共 9 个，按评估顺序；★ 为最终选中的参数）：

| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | default | （默认） | 0.02109 | 是 | 4.4 |
| 2 | evolution | `hazard`=0.00896, `min_regime_obs`=30 | 0.02109 | 是 | 4.3 |
| 3 | evolution | `lo`=0.4565 | 0.02128 | 是 | 3.8 |
| 4 | evolution | `hazard`=0.005125, `method`=ridge | 0.01961 | 是 | 3.6 |
| 5 | evolution | `lo`=0.4147, `shrink_on_change`=0.5074 | 0.01551 | 否 | 3.9 |
| 6 | evolution | ★ `hurst_window`=180, `shrink_on_change`=0.402 | 0.02338 | 是 | 3.7 |
| 7 | evolution | `hi`=0.5688 | 0.01593 | 否 | 3.6 |
| 8 | evolution | `hurst_window`=60, `lo`=0.4565 | -0.0007075 | 否 | 3.3 |
| 9 | evolution | `shrink_on_change`=0.402 | 0.02109 | 是 | 3.4 |

- 重新生成：`python -m autoalgo generate --families alpha.regime_gated --seed 0 --budget standard`
- 只重排文档（不重新评估）：`python -m autoalgo rerender`
- 产物：`generated/regime_gated_alpha/algorithm.py`、`regime_gated_alpha.md`、`blueprint.json`、`evaluation.json`
