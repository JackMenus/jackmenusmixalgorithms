# 随机矩阵去噪协方差的多空组合（均值-方差 / 层次风险平价）（`denoised_riskparity_portfolio`）

> 自主生成算法库（autoalgo）生成 · 协议 `PortfolioConstructor`（组合层） · 模板族 `portfolio.denoised_riskparity` · 生成时间 2026-10-22T11:38:44+00:00 · 生成器 v0.1.0

## 1. 概述

按信号强度加权的多空组合不看协方差：它把风险集中在高波动标的上，也不对冲共同因子暴露。样本协方差在 N/T 不小时被噪声特征值主导（Marchenko–Pastur），直接做均值-方差会放大噪声；用随机矩阵理论（RIE / 特征值裁剪）去噪后再做均值-方差（或层次风险平价 + 信号倾斜），应当在同等信号下获得更高的风险调整收益。注意：若 alpha 本身与个股波动率成正比（如 quantstack 合成面板的植入信号），分数加权已接近最优，可提升空间有限。

结论：**评估通过**。 样本外夏普（3 个面板平均）4.04 vs 基线 3.94（逐面板胜率 67%）；年化波动 5.28% vs 6.00%；集中度 HHI 0.041 vs 0.038；换手 1.58 vs 1.59。

## 2. 所处协议层与接口

- 协议：`quantstack.contracts.PortfolioConstructor`
- 接口：`construct(signals, portfolio, market, limits) -> [TargetPosition]`
- 输入 / 输出：把信号变成目标持仓（数量、权重、紧急度、参与率）
- quantstack 注册名：`autoalgo.denoised_riskparity_portfolio`

## 3. 设计动机与假设

- 历史收益窗口 ≥ 30 期且覆盖全部候选标的；相关结构在窗口内近似平稳。
- 信号分数与期望收益的方向一致。

## 4. 组件来源（建立在哪些库之上）

| 库 | 组件 | 在本算法中的作用 |
|---|---|---|
| `crossquant` | `crossquant.physics.random_matrix.denoise_covariance` | 随机矩阵理论（MP 裁剪 / RIE / Ledoit–Wolf）协方差去噪 |
| `crossquant` | `crossquant.physics.networks.hierarchical_risk_parity` | 层次风险平价权重 |
| `crossquant` | `crossquant.mathematics.kelly.kelly_multi_asset` | 多资产 Kelly 杠杆 |
| `quantstack` | `quantstack.reference.portfolio.LongShortConstructor` | 多空选股与单票上限规则（基线） |

涉及库：`crossquant`, `quantstack`。

## 5. 算法流程与关键公式

1. 按分数取多头 top_k（score>0）与空头 top_k（score<0）；cost_threshold_bps>0 时剔除期望收益低于门槛的标的（默认 0：弱 alpha 下硬门槛会把组合压成一两只股票）。
2. 最近 lookback 期收益 → denoise_covariance(method) 得去噪协方差 Σ。
3. allocator=mv：Σ̂ 的最大特征对给出共同因子载荷 β̂=√λ₁·v₁ 与特质方差 σ²_ε=diag(Σ̂)−β̂²；腿内 w_i ∝ μ_i/σ²_ε,i；neutralize=dollar 按 net_target 分配两腿；neutralize=beta 用多空两腿规模抵消 Σw_iβ̂_i（每腿 35%–65%）；最后迭代裁剪单票上限。
4. allocator=hrp/ivp/equal：每腿先求风险权重 w_risk，再 w ∝ w_risk^(1−tilt) · (|z|/mean|z|)^tilt，归一到腿目标敞口并裁剪。
5. kelly_fraction>0 时用多资产 Kelly f*=Σ⁻¹μ 的分数杠杆限制总敞口。
6. 生成 TargetPosition：紧急度由置信度与 IC 半衰期决定（沿用参考实现规则），不在目标集合的持仓平仓。

关键公式：

- `λ₊ = σ²(1+√q)², q = N/T（MP 上界）`
- `mv: β̂ = √λ₁ v₁，σ²_ε = diag(Σ̂) − β̂²，w_i ∝ μ_i/σ²_ε,i（Grinold：μ_i = IC·σ_i·z_i 时即 w_i ∝ z_i/σ_i）；beta 中性时 g_L/g_S 使 Σ w_i β̂_i = 0`
- `HRP: w_cluster ∝ 1/Var(cluster)，自顶向下二分`
- `w_i ∝ w_risk,i^(1−tilt) · s_i^tilt`

## 6. 参数

| 参数 | 取值 | 含义 | 搜索范围 |
|---|---|---|---|
| `top_k` | 15 | 每腿最多标的数 | {8, 10, 15} |
| `gross_target` | 1 | 总敞口/NAV | — |
| `max_weight` | 0.08 | 单票权重上限 | {0.06, 0.08, 0.12} |
| `lookback` | 120 | 协方差估计窗口 | {60, 120, 180} |
| `denoise` | rie | 去噪方法 | {rie, clip, lw, none} |
| `allocator` | mv | 权重配置方法（mv/hrp/ivp/equal） | {mv, hrp, ivp, equal} |
| `tilt` | 0.9005 | 信号倾斜强度（仅 hrp/ivp/equal） | [0, 1] |
| `kelly_fraction` | 0 | 分数 Kelly 比例（0 关闭） | {0, 0.25, 0.5} |
| `cost_threshold_bps` | 0 | 期望收益成本门槛（0 关闭） | {0, 1, 2} |
| `neutralize` | dollar | mv 的两腿规模规则：beta 中性 / dollar 中性 | {dollar, beta} |

## 7. 评估结果

- 主目标：`sharpe`（越大越好）；评估设置：{"train_ratio": 0.7, "n_rebalances_per_panel": 89, "n_panels": 3, "rebalance_every": 2, "cost_bps": 5.0, "alpha": "ControlledICAlpha", "alpha_params": {"model_id": "bench_controlled_ic", "ic": 0.08, "horizon": 5, "seed": 0}, "budget": "standard", "seed": 0}
- 相对第一基线的改善：+0.09602（已按方向调整，正数 = 更好）

| 指标 | 说明 | 本算法 | reference.LongShortConstructor |
|---|---|---|---|
| `sharpe` | 组合年化夏普（扣成本） | 4.036 | 3.94 |
| `ann_return` | 年化收益 | 0.2112 | 0.2344 |
| `ann_vol` | 年化波动 | 0.0528 | 0.05999 |
| `max_drawdown` | 最大回撤 | -0.01729 | -0.01833 |
| `avg_turnover` | 每次再平衡的单边换手 | 1.583 | 1.585 |
| `avg_hhi` | 权重集中度 HHI 均值 | 0.0408 | 0.03827 |
| `max_weight` | 最大单票权重 | 0.08 | 0.08 |
| `n_rebalances` | 每个面板的再平衡次数 | 89 | 89 |
| `sharpe_per_panel` | 逐面板夏普 | [4.135, 2.821, 5.152] | [4.01, 2.884, 4.925] |
| `n_panels` | 评估面板数 | 3 | — |
| `wins_vs_baseline` | 逐种子 / 逐面板对比胜率 | 0.6667 | — |

- 判定：通过。样本外夏普（3 个面板平均）4.04 vs 基线 3.94（逐面板胜率 67%）；年化波动 5.28% vs 6.00%；集中度 HHI 0.041 vs 0.038；换手 1.58 vs 1.59。
- 通过条件：夏普 > 0 且 ≥ 基线 − 10%·|基线|；两者使用同一 alpha（ControlledICAlpha）与成本假设（5.0 bps 单边），每 2 日再平衡。
- 信号为受控 IC=0.08 的合成信号（含未来信息，只用于把组合构造质量与 alpha 质量分离），不代表可交易收益。
- 评估耗时：5.2 秒

## 8. 已知风险与失效条件

- 风险平价会把权重推向低波动资产，若信号主要在高波动资产上，tilt 过小会稀释 alpha；mv 对期望收益的尺度误差敏感（已用去噪 + 岭正则 + 单票上限约束）。
- HRP 的聚类顺序对相关矩阵的小扰动敏感；去噪方法 lw 在 T≫N 时与样本协方差差别很小。
- Kelly 杠杆依赖期望收益的尺度，信号 IC 高估时会过度加杠杆（已用 fraction 与上限约束）。

## 9. 接入方式

```python
from autoalgo.registry import make_portfolio
pc = make_portfolio('autoalgo.denoised_riskparity_portfolio')
pc.fit(panel)                     # 喂入历史收益用于协方差
targets = pc.construct(signals, portfolio_snapshot, market, limits)

# 全栈流水线
pipe = build_sim_pipeline(panel, kinds={'portfolio': 'autoalgo.denoised_riskparity_portfolio'}); pipe.portfolio.fit(panel)
```

quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind='autoalgo.denoised_riskparity_portfolio'` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；也可用 `autoalgo.registry.make('denoised_riskparity_portfolio')` 直接实例化。

## 10. 复现与血缘

- 生成器：autoalgo.pipeline.generate
- 生成器版本：0.1.0
- 提议器：evolution
- 随机种子：0
- 评估预算：standard
- 训练集比例：0.7
- 评估候选数：9
- 评估面板：{"n_days": 600, "n_assets": 60, "start": "2021-01-04", "end": "2023-04-21"}
- 选中候选得分：4.036

候选评估记录（共 9 个，按评估顺序；★ 为最终选中的参数）：

| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | default | （默认） | 3.383 | 是 | 4.4 |
| 2 | evolution | `neutralize`=beta | 3.194 | 是 | 4.3 |
| 3 | evolution | ★ `top_k`=15, `tilt`=0.9005 | 4.036 | 是 | 5.2 |
| 4 | evolution | `tilt`=0.3734 | 3.383 | 是 | 4.2 |
| 5 | evolution | `top_k`=15, `allocator`=hrp, `tilt`=0.9005 | 3.921 | 是 | 6.3 |
| 6 | evolution | `top_k`=15, `max_weight`=0.06, `allocator`=hrp, `tilt`=0.9005, `kelly_fraction`=0.5 | 3.919 | 是 | 7.2 |
| 7 | evolution | `top_k`=15, `lookback`=180, `tilt`=0.9005 | 3.99 | 是 | 5.2 |
| 8 | evolution | `top_k`=15, `max_weight`=0.06, `lookback`=180, `tilt`=0.9005, `cost_threshold_bps`=2 | 3.984 | 是 | 5.9 |
| 9 | evolution | `top_k`=15, `tilt`=0.9005, `neutralize`=beta | 3.854 | 是 | 5.2 |

- 重新生成：`python -m autoalgo generate --families portfolio.denoised_riskparity --seed 0 --budget standard`
- 只重排文档（不重新评估）：`python -m autoalgo rerender`
- 产物：`generated/denoised_riskparity_portfolio/algorithm.py`、`denoised_riskparity_portfolio.md`、`blueprint.json`、`evaluation.json`
