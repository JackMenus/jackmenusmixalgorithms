# 路径签名因子模型（`signature_alpha`）

> 自主生成算法库（autoalgo）生成 · 协议 `AlphaModel`（因子层） · 模板族 `alpha.signature` · 生成时间 2026-10-22T11:36:39+00:00 · 生成器 v0.1.0

## 1. 概述

粗路径理论中的截断签名是路径的'万能特征'：一阶项是净变化（动量/反转），二阶项（含 Lévy 面积）刻画价量之间谁领先、路径是凸还是凹。把每个资产近 window 根 bar 的 [时间, log 价格, log 成交量] 路径签名作为特征，截面标准化后与未来收益做池化岭回归。

结论：**评估通过**。 样本外 RankIC=0.0219（t=2.29，显著），不及基线 0.0341；换手 0.605。

## 2. 所处协议层与接口

- 协议：`quantstack.contracts.AlphaModel`
- 接口：`fit(panel)` / `predict(panel, as_of) -> SignalBatch` / `describe()`
- 输入 / 输出：输入 `Panel = {字段: DataFrame(时间×资产)}`，输出每个资产的分数、期望收益（bps）与置信度
- quantstack 注册名：`autoalgo.signature_alpha`

## 3. 设计动机与假设

- 价量路径形状与未来短期收益之间存在线性可分的关系，且在训练期与预测期稳定。
- 窗口内 bar 数 ≥ 10；成交量为正。

## 4. 组件来源（建立在哪些库之上）

| 库 | 组件 | 在本算法中的作用 |
|---|---|---|
| `crossquant` | `crossquant.mathematics.signatures.signature / add_time` | 路径签名（深度 2）特征，Lévy 面积 |

涉及库：`crossquant`。

## 5. 算法流程与关键公式

1. 构造路径 X_t = [t/(w−1), log close, log(1+volume)]，每通道增量按窗口标准差归一（只保留形状）。
2. 按 Chen 恒等式向量化计算深度-2 签名：S¹ = ΣΔ，S² = Σ (Σ_{i<k}Δ_i)⊗Δ_k + ½ΣΔ_k⊗Δ_k（与 crossquant.signature 等价）。
3. 特征逐日截面 z 标准化、裁剪到 ±5；标签为截面标准化的未来收益。
4. 池化岭回归 β = (XᵀX + λnI)⁻¹Xᵀy；预测分数 z = cs_z(X_T β)。

关键公式：

- `S²_{ij} = ∫∫_{s<t} dX_i(s) dX_j(t)`
- `Lévy 面积 A_{ij} = ½(S²_{ij} − S²_{ji})`
- `β = (XᵀX + λ n I)⁻¹ Xᵀ y`

## 6. 参数

| 参数 | 取值 | 含义 | 搜索范围 |
|---|---|---|---|
| `window` | 30 | 路径窗口长度（bar） | {10, 15, 20, 30, 40} |
| `ridge` | 1.175 | 岭惩罚（相对样本量 ×1e-3） | [0.1, 30]（对数均匀） |
| `channels` | [logprice, logvolume] | 路径通道 | {[logprice], [logprice, logvolume]} |
| `normalize` | 是 | 增量是否按窗口标准差归一 | {是, 否} |

## 7. 评估结果

- 主目标：`rank_ic`（越大越好）；评估设置：{"train_ratio": 0.7, "n_train": 420, "n_oos": 179, "stride": 1, "horizon": 1, "n_assets": 60, "budget": "standard", "seed": 0}
- 相对第一基线的改善：-0.01218（已按方向调整，正数 = 更好）

| 指标 | 说明 | 本算法 | reference.ReversalVolumeAlpha |
|---|---|---|---|
| `rank_ic` | 样本外 RankIC 均值（越大越好） | 0.0219 | 0.03408 |
| `rank_icir` | RankIC 均值/标准差（年化） | 2.713 | 4.085 |
| `ic_t` | RankIC 的 t 统计量 | 2.287 | 3.443 |
| `ls_sharpe` | 五分组多空年化夏普 | 2.723 | 3.386 |
| `turnover` | 日均换手（分数变化幅度） | 0.605 | 0.5257 |
| `coverage` | 有效信号覆盖率 | 1 | 1 |
| `n_periods` | 样本外期数 | 179 | 179 |

- 判定：通过。样本外 RankIC=0.0219（t=2.29，显著），不及基线 0.0341；换手 0.605。
- 通过条件：样本外 RankIC > 0 且 t ≥ 2.0；'优于基线' 不是通过的必要条件（算法假设不同，覆盖的信息也不同）。
- 提示：合成面板只植入了反转与量价背离两类信号，参考基线恰好直接使用它们，因此在合成数据上难以被超越；请在真实面板上复核。
- 评估耗时：3.2 秒

## 8. 已知风险与失效条件

- 二阶签名特征之间高度相关，λ 太小会过拟合；λ 太大退化为一阶动量/反转。
- 特征对极端成交量敏感（已用 log1p 与裁剪缓解）。
- 线性读出忽略了签名的高阶交互；深度 >2 会使特征维数指数增长。

## 9. 接入方式

```python
from autoalgo.registry import make_alpha_model
model = make_alpha_model('autoalgo.signature_alpha')          # 或 quantstack: make_alpha_model(kind='autoalgo.signature_alpha')
model.fit(panel)
batch = model.predict(panel, as_of=panel['close'].index[-1])

# 全栈流水线
from quantstack.core.registry import build_sim_pipeline
pipe = build_sim_pipeline(panel, kinds={'alpha': 'autoalgo.signature_alpha'})
```

quantstack 侧：`quantstack.core.registry` 发现 `auto_algorithms/` 后，`kind='autoalgo.signature_alpha'` 可直接用于 `make_*` 工厂与 `build_sim_pipeline(kinds=...)`；也可用 `autoalgo.registry.make('signature_alpha')` 直接实例化。

## 10. 复现与血缘

- 生成器：autoalgo.pipeline.generate
- 生成器版本：0.1.0
- 提议器：evolution
- 随机种子：0
- 评估预算：standard
- 训练集比例：0.7
- 评估候选数：9
- 评估面板：{"n_days": 600, "n_assets": 60, "start": "2021-01-04", "end": "2023-04-21"}
- 选中候选得分：0.0219

候选评估记录（共 9 个，按评估顺序；★ 为最终选中的参数）：

| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |
|---|---|---|---|---|---|
| 1 | default | （默认） | 0.007091 | 否 | 3.3 |
| 2 | evolution | `ridge`=1.175 | 0.007282 | 否 | 3.5 |
| 3 | evolution | `normalize`=否 | 0.004305 | 否 | 3.2 |
| 4 | evolution | `channels`=[logprice], `normalize`=否 | 0.00778 | 否 | 2.9 |
| 5 | evolution | `ridge`=0.8011 | 0.007213 | 否 | 3.3 |
| 6 | evolution | `window`=40, `ridge`=1.175 | 0.01844 | 否 | 3.7 |
| 7 | evolution | `ridge`=13.34, `normalize`=否 | 0.004783 | 否 | 3.1 |
| 8 | evolution | `ridge`=0.5844, `channels`=[logprice] | 0.008191 | 否 | 3.1 |
| 9 | evolution | ★ `window`=30, `ridge`=1.175 | 0.0219 | 是 | 3.2 |

- 重新生成：`python -m autoalgo generate --families alpha.signature --seed 0 --budget standard`
- 只重排文档（不重新评估）：`python -m autoalgo rerender`
- 产物：`generated/signature_alpha/algorithm.py`、`signature_alpha.md`、`blueprint.json`、`evaluation.json`
