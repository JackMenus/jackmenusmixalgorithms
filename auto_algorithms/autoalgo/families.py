"""模板族（family）定义：每个模板族 = 一个可实例化的新算法构造方式 + 参数搜索空间 + 可解释性文本骨架。

生成管线对每个模板族做"提议参数 → 实例化 → 评估 → 选优 → 生成代码与中文说明"。
模板族的实现类都在 ``autoalgo.components``；本模块只负责登记与描述，不含数值代码。

搜索维度写法（``search_space``）：
  ("int", lo, hi) / ("float", lo, hi) / ("logfloat", lo, hi) / ("choice", [a, b, c])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .blueprint import ComponentRef
from .catalog import components_for


@dataclass
class FamilySpec:
    family: str
    protocol: str
    name_zh: str
    default_algo_id: str
    import_path: str                               # "autoalgo.components.alpha:RegimeGatedAlpha"
    build_kind: str                                # class / exec_factory / router / selector
    default_params: Dict[str, Any]
    search_space: Dict[str, Tuple]
    hypothesis: str
    mechanism: List[str]
    assumptions: List[str]
    risks: List[str]
    param_doc: Dict[str, str]
    objective: str
    higher_is_better: bool = True
    baseline_names: List[str] = field(default_factory=list)
    formula: List[str] = field(default_factory=list)      # 关键公式（纯文本 / LaTeX 片段）
    components: List[ComponentRef] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.components:
            self.components = components_for(self.family)

    @property
    def cls(self):
        mod, name = self.import_path.split(":")
        import importlib

        return getattr(importlib.import_module(mod), name)

    def build(self, params: Optional[Dict[str, Any]] = None, **ctx):
        """实例化：class → cls(**params)；exec_factory → AutoExecutionFactory(params=params, base=ctx.get('base'))；
        router → cls(ctx['base'], ctx['venues'], **params)；selector → cls(**params)。"""
        p = {**self.default_params, **(params or {})}
        cls = self.cls
        if self.build_kind == "exec_factory":
            return cls(base=ctx.get("base"), params=p, **ctx.get("factory_cfg", {}))
        if self.build_kind == "router":
            return cls(ctx["base"], ctx["venues"], **p)
        return cls(**p)


FAMILIES: Dict[str, FamilySpec] = {}


def _reg(spec: FamilySpec) -> FamilySpec:
    FAMILIES[spec.family] = spec
    return spec


# ============================================================================ 因子层
_reg(FamilySpec(
    family="alpha.regime_gated", protocol="AlphaModel", name_zh="市场状态门控多因子模型", default_algo_id="regime_gated_alpha",
    import_path="autoalgo.components.alpha:RegimeGatedAlpha", build_kind="class",
    default_params={"hurst_window": 120, "lo": 0.45, "hi": 0.55, "hazard": 1 / 120, "method": "max_ir", "shrink_on_change": 0.5, "min_regime_obs": 40},
    search_space={"hurst_window": ("choice", [60, 90, 120, 180]), "lo": ("float", 0.40, 0.49), "hi": ("float", 0.51, 0.60), "hazard": ("logfloat", 1 / 400, 1 / 40),
                  "method": ("choice", ["max_ir", "icir", "ridge", "equal"]), "shrink_on_change": ("float", 0.0, 0.9), "min_regime_obs": ("choice", [30, 40, 60])},
    hypothesis="同一组因子在不同市场状态下的有效性不同：均值回复状态（Hurst<0.5）里反转、量价背离因子更强，趋势状态（Hurst>0.5）里动量更强。"
               "用物理学中的 DFA-Hurst 指数识别状态、分状态估计合成权重，并在变点刚发生（BOCPD 概率高、状态标签不可靠）时收缩置信度。",
    mechanism=["用 factormine 求值 K 个因子表达式，并计算未来 h 期收益。",
               "等权市场收益 → 滚动 DFA Hurst（crossquant）→ 按阈值 lo/hi 标注 均值回复 / 中性 / 趋势。",
               "每个状态子样本上用 factormine.fit_weights(method) 估合成权重；样本不足回退全局权重。",
               "预测时按当前状态取权重合成分数 z，期望收益 E[r]=IC_state·σ_idio·z（Grinold）。",
               "BOCPD 给出'最近发生变点'的概率 p，置信度乘 (1−shrink·p)。"],
    assumptions=["市场状态可由 Hurst 指数在 60–180 bar 窗口内识别，且状态持续时间长于窗口更新步长。",
                 "各状态内因子权重稳定，训练期每个状态有足够样本（min_regime_obs）。",
                 "面板为日频、横截面 ≥ 20 个资产，字段齐全（open/high/low/close/volume/returns）。"],
    risks=["状态识别滞后：Hurst 估计需要较长窗口，状态切换初期会用错权重（变点收缩只能部分缓解）。",
           "分状态估计样本更少，权重估计噪声更大；过细的阈值（lo/hi 太靠近 0.5）会放大噪声。",
           "合成数据上的改善未必迁移到真实市场；应在真实面板上重新运行生成管线复核。"],
    param_doc={"hurst_window": "滚动 Hurst 窗口（bar）", "lo": "均值回复阈值（H<lo）", "hi": "趋势阈值（H>hi）", "hazard": "BOCPD 先验变点概率",
               "method": "合成方法", "shrink_on_change": "变点概率对置信度的收缩强度", "min_regime_obs": "状态样本下限"},
    objective="rank_ic", baseline_names=["reference.ReversalVolumeAlpha"],
    formula=["H = DFA 斜率：F(s) ∝ s^H", "w_state = argmax IC_IR（Qian–Hua）: w ∝ Σ_IC⁻¹ μ_IC", "E[r_i] = IC_state · σ_i · z_i", "conf_i = clip(|IC|/0.1)·(1 − shrink·p_change)·min(|z_i|/2, 1)"],
))

_reg(FamilySpec(
    family="alpha.signature", protocol="AlphaModel", name_zh="路径签名因子模型", default_algo_id="signature_alpha",
    import_path="autoalgo.components.alpha:SignatureAlpha", build_kind="class",
    default_params={"window": 20, "ridge": 1.0, "channels": ["logprice", "logvolume"], "normalize": True},
    search_space={"window": ("choice", [10, 15, 20, 30, 40]), "ridge": ("logfloat", 0.1, 30.0), "channels": ("choice", [["logprice"], ["logprice", "logvolume"]]), "normalize": ("choice", [True, False])},
    hypothesis="粗路径理论中的截断签名是路径的'万能特征'：一阶项是净变化（动量/反转），二阶项（含 Lévy 面积）刻画价量之间谁领先、路径是凸还是凹。"
               "把每个资产近 window 根 bar 的 [时间, log 价格, log 成交量] 路径签名作为特征，截面标准化后与未来收益做池化岭回归。",
    mechanism=["构造路径 X_t = [t/(w−1), log close, log(1+volume)]，每通道增量按窗口标准差归一（只保留形状）。",
               "按 Chen 恒等式向量化计算深度-2 签名：S¹ = ΣΔ，S² = Σ (Σ_{i<k}Δ_i)⊗Δ_k + ½ΣΔ_k⊗Δ_k（与 crossquant.signature 等价）。",
               "特征逐日截面 z 标准化、裁剪到 ±5；标签为截面标准化的未来收益。",
               "池化岭回归 β = (XᵀX + λnI)⁻¹Xᵀy；预测分数 z = cs_z(X_T β)。"],
    assumptions=["价量路径形状与未来短期收益之间存在线性可分的关系，且在训练期与预测期稳定。",
                 "窗口内 bar 数 ≥ 10；成交量为正。"],
    risks=["二阶签名特征之间高度相关，λ 太小会过拟合；λ 太大退化为一阶动量/反转。",
           "特征对极端成交量敏感（已用 log1p 与裁剪缓解）。", "线性读出忽略了签名的高阶交互；深度 >2 会使特征维数指数增长。"],
    param_doc={"window": "路径窗口长度（bar）", "ridge": "岭惩罚（相对样本量 ×1e-3）", "channels": "路径通道", "normalize": "增量是否按窗口标准差归一"},
    objective="rank_ic", baseline_names=["reference.ReversalVolumeAlpha"],
    formula=["S²_{ij} = ∫∫_{s<t} dX_i(s) dX_j(t)", "Lévy 面积 A_{ij} = ½(S²_{ij} − S²_{ji})", "β = (XᵀX + λ n I)⁻¹ Xᵀ y"],
))

_reg(FamilySpec(
    family="alpha.cross_operator_mined", protocol="AlphaModel", name_zh="跨学科算子遗传规划因子", default_algo_id="cross_operator_mined_alpha",
    import_path="autoalgo.components.alpha:CrossOperatorMinedAlpha", build_kind="class",
    default_params={"budget": 40, "seed": 0, "method": "max_ir", "extended_only": True, "pool_capacity": 12, "max_corr": 0.7, "min_ic": 0.01},
    search_space={"budget": ("choice", [30, 40, 60]), "method": ("choice", ["max_ir", "icir", "ridge"]), "pool_capacity": ("choice", [8, 12, 16]), "max_corr": ("float", 0.5, 0.8), "min_ic": ("float", 0.005, 0.02)},
    hypothesis="因子挖掘的搜索空间由算子表决定。把物理/数学/控制中的 6 个工具（分数阶差分、方差比、Hurst、排列熵、Kalman 残差、Lévy 面积）"
               "改写成面板算子注入 factormine，遗传规划就能搜到传统算子表达不出的因子（如'价量 Lévy 面积为正且方差比显示均值回复'）。",
    mechanism=["components.operators.register_all 把扩展算子注册进 factormine.OPERATORS，并为常数参数提供候选值。",
               "factormine.mine_factors（GP：种群 max(20,budget)，代数 max(2,budget//20)）在训练段挖掘，样本外 RankIC 同号且 ≥ min_ic 才入库。",
               "factormine.AlphaModel(library, method) 估合成权重，signal 给出分数、Grinold 期望收益与 IC 半衰期。",
               "describe 报告因子库中含扩展算子的因子比例（跨学科贡献度）。"],
    assumptions=["训练段足够长（≥ 200 bar）让 GP 的样本外过滤有意义。", "扩展算子的窗口/常数候选覆盖了数据的时间尺度。"],
    risks=["GP 本质是大规模多重检验，样本外过滤只能部分抑制过拟合；因子库应定期用新数据重挖。",
           "ts_hurst / ts_fracdiff 计算较慢，budget 过大时挖掘时间显著增加。", "随机种子影响结果；生成文档记录了 seed 以便复现。"],
    param_doc={"budget": "计算预算（≈ 候选评估次数/10）", "seed": "随机种子", "method": "合成方法", "extended_only": "是否只用基础+扩展算子",
               "pool_capacity": "因子池容量", "max_corr": "池内因子最大相关", "min_ic": "入池最低 |RankIC|"},
    objective="rank_ic", baseline_names=["reference.ReversalVolumeAlpha"],
    formula=["FFD: x̃_t = Σ_k w_k x_{t−k}, w_k = −w_{k−1}(d−k+1)/k", "VR(k) = Var(Σ_{i<k} r_{t−i}) / (k·Var r)", "K_ss = (P/R)/(P/R+1), P/R = (q+√(q²+4q))/2"],
))

# ============================================================================ 组合层
_reg(FamilySpec(
    family="portfolio.denoised_riskparity", protocol="PortfolioConstructor", name_zh="随机矩阵去噪协方差的多空组合（均值-方差 / 层次风险平价）", default_algo_id="denoised_riskparity_portfolio",
    import_path="autoalgo.components.portfolio:DenoisedRiskParityConstructor", build_kind="class",
    default_params={"top_k": 10, "gross_target": 1.0, "max_weight": 0.08, "lookback": 120, "denoise": "rie", "allocator": "mv", "tilt": 0.5, "kelly_fraction": 0.0,
                    "cost_threshold_bps": 0.0, "neutralize": "dollar"},
    search_space={"top_k": ("choice", [8, 10, 15]), "lookback": ("choice", [60, 120, 180]), "denoise": ("choice", ["rie", "clip", "lw", "none"]), "allocator": ("choice", ["mv", "hrp", "ivp", "equal"]),
                  "tilt": ("float", 0.0, 1.0), "kelly_fraction": ("choice", [0.0, 0.25, 0.5]), "max_weight": ("choice", [0.06, 0.08, 0.12]),
                  "cost_threshold_bps": ("choice", [0.0, 1.0, 2.0]), "neutralize": ("choice", ["dollar", "beta"])},
    hypothesis="按信号强度加权的多空组合不看协方差：它把风险集中在高波动标的上，也不对冲共同因子暴露。样本协方差在 N/T 不小时被噪声特征值主导（Marchenko–Pastur），"
               "直接做均值-方差会放大噪声；用随机矩阵理论（RIE / 特征值裁剪）去噪后再做均值-方差（或层次风险平价 + 信号倾斜），"
               "应当在同等信号下获得更高的风险调整收益。注意：若 alpha 本身与个股波动率成正比（如 quantstack 合成面板的植入信号），分数加权已接近最优，可提升空间有限。",
    mechanism=["按分数取多头 top_k（score>0）与空头 top_k（score<0）；cost_threshold_bps>0 时剔除期望收益低于门槛的标的（默认 0：弱 alpha 下硬门槛会把组合压成一两只股票）。",
               "最近 lookback 期收益 → denoise_covariance(method) 得去噪协方差 Σ。",
               "allocator=mv：Σ̂ 的最大特征对给出共同因子载荷 β̂=√λ₁·v₁ 与特质方差 σ²_ε=diag(Σ̂)−β̂²；腿内 w_i ∝ μ_i/σ²_ε,i；"
               "neutralize=dollar 按 net_target 分配两腿；neutralize=beta 用多空两腿规模抵消 Σw_iβ̂_i（每腿 35%–65%）；最后迭代裁剪单票上限。",
               "allocator=hrp/ivp/equal：每腿先求风险权重 w_risk，再 w ∝ w_risk^(1−tilt) · (|z|/mean|z|)^tilt，归一到腿目标敞口并裁剪。",
               "kelly_fraction>0 时用多资产 Kelly f*=Σ⁻¹μ 的分数杠杆限制总敞口。",
               "生成 TargetPosition：紧急度由置信度与 IC 半衰期决定（沿用参考实现规则），不在目标集合的持仓平仓。"],
    assumptions=["历史收益窗口 ≥ 30 期且覆盖全部候选标的；相关结构在窗口内近似平稳。", "信号分数与期望收益的方向一致。"],
    risks=["风险平价会把权重推向低波动资产，若信号主要在高波动资产上，tilt 过小会稀释 alpha；mv 对期望收益的尺度误差敏感（已用去噪 + 岭正则 + 单票上限约束）。",
           "HRP 的聚类顺序对相关矩阵的小扰动敏感；去噪方法 lw 在 T≫N 时与样本协方差差别很小。",
           "Kelly 杠杆依赖期望收益的尺度，信号 IC 高估时会过度加杠杆（已用 fraction 与上限约束）。"],
    param_doc={"top_k": "每腿最多标的数", "gross_target": "总敞口/NAV", "max_weight": "单票权重上限", "lookback": "协方差估计窗口", "denoise": "去噪方法",
               "allocator": "权重配置方法（mv/hrp/ivp/equal）", "tilt": "信号倾斜强度（仅 hrp/ivp/equal）", "kelly_fraction": "分数 Kelly 比例（0 关闭）", "cost_threshold_bps": "期望收益成本门槛（0 关闭）", "neutralize": "mv 的两腿规模规则：beta 中性 / dollar 中性"},
    objective="sharpe", baseline_names=["reference.LongShortConstructor"],
    formula=["λ₊ = σ²(1+√q)², q = N/T（MP 上界）", "mv: β̂ = √λ₁ v₁，σ²_ε = diag(Σ̂) − β̂²，w_i ∝ μ_i/σ²_ε,i（Grinold：μ_i = IC·σ_i·z_i 时即 w_i ∝ z_i/σ_i）；beta 中性时 g_L/g_S 使 Σ w_i β̂_i = 0", "HRP: w_cluster ∝ 1/Var(cluster)，自顶向下二分", "w_i ∝ w_risk,i^(1−tilt) · s_i^tilt"],
))

# ============================================================================ 执行层
_reg(FamilySpec(
    family="execution.kalman_pid", protocol="ExecutionStrategy", name_zh="Kalman 公允价 + PID 反馈的 Almgren–Chriss 执行", default_algo_id="kalman_pid_execution",
    import_path="autoalgo.components.execution:AutoExecutionFactory", build_kind="exec_factory",
    default_params={"mode": "reversion", "controller": "pid", "kp": 0.5, "ki": 0.05, "ctrl_out_max": 0.2, "kf_obs_mult": 10.0, "fair_horizon_s": 30.0, "drift_threshold_bps": 1.0,
                    "drift_gain": 1.0, "improve_threshold_bps": 1.5, "backoff": 1.0},
    search_space={"mode": ("choice", ["reversion", "momentum"]), "controller": ("choice", ["pid", "lqr"]), "kp": ("float", 0.1, 1.5), "ki": ("logfloat", 0.002, 0.2),
                  "ctrl_out_max": ("choice", [0.1, 0.2, 0.3, 0.5]), "kf_obs_mult": ("choice", [1.0, 10.0, 50.0, 200.0]), "fair_horizon_s": ("choice", [10.0, 30.0, 60.0]),
                  "drift_threshold_bps": ("float", 0.3, 4.0), "drift_gain": ("float", 0.0, 2.0), "improve_threshold_bps": ("float", 0.5, 6.0), "backoff": ("choice", [0.0, 1.0])},
    hypothesis="Almgren–Chriss 轨迹是开环的：被动成交偏慢时会持续落后，最后不得不大量吃单；它也不看盘中价格相对公允价的位置。"
               "用 Kalman 局部线性趋势模型滤波中间价得到平滑公允价：reversion 模式把中间价对公允价的偏离视为暂时性（自身冲击、报价噪声），"
               "偏贵时放慢并后撤一跳等价格回来、便宜时加速并改善一跳；momentum 模式则按斜率外推追趋势。两种用法谁对由评估决定。"
               "再用 PID/LQR 把'计划−实际完成比例'的误差闭环反馈到计划量上，防止持续落后或超前。",
    mechanism=["继承 order_placement.QuantstackExecution，计划曲线取 AlmgrenChrissExec.plan（urgency→λ，alpha→漂移）。",
               "每个 on_market：Kalman 更新 (level, slope)。reversion：pred = −side·(mid − level)/mid；momentum：pred = side·(slope·h)/mid（bps，正 = 不利 / 应加速）。",
               "urgency_target = AC 目标 + u·Q（u = 控制器对误差 e=(目标−已成交)/Q 的输出）± drift_gain·k·切片（k=|pred|/阈值，上限 drift_max_slices）。",
               "passive_price：pred > improve_threshold → 近端改善一跳（不穿越对手价）；pred < −阈值且 backoff → 后撤一跳。",
               "追单 / 扫尾 / 撤单簿记全部沿用基类，满足 §8 全部约定。"],
    assumptions=["中间价对平滑公允价的偏离在 10–60 秒尺度上会部分回复（冲击衰减、报价噪声），或存在可外推的短期趋势。", "快照包含 adv / volatility，用于 AC 参数与 Kalman 噪声尺度。"],
    risks=["reversion 模式在真实趋势行情中会越贵越等、越便宜越追（被动等待的机会成本）；momentum 模式会把自身冲击当成趋势追涨杀跌。",
           "阈值过低会频繁改价增加撤改量；PID 积分项在长时间无成交时累积，ctrl_out_max 限幅防止突然大量吃单。",
           "模拟器只对主动成交施加冲击、被动成交近乎免费，因此更被动的配置在模拟器里天然占优；真实市场的逆向选择成本会改变最优参数。"],
    param_doc={"mode": "Kalman 信号用法：reversion / momentum", "controller": "pid / lqr", "kp": "比例增益", "ki": "积分增益（每秒）", "ctrl_out_max": "控制量上限（占母单比例）",
               "kf_obs_mult": "观测噪声倍数（越大公允价越平滑）", "fair_horizon_s": "momentum 外推时长（秒）", "drift_threshold_bps": "信号生效阈值（bps）",
               "drift_gain": "每倍阈值提前/推后的切片数（0 = 关闭节奏调节）", "improve_threshold_bps": "改善/后撤一跳的阈值（bps）", "backoff": "信号有利时是否后撤一跳"},
    objective="shortfall_bps", higher_is_better=False, baseline_names=["order_placement.almgren_chriss", "order_placement.twap"],
    formula=["x_{t+dt} = [[1, dt],[0, 1]] x_t + w，观测 mid = [1 0] x + v", "reversion: pred = −side·(mid − level)/mid·1e4；momentum: pred = side·slope·h/mid·1e4",
             "u_t = Kp e_t + Ki Σ e dt + Kd de/dt（限幅）", "target_t = AC(t) + u_t·Q + sign(pred)·g·min(|pred|/θ, 3)·slice"],
))

# ============================================================================ 执行规划
_reg(FamilySpec(
    family="selector.bandit", protocol="StrategySelector", name_zh="上下文老虎机执行策略选择器", default_algo_id="bandit_strategy_selector",
    import_path="autoalgo.components.selector:BanditStrategySelector", build_kind="selector",
    default_params={"arms": ["twap", "almgren_chriss", "pov", "adaptive_passive", "kalman_pid"], "alpha": 0.6, "ridge": 1.0, "reward_scale_bps": 20.0, "warmup": 2, "epsilon": 0.05},
    search_space={"alpha": ("float", 0.2, 1.5), "ridge": ("logfloat", 0.3, 5.0), "reward_scale_bps": ("choice", [10.0, 20.0, 40.0]), "warmup": ("choice", [1, 2, 4]), "epsilon": ("float", 0.0, 0.15)},
    hypothesis="没有一种执行策略在所有母单上最优：大单/紧急单适合前置轨迹，小单/低紧急度适合被动赚价差。参考规则是手写阈值，"
               "LinUCB 用母单特征作上下文、以实现偏差为奖励在线学习策略选择，长期应不劣于规则并随数据改进。",
    mechanism=["上下文 x = [1, log10(%ADV), urgency, tanh(alpha/10), 价差, 时长, 波动率, %ADV]。",
               "前 warmup 次与 ε 探索沿用 quantstack.reference.execution.choose_strategy，并用其结果训练对应臂（冷启动不劣于规则）。",
               "其余按 LinUCB 选臂：θ_a = A_a⁻¹ b_a，score = θ_aᵀx + α√(xᵀA_a⁻¹x)。",
               "attach(pipeline) 订阅 order_done：reward = −clip(IS_bps, ±3s)/s（未成交按最差计），更新 A_a += xxᵀ, b_a += r·x。"],
    assumptions=["同一流水线内母单特征与成本的关系平稳；成本以 TCA 实现偏差衡量。", "候选策略名存在于执行工厂 available() 中。"],
    risks=["奖励噪声极大（单路径 IS 噪声可达 20 bps），需要数十笔以上母单才能显著优于规则。",
           "臂集合变化（新策略加入）需重新学习；线性模型不能表达特征交互。"],
    param_doc={"arms": "候选策略", "alpha": "LinUCB 探索系数", "ridge": "岭惩罚", "reward_scale_bps": "奖励归一化尺度", "warmup": "冷启动次数", "epsilon": "ε-贪心探索概率"},
    objective="shortfall_bps", higher_is_better=False, baseline_names=["reference.choose_strategy"],
    formula=["a_t = argmax_a θ_aᵀx_t + α√(x_tᵀA_a⁻¹x_t)", "r = −clip(IS_bps, −3s, 3s)/s"],
))

# ============================================================================ 路由层
_reg(FamilySpec(
    family="router.bayesian_venue", protocol="Router", name_zh="Thompson 采样 + 变点监测的场所路由叠加层", default_algo_id="bayesian_venue_router",
    import_path="autoalgo.components.routing:BayesianVenueRouter", build_kind="router",
    default_params={"mix": 0.5, "obs_var_bps2": 4.0, "prior_var": 25.0, "min_share": 0.05, "max_share": 0.8, "latency_window": 40, "degrade_hazard": 0.05, "degrade_threshold": 0.5, "degrade_penalty": 0.6},
    search_space={"mix": ("float", 0.2, 0.9), "obs_var_bps2": ("logfloat", 1.0, 16.0), "prior_var": ("choice", [9.0, 25.0, 100.0]), "max_share": ("choice", [0.6, 0.8, 1.0]),
                  "degrade_threshold": ("float", 0.3, 0.8), "degrade_penalty": ("float", 0.3, 1.0), "latency_window": ("choice", [20, 40, 80])},
    hypothesis="延迟感知路由器按费用与延迟分位数做静态最优拆分，但场所的真实成交质量（滑点、隐藏流动性、拒单）只能事后观察，并会随时间漂移。"
               "把每个场所当成一条臂，用高斯 Thompson 采样在'探索冷门场所'与'利用便宜场所'间权衡；再用 BOCPD 监测延迟分布突变，退化场所降权或改投。",
    mechanism=["底层路由器（latency_routing 适配器）给出基准分配 q_base。",
               "吃单切片：对每个场所从后验 N(m_v, v_v) 采样，平移为正后归一得 w_TS；最终份额 = (1−mix)·q_base/Q + mix·w_TS，裁剪到 [min_share, max_share] 与 1.3× 可见量。",
               "成交回报：奖励 = −(side·(成交价−路由时中间价)/中间价 + 费用)·1e4 bps，更新对应场所后验；拒单计 −reject_penalty。",
               "每个场所保留最近 latency_window 个回报延迟，log 标准化后 BOCPD；变点概率 > 阈值且延迟上行 → 退化：吃单份额 ×(1−penalty)，被动单改投后验均值最高的正常场所。"],
    assumptions=["场所质量差异在同一会话内持续足够长，使得几十笔成交就能学到。", "Fill.latency_us 与 fee 由网关正确填写（§8）。"],
    risks=["Thompson 采样会在早期把量导向噪声中偶然'便宜'的场所；prior_var 与 obs_var 控制探索速度。",
           "份额重分配可能超过某场所可见量导致部分未成交（已限 1.3×），底层的同步到达计划在重分配后仍沿用。",
           "BOCPD 对延迟突刺敏感，latency_window 太短会误判。"],
    param_doc={"mix": "向 Thompson 权重倾斜的比例", "obs_var_bps2": "奖励观测方差", "prior_var": "先验方差", "min_share": "单场所最小份额", "max_share": "单场所最大份额",
               "latency_window": "延迟样本窗口", "degrade_hazard": "BOCPD 先验变点概率", "degrade_threshold": "退化判定阈值", "degrade_penalty": "退化场所份额惩罚"},
    objective="shortfall_bps", higher_is_better=False, baseline_names=["latency_routing.make_router"],
    formula=["m_v | data ~ N((Σr/σ²)/(1/τ²+n/σ²), 1/(1/τ²+n/σ²))", "share_v = (1−mix)·q_v/Q + mix·w_TS,v"],
))


def get(family: str) -> FamilySpec:
    if family not in FAMILIES:
        raise KeyError(f"未知模板族 {family!r}，可选：{sorted(FAMILIES)}")
    return FAMILIES[family]


def families_for(protocols=None) -> List[FamilySpec]:
    out = list(FAMILIES.values())
    if protocols:
        out = [f for f in out if f.protocol in set(protocols)]
    return out
