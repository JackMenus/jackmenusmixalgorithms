"""组件目录：autoalgo 可用来拼装新算法的全部"积木"，按来源库登记。

每条记录给出：导入路径、所属协议层、一句话作用、以及它在哪些模板族中被用到。目录同时服务三处：
  1. 解释文档的"组件来源"表；
  2. LLM 提议器的提示词（让模型知道有哪些积木可用）；
  3. ``autoalgo.cli list-components``。
"""
from __future__ import annotations

from typing import Dict, List

from .blueprint import ComponentRef

# library, path, layer, role, used_by
_CATALOG: List[Dict[str, object]] = [
    # ---- factormine 因子挖掘库
    {"library": "factormine", "path": "factormine.Evaluator / factormine.parse", "layer": "AlphaModel", "role": "因子表达式解析与带缓存求值", "used_by": ["alpha.regime_gated", "alpha.cross_operator_mined"]},
    {"library": "factormine", "path": "factormine.fit_weights / combine", "layer": "AlphaModel", "role": "多因子合成权重（equal/ic/icir/max_ir/ridge）与标准化合成", "used_by": ["alpha.regime_gated", "alpha.cross_operator_mined"]},
    {"library": "factormine", "path": "factormine.ic_series / evaluate_factor / forward_returns", "layer": "评估", "role": "RankIC、ICIR、换手等因子评估", "used_by": ["evaluate.alpha"]},
    {"library": "factormine", "path": "factormine.register", "layer": "AlphaModel", "role": "算子注册入口（扩展算子注入点）", "used_by": ["alpha.cross_operator_mined"]},
    {"library": "factormine", "path": "factormine.mine_factors / GPConfig / MiningConfig", "layer": "FactorMiner", "role": "遗传规划因子挖掘（样本外过滤入库）", "used_by": ["alpha.cross_operator_mined"]},
    {"library": "factormine", "path": "factormine.AlphaModel", "layer": "AlphaModel", "role": "因子库 → 合成信号（Grinold 期望收益、IC 半衰期）", "used_by": ["alpha.cross_operator_mined"]},
    # ---- crossquant 跨学科库
    {"library": "crossquant", "path": "crossquant.physics.hurst.dfa", "layer": "状态探测", "role": "DFA Hurst 指数：趋势 / 均值回复强度", "used_by": ["alpha.regime_gated", "operators.ts_hurst"]},
    {"library": "crossquant", "path": "crossquant.control.changepoint.bocpd", "layer": "状态探测", "role": "贝叶斯在线变点概率", "used_by": ["alpha.regime_gated", "router.bayesian_venue"]},
    {"library": "crossquant", "path": "crossquant.mathematics.signatures.signature / add_time", "layer": "AlphaModel", "role": "路径签名（深度 2）特征，Lévy 面积", "used_by": ["alpha.signature", "operators.ts_levy_area"]},
    {"library": "crossquant", "path": "crossquant.mathematics.fracdiff.frac_diff_weights", "layer": "AlphaModel", "role": "分数阶差分权重（长记忆平稳化）", "used_by": ["operators.ts_fracdiff"]},
    {"library": "crossquant", "path": "crossquant.physics.entropy.permutation_entropy", "layer": "AlphaModel", "role": "排列熵（序数模式复杂度）", "used_by": ["operators.ts_perm_entropy"]},
    {"library": "crossquant", "path": "crossquant.control.kalman.local_level_filter", "layer": "AlphaModel", "role": "局部水平 Kalman 滤波（稳态增益形式）", "used_by": ["operators.ts_kalman_resid"]},
    {"library": "crossquant", "path": "crossquant.physics.random_matrix.denoise_covariance", "layer": "PortfolioConstructor", "role": "随机矩阵理论（MP 裁剪 / RIE / Ledoit–Wolf）协方差去噪", "used_by": ["portfolio.denoised_riskparity"]},
    {"library": "crossquant", "path": "crossquant.physics.networks.hierarchical_risk_parity", "layer": "PortfolioConstructor", "role": "层次风险平价权重", "used_by": ["portfolio.denoised_riskparity"]},
    {"library": "crossquant", "path": "crossquant.mathematics.kelly.kelly_multi_asset", "layer": "PortfolioConstructor", "role": "多资产 Kelly 杠杆", "used_by": ["portfolio.denoised_riskparity"]},
    {"library": "crossquant", "path": "crossquant.KalmanTrendEstimator", "layer": "StateEstimator", "role": "局部线性趋势 Kalman：公允价滤波与外推", "used_by": ["execution.kalman_pid"]},
    {"library": "crossquant", "path": "crossquant.PIDParticipationController / LQRTrackingController", "layer": "Controller", "role": "执行进度反馈控制", "used_by": ["execution.kalman_pid"]},
    {"library": "crossquant", "path": "crossquant.control.bandits.LinUCB", "layer": "StrategySelector", "role": "上下文线性老虎机", "used_by": ["selector.bandit"]},
    {"library": "crossquant", "path": "crossquant.control.bandits.ThompsonGaussian", "layer": "Router", "role": "高斯 Thompson 采样（场所选择）", "used_by": ["router.bayesian_venue"]},
    # ---- order_placement 挂单策略库
    {"library": "order_placement", "path": "order_placement.adapters.quantstack_exec.QuantstackExecution", "layer": "ExecutionStrategy", "role": "执行策略基类：计划曲线、追单/被动挂单、在途簿记、§8 撤单约定", "used_by": ["execution.kalman_pid"]},
    {"library": "order_placement", "path": "order_placement.adapters.quantstack_exec.AlmgrenChrissExec.plan", "layer": "ExecutionStrategy", "role": "Almgren–Chriss 最优轨迹（urgency→λ，alpha→漂移）", "used_by": ["execution.kalman_pid"]},
    {"library": "order_placement", "path": "order_placement.queue.models.queue_value", "layer": "ExecutionStrategy", "role": "排队价值判断（挂近端还是改善一跳）", "used_by": ["execution.kalman_pid"]},
    {"library": "order_placement", "path": "order_placement.make_execution_factory", "layer": "ExecutionStrategyFactory", "role": "8 种基础执行策略工厂（被 AutoExecutionFactory 包裹）", "used_by": ["execution.kalman_pid", "selector.bandit"]},
    # ---- latency_routing 路由延迟库
    {"library": "latency_routing", "path": "latency_routing.make_router（经 quantstack.adapters.latency_routing_adapter）", "layer": "Router", "role": "延迟感知 SOR：扫单 + Cont-Kukanov 拆分 + 同步到达", "used_by": ["router.bayesian_venue"]},
    # ---- quantstack 集成层
    {"library": "quantstack", "path": "quantstack.contracts.*", "layer": "全部", "role": "跨库数据契约与协议（SignalBatch / TargetPosition / OrderSlice / RoutingDecision …）", "used_by": ["全部"]},
    {"library": "quantstack", "path": "quantstack.reference.portfolio.LongShortConstructor", "layer": "PortfolioConstructor", "role": "多空选股与单票上限规则（基线）", "used_by": ["portfolio.denoised_riskparity"]},
    {"library": "quantstack", "path": "quantstack.reference.execution.choose_strategy", "layer": "StrategySelector", "role": "确定性执行策略选择规则（冷启动与基线）", "used_by": ["selector.bandit"]},
    {"library": "quantstack", "path": "quantstack.core.registry.build_sim_pipeline", "layer": "评估", "role": "合成多场所市场 + 撮合模拟器 + TCA（执行/路由/选择器评估台）", "used_by": ["evaluate.execution", "evaluate.router", "evaluate.selector"]},
    {"library": "quantstack", "path": "quantstack.core.tca.TCAEngine", "layer": "评估", "role": "实现偏差、成交率、maker 比例", "used_by": ["evaluate.execution", "selector.bandit"]},
]


def catalog() -> List[Dict[str, object]]:
    return [dict(c) for c in _CATALOG]


def components_for(family: str) -> List[ComponentRef]:
    out = []
    for c in _CATALOG:
        used = c["used_by"]
        if family in used or any(str(u).startswith(family) for u in used):
            out.append(ComponentRef(name=str(c["path"]).split(" ")[0].split(".")[-1], library=str(c["library"]), path=str(c["path"]), role=str(c["role"])))
    return out


def describe_catalog(library: str | None = None) -> str:
    lines = []
    for c in _CATALOG:
        if library and c["library"] != library:
            continue
        lines.append(f"- [{c['library']}] {c['path']}  ({c['layer']}) —— {c['role']}")
    return "\n".join(lines)
