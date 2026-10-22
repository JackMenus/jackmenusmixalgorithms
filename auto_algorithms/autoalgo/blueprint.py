"""算法蓝图（Blueprint）与评估报告：生成管线中流转的两个核心数据结构。

* ``AlgorithmBlueprint``：一份可复现的“算法设计说明书”——它实现哪个协议、由哪些兄弟库组件拼装、参数是什么、
  设计假设 / 机制 / 失效条件是什么、以及它是怎么被生成出来的（血缘）。代码生成器据此渲染 ``algorithm.py``，
  解释文档生成器据此渲染中文 ``.md``。
* ``EvaluationReport``：评估台对蓝图的量化结论（指标、基线、是否通过），同样写进 ``.md`` 与 ``evaluation.json``。
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

GENERATOR_VERSION = "0.1.0"

PROTOCOLS = (
    "AlphaModel",            # 因子层：Panel -> SignalBatch
    "FactorMiner",           # 因子层：Panel -> [FactorSpec]
    "PortfolioConstructor",  # 组合层：SignalBatch -> [TargetPosition]
    "ExecutionStrategy",     # 执行层：母单 -> 子单切片
    "StrategySelector",      # 执行规划：母单特征 -> 执行策略名与参数
    "Router",                # 路由层：子单 -> 场所订单
    "StateEstimator",        # 钩子：隐状态估计
    "Controller",            # 钩子：进度控制
    "ImpactModel",           # 钩子：成本模型
)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ComponentRef:
    """算法用到的一个外部组件。"""

    name: str              # 组件名，如 KalmanTrendEstimator
    library: str           # 来源库：factormine / order_placement / latency_routing / crossquant / quantstack / autoalgo
    path: str              # 导入路径，如 crossquant.adapters.KalmanTrendEstimator
    role: str = ""         # 在本算法中的作用（中文）

    def to_dict(self) -> Dict[str, str]:
        return dataclasses.asdict(self)


@dataclass
class AlgorithmBlueprint:
    algo_id: str
    name_zh: str
    protocol: str
    family: str                                    # 模板族，如 alpha.regime_gated
    params: Dict[str, Any] = field(default_factory=dict)
    components: List[ComponentRef] = field(default_factory=list)
    hypothesis: str = ""                           # 设计动机与假设
    mechanism: List[str] = field(default_factory=list)   # 算法流程（分步）
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    param_doc: Dict[str, str] = field(default_factory=dict)        # 参数含义
    search_space: Dict[str, Any] = field(default_factory=dict)     # 参数搜索范围（供文档与再生成）
    lineage: Dict[str, Any] = field(default_factory=dict)          # 生成来源：generator / parent / mutation / seed / model
    registry_kind: str = ""                        # 在 quantstack 注册表里的 kind 名，如 autoalgo.regime_gated_alpha
    created_at: str = field(default_factory=_now_iso)
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"未知协议 {self.protocol!r}，可选：{PROTOCOLS}")
        if not self.registry_kind:
            self.registry_kind = f"autoalgo.{self.algo_id}"

    @property
    def libraries(self) -> List[str]:
        seen: List[str] = []
        for c in self.components:
            if c.library not in seen:
                seen.append(c.library)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlgorithmBlueprint":
        d = dict(d)
        d["components"] = [ComponentRef(**c) for c in d.get("components", [])]
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class EvaluationReport:
    algo_id: str
    protocol: str
    metrics: Dict[str, float] = field(default_factory=dict)              # 本算法指标
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)  # 基线名 -> 指标
    settings: Dict[str, Any] = field(default_factory=dict)               # 评估设置（数据规模、种子、窗口…）
    objective: str = ""                                                  # 主目标指标名
    higher_is_better: bool = True
    passed: bool = False
    verdict: str = ""                                                    # 中文结论
    notes: List[str] = field(default_factory=list)
    wall_s: float = 0.0
    created_at: str = field(default_factory=_now_iso)

    def primary(self) -> Optional[float]:
        return self.metrics.get(self.objective)

    def baseline_primary(self, name: Optional[str] = None) -> Optional[float]:
        if not self.baselines:
            return None
        key = name or next(iter(self.baselines))
        return self.baselines.get(key, {}).get(self.objective)

    def improvement(self) -> Optional[float]:
        """相对第一个基线在主目标上的改善（已按方向调整：正数 = 更好）。"""
        a, b = self.primary(), self.baseline_primary()
        if a is None or b is None:
            return None
        return (a - b) if self.higher_is_better else (b - a)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationReport":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=float)
