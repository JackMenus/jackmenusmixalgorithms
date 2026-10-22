"""模型路由策略：哪个任务用哪个模型、多大思考强度。

用户要求：
  * **强推理阶段**（架构/策略设计、因子假设、研究评审、复盘、风险事故、市场状态判断）—— Claude Fable 5.1，``effort="max"``；
  * **其他内容**—— Opus 5.5 与 Fable 5.1 **思考强度自适应**：按任务复杂度打分选择起始档位，失败/低置信时沿阶梯升级。

API 要点（见 claude-api skill）：Fable 5.1 / Opus 5.5 的 thinking 始终开启（省略 ``thinking`` 即自适应），
深度只由 ``output_config.effort`` 控制（low/medium/high/xhigh/max）；Opus 5.5 默认 medium，必须显式设置；
两者都不支持强制 tool_choice，用提示词引导 + ``strict`` 工具；默认开启服务端 refusal fallback。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

FABLE_51 = "claude-fable-5-1"
OPUS_55 = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


class TaskClass(str, Enum):
    # ---- 强推理（Fable 5.1, effort=max）
    ARCHITECTURE = "architecture"              # 系统/策略架构决策
    STRATEGY_DESIGN = "strategy_design"        # 新策略设计
    FACTOR_HYPOTHESIS = "factor_hypothesis"    # 因子假设生成（驱动挖掘）
    RESEARCH_REVIEW = "research_review"        # 决定部署/下线哪些因子
    POST_MORTEM = "post_mortem"                # 深度复盘
    RISK_INCIDENT = "risk_incident"            # 风控事故响应
    REGIME_ASSESSMENT = "regime_assessment"    # 市场状态/制度判断
    # ---- 自适应（Opus 5.5 起步，必要时升级到 Fable 5.1）
    DATA_QUALITY = "data_quality"
    EXECUTION_PLAN = "execution_plan"          # 为每张母单选执行策略与参数
    ROUTING_DIAGNOSTICS = "routing_diagnostics"
    POST_TRADE_REVIEW = "post_trade_review"    # 常规交易后复盘
    PARAM_TUNING = "param_tuning"
    DAILY_REPORT = "daily_report"
    LOG_SUMMARY = "log_summary"
    CODE_ROUTINE = "code_routine"


STRONG_TASKS = frozenset({
    TaskClass.ARCHITECTURE, TaskClass.STRATEGY_DESIGN, TaskClass.FACTOR_HYPOTHESIS, TaskClass.RESEARCH_REVIEW,
    TaskClass.POST_MORTEM, TaskClass.RISK_INCIDENT, TaskClass.REGIME_ASSESSMENT,
})


@dataclass(frozen=True)
class ModelSpec:
    model: str
    effort: str
    max_tokens: int = 16000
    tier: str = "adaptive"            # strong / adaptive
    thinking_display: str = "omitted"  # omitted / summarized
    stream: bool = False
    timeout_s: float = 600.0

    def request_kwargs(self) -> Dict[str, object]:
        """转换为 Anthropic SDK 的请求参数（不含 messages/system/tools）。"""
        kw: Dict[str, object] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
        }
        if self.thinking_display != "omitted":
            kw["thinking"] = {"type": "adaptive", "display": self.thinking_display}
        return kw

    @property
    def label(self) -> str:
        return f"{self.model}@{self.effort}"


@dataclass
class ComplexitySignals:
    """自适应档位的输入信号。全部可选；缺省即为低复杂度。"""

    input_chars: int = 0          # 上下文规模
    n_entities: int = 0           # 涉及的订单/因子/场所数量
    stakes: float = 0.0           # 0..1：资金/风险权重（如本次名义金额 / NAV）
    novelty: float = 0.0          # 0..1：与历史情形的差异（新标的、新市场状态）
    anomaly: float = 0.0          # 0..1：异常程度（TCA 异常、拒单率）
    prior_failures: int = 0       # 之前尝试失败/低置信次数

    def score(self) -> float:
        size = 1 - math.exp(-self.input_chars / 20_000)             # 20k 字符 ~ 0.63
        ents = 1 - math.exp(-self.n_entities / 15)
        s = 0.10 * size + 0.15 * ents + 0.30 * self.stakes + 0.20 * self.novelty + 0.25 * self.anomaly
        return float(min(max(s, 0.0), 1.0))


@dataclass
class ModelPolicy:
    strong: ModelSpec = field(default_factory=lambda: ModelSpec(FABLE_51, "max", max_tokens=64000, tier="strong", stream=True, timeout_s=1800.0))
    # 自适应阶梯：复杂度越高 / 重试越多，档位越靠后
    ladder: Tuple[ModelSpec, ...] = (
        ModelSpec(OPUS_55, "low", 8000),
        ModelSpec(OPUS_55, "medium", 16000),
        ModelSpec(OPUS_55, "high", 16000),
        ModelSpec(FABLE_51, "high", 32000, stream=True, timeout_s=1200.0),
        ModelSpec(FABLE_51, "xhigh", 64000, stream=True, timeout_s=1800.0),
    )
    thresholds: Tuple[float, float, float] = (0.25, 0.50, 0.65)
    overrides: Dict[TaskClass, ModelSpec] = field(default_factory=dict)

    def is_strong(self, task: TaskClass) -> bool:
        return task in STRONG_TASKS

    def base_index(self, signals: Optional[ComplexitySignals]) -> int:
        c = signals.score() if signals else 0.0
        lo, mid, hi = self.thresholds
        if c < lo:
            return 0
        if c < mid:
            return 1
        if c < hi:
            return 2
        return 3

    def resolve(self, task: TaskClass, signals: Optional[ComplexitySignals] = None, attempt: int = 0) -> ModelSpec:
        if task in self.overrides:
            return self.overrides[task]
        if self.is_strong(task):
            return self.strong
        idx = min(self.base_index(signals) + max(attempt, 0) + (signals.prior_failures if signals else 0), len(self.ladder) - 1)
        return self.ladder[idx]

    def table(self) -> List[Dict[str, str]]:
        rows = []
        for t in TaskClass:
            spec = self.resolve(t)
            rows.append({"task": t.value, "tier": "强推理" if self.is_strong(t) else "自适应", "start": spec.label, "max": self.strong.label if self.is_strong(t) else self.ladder[-1].label})
        return rows
