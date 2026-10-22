"""agent 结构化决策（pydantic）。模型用 ``output_format`` 直接产出这些对象；离线后端产出同构对象。

所有决策都会再经确定性校验：策略名必须在 ``available()`` 中、参数必须是数值、风控参数必须在允许区间。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class OrderPlan(BaseModel):
    order_id: str
    strategy: str = Field(description="执行策略名，必须来自可用策略列表")
    params: Dict[str, float] = Field(default_factory=dict, description="策略参数（数值）")
    urgency: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="覆盖母单紧急度（可选）")
    rationale: str = ""


class ExecutionPlan(BaseModel):
    plans: List[OrderPlan]
    notes: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ParamChange(BaseModel):
    name: str
    value: float
    rationale: str = ""


class ResearchDecision(BaseModel):
    deploy_factors: List[str] = Field(default_factory=list, description="建议部署的因子名")
    retire_factors: List[str] = Field(default_factory=list, description="建议下线的因子名")
    hypotheses: List[str] = Field(default_factory=list, description="下一轮挖掘的假设/方向")
    rationale: str = ""
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class PostTradeReview(BaseModel):
    summary: str
    anomalies: List[str] = Field(default_factory=list)
    param_changes: List[ParamChange] = Field(default_factory=list)
    strategy_feedback: Dict[str, str] = Field(default_factory=dict, description="策略名 -> 反馈")
    venue_feedback: Dict[str, str] = Field(default_factory=dict, description="场所 -> 反馈")
    escalate: bool = Field(default=False, description="是否需要升级为强推理复盘")
    escalation_reason: str = ""
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class IncidentResponse(BaseModel):
    diagnosis: str
    root_causes: List[str] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    resume_trading: bool = False
    param_changes: List[ParamChange] = Field(default_factory=list)
    follow_ups: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class DataQualityReport(BaseModel):
    ok: bool
    issues: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)


class DailyReport(BaseModel):
    headline: str
    performance: str
    execution: str
    routing: str
    risks: str
    next_steps: List[str] = Field(default_factory=list)
