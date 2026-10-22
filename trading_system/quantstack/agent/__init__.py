"""quantstack.agent —— LLM 编排层。

* ``model_policy``：任务分级与模型路由（强推理 -> Claude Fable 5.1 effort=max；其余 -> Opus 5.5 / Fable 5.1 自适应升级阶梯）
* ``decisions``：agent 结构化决策的 pydantic 模型（执行计划 / 研究评审 / 交易后复盘 / 事故响应 / 日报）
* ``backend``：ClaudeBackend（Anthropic SDK）与 ScriptedBackend（离线、确定性、用于测试与无凭据演示）
* ``tools``：暴露给模型的工具（只读查询 + 有边界的调参 + 熔断）
* ``roles``：各角色系统提示
* ``orchestrator``：TradingAgent，把流水线的四个决策点串起来
"""
from .model_policy import FABLE_51, OPUS_55, ComplexitySignals, ModelPolicy, ModelSpec, TaskClass
from .orchestrator import TradingAgent

__all__ = ["FABLE_51", "OPUS_55", "ComplexitySignals", "ModelPolicy", "ModelSpec", "TaskClass", "TradingAgent"]
