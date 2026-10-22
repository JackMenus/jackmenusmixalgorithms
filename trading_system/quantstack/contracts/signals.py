"""因子/信号契约：因子挖掘库 -> 组合层 的接口对象。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .serialization import Contract


@dataclass
class FactorSpec(Contract):
    """一个被挖掘出来的因子的描述（可复现）。

    ``expression`` 使用因子挖掘库的表达式语法（如 ``cs_rank(ts_mean(close, 5) / close)``），
    ``family`` 标明来源方法：gp（遗传规划）/ rl（强化学习）/ llm / dl（深度模型）/ handcrafted。
    """

    name: str
    expression: str
    family: str = "handcrafted"
    params: Dict[str, float] = field(default_factory=dict)
    source_library: str = "unknown"
    horizon: int = 1                       # 预测期（bar 数）
    created_at: str = ""
    notes: str = ""


@dataclass
class FactorReport(Contract):
    """因子评估报告（样本内/样本外均用此结构）。"""

    spec: FactorSpec
    ic_mean: float
    ic_ir: float
    rank_ic_mean: float = 0.0
    turnover: float = 0.0                  # 日均换手（0..1）
    long_short_sharpe: float = 0.0
    decay_half_life: float = 0.0           # IC 半衰期（bar）
    coverage: float = 1.0                  # 有效截面比例
    period_start: str = ""
    period_end: str = ""
    sample: str = "oos"                    # "is" / "oos"
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def is_tradeable(self) -> bool:
        """一个粗略的可交易门槛，agent 会在此之上再做判断。"""
        return abs(self.ic_mean) >= 0.02 and abs(self.ic_ir) >= 0.3 and self.coverage >= 0.8


@dataclass
class AlphaSignal(Contract):
    """单个标的在某时刻的 alpha 预测。这是因子层对下游的 *唯一* 输出。

    * ``expected_return_bps``：预测期内的期望超额收益（基点，正数看多）。
    * ``horizon_us``：预测有效期（微秒）。日频信号可填 1 个交易日 ≈ 6.5h*3600*1e6。
    * ``confidence``：0..1，模型对该预测的置信度（可来自集成分歧、历史 IC 等）。
    * ``score``：横截面标准化分数（z-score 或 rank 分数），用于组合权重。
    """

    symbol: str
    ts_us: int
    horizon_us: int
    expected_return_bps: float
    confidence: float = 0.5
    score: float = 0.0
    source: str = "unknown"
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def direction(self) -> int:
        return 1 if self.expected_return_bps > 0 else (-1 if self.expected_return_bps < 0 else 0)


@dataclass
class SignalBatch(Contract):
    """一次预测产出的全部信号，附带模型元信息，便于审计和复盘。"""

    ts_us: int
    signals: List[AlphaSignal]
    model_id: str = "unknown"
    universe_size: int = 0
    notes: str = ""

    def top(self, n: int, long: bool = True) -> List[AlphaSignal]:
        ordered = sorted(self.signals, key=lambda s: s.score, reverse=long)
        return ordered[:n]

    def by_symbol(self) -> Dict[str, AlphaSignal]:
        return {s.symbol: s for s in self.signals}
