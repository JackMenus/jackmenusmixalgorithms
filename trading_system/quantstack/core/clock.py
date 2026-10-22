"""时钟抽象：同一套策略代码在回测（SimClock）与实盘（WallClock）下运行。"""
from __future__ import annotations

import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now_us(self) -> int: ...


class SimClock:
    """可手动推进的模拟时钟。"""

    def __init__(self, start_us: int = 0):
        self._now = int(start_us)

    def now_us(self) -> int:
        return self._now

    def advance(self, dt_us: int) -> int:
        if dt_us < 0:
            raise ValueError("时间不能倒流")
        self._now += int(dt_us)
        return self._now

    def set_time(self, ts_us: int) -> None:
        if ts_us < self._now:
            raise ValueError("时间不能倒流")
        self._now = int(ts_us)


class WallClock:
    def now_us(self) -> int:
        return time.time_ns() // 1_000
