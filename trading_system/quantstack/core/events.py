"""最小事件总线 + 审计日志。

所有层之间的交互都发布到总线上（signal / target / order / slice / route / fill / risk / agent），
日志（journal）是 agent 复盘与解释的原始材料，也是回放测试的依据。
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Deque, Dict, List, Optional

from ..contracts.serialization import _to_jsonable

Handler = Callable[["Event"], None]


@dataclass
class Event:
    kind: str
    ts_us: int
    payload: Any = None
    seq: int = 0
    source: str = ""

    def to_record(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "ts_us": self.ts_us,
            "source": self.source,
            "payload": _to_jsonable(self.payload),
        }


class EventBus:
    def __init__(self, journal_size: int = 100_000):
        self._handlers: DefaultDict[str, List[Handler]] = defaultdict(list)
        self._wildcard: List[Handler] = []
        self.journal: Deque[Event] = deque(maxlen=journal_size)
        self._seq = 0
        self.counts: DefaultDict[str, int] = defaultdict(int)

    def subscribe(self, kind: Optional[str], handler: Handler) -> None:
        """``kind=None`` 订阅全部事件。"""
        if kind is None:
            self._wildcard.append(handler)
        else:
            self._handlers[kind].append(handler)

    def publish(self, kind: str, payload: Any = None, ts_us: int = 0, source: str = "") -> Event:
        self._seq += 1
        ev = Event(kind=kind, ts_us=ts_us, payload=payload, seq=self._seq, source=source)
        self.journal.append(ev)
        self.counts[kind] += 1
        for h in self._handlers.get(kind, ()):
            h(ev)
        for h in self._wildcard:
            h(ev)
        return ev

    def tail(self, n: int = 50, kind: Optional[str] = None) -> List[Event]:
        items = [e for e in self.journal if kind is None or e.kind == kind]
        return items[-n:]

    def records(self, n: Optional[int] = None, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        evs = self.tail(n or len(self.journal), kind)
        return [e.to_record() for e in evs]
