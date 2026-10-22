"""单生产者单消费者（SPSC）无锁环形队列 —— 参考实现。

这是 LMAX Disruptor / rigtorp::SPSCQueue 设计的 Python 版本，主要用于：
1. 说明算法（与 ``cpp/spsc_ring.hpp`` 一一对应），方便在研究阶段验证逻辑；
2. 在 Python 回测/仿真里作为行情→策略→路由之间的有界队列。

Python 有 GIL，这里不追求真正的无锁性能；生产环境请使用 C++ 版本。

关键设计点（C++ 版同样适用）：
- 容量取 2 的幂，用 ``idx & mask`` 代替取模；
- 读写指针单调递增、永不回绕，满/空判断为 ``write - read == capacity`` / ``write == read``；
- 生产者缓存一份「上次看到的读指针」(cached_read)，只有在看似已满时才去读对方的原子变量，
  消费者同理，从而把跨核缓存行流量降到最低；
- 读写指针分别独占一个 cache line（C++ 中 alignas(64)）以避免伪共享。
"""
from __future__ import annotations

from typing import Any, Generic, List, Optional, TypeVar

T = TypeVar("T")


class SPSCRing(Generic[T]):
    __slots__ = ("_buf", "_mask", "_cap", "_write", "_read", "_cached_read", "_cached_write")

    def __init__(self, capacity: int):
        if capacity < 2 or capacity & (capacity - 1):
            raise ValueError("capacity 必须是 ≥2 的 2 的幂")
        self._cap = capacity
        self._mask = capacity - 1
        self._buf: List[Optional[T]] = [None] * capacity
        self._write = 0          # 仅生产者写
        self._read = 0           # 仅消费者写
        self._cached_read = 0    # 生产者侧缓存
        self._cached_write = 0   # 消费者侧缓存

    # ---- 生产者 ----
    def try_push(self, item: T) -> bool:
        w = self._write
        if w - self._cached_read == self._cap:
            self._cached_read = self._read        # C++: load(acquire)
            if w - self._cached_read == self._cap:
                return False
        self._buf[w & self._mask] = item
        self._write = w + 1                        # C++: store(release)
        return True

    # ---- 消费者 ----
    def try_pop(self) -> Optional[T]:
        r = self._read
        if r == self._cached_write:
            self._cached_write = self._write       # C++: load(acquire)
            if r == self._cached_write:
                return None
        item = self._buf[r & self._mask]
        self._buf[r & self._mask] = None
        self._read = r + 1                         # C++: store(release)
        return item

    def drain(self, max_items: int = 1 << 30) -> List[Any]:
        out = []
        while len(out) < max_items:
            x = self.try_pop()
            if x is None:
                break
            out.append(x)
        return out

    def __len__(self) -> int:
        return self._write - self._read

    @property
    def capacity(self) -> int:
        return self._cap
