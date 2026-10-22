"""OrderPlacementEngine：集成 agent 使用的统一门面。

职责：
1. 管理多个标的上的策略实例（做市 / 母单执行）；
2. 每次行情到达时，从 AlphaProvider 拉取最新 alpha、从 Router 获取延迟画像，
   调用策略生成子单并交给 Router 发送；
3. 接收成交回报并分发给对应策略。

因子库只需实现 ``AlphaProvider.latest``，路由库只需实现 ``Router.route`` 与
``Router.latency_profile``，即可与本库无缝对接。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .interfaces import (
    AlphaProvider,
    BookSnapshot,
    ChildOrder,
    Fill,
    OrderPlacementStrategy,
    Router,
    StrategyState,
)


class OrderPlacementEngine:
    def __init__(self, router: Optional[Router] = None, alpha_provider: Optional[AlphaProvider] = None):
        self.router = router
        self.alpha_provider = alpha_provider
        self.strategies: Dict[str, OrderPlacementStrategy] = {}
        self.states: Dict[str, StrategyState] = {}
        self.symbol_of: Dict[str, str] = {}
        self.order_owner: Dict[str, str] = {}
        self.log: List[ChildOrder] = []

    def add_strategy(self, key: str, symbol: str, strategy: OrderPlacementStrategy,
                     state: Optional[StrategyState] = None) -> None:
        self.strategies[key] = strategy
        self.states[key] = state or StrategyState()
        self.symbol_of[key] = symbol

    def on_book(self, book: BookSnapshot) -> List[ChildOrder]:
        alpha = self.alpha_provider.latest(book.symbol, book.ts) if self.alpha_provider else None
        latency = self.router.latency_profile() if self.router else None
        out: List[ChildOrder] = []
        for key, strat in self.strategies.items():
            if self.symbol_of[key] != book.symbol:
                continue
            st = self.states[key]
            st.t = book.ts
            orders = strat.on_book(book, st, alpha, latency)
            for o in orders:
                self.order_owner[o.order_id] = key
            out += orders
        if out and self.router is not None:
            self.router.route(out)
        self.log += out
        return out

    def on_fill(self, fill: Fill) -> None:
        key = self.order_owner.get(fill.order_id)
        if key is None:
            return
        self.strategies[key].on_fill(fill, self.states[key])
