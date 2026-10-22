"""做市策略基类：报价舍入、库存上限、alpha 偏移、延迟感知加宽。

子类只需实现 ``_offsets(mid, q, tau, sigma, mu)``，返回 (δ_bid, δ_ask)，
即相对“参考价”的买卖挂单距离（价格单位，可为负）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..interfaces import (
    AlphaSignal,
    BookSnapshot,
    ChildOrder,
    Fill,
    LatencyProfile,
    OrderType,
    Side,
    StrategyState,
)


@dataclass
class MarketMakerConfig:
    gamma: float = 0.1          # 风险厌恶系数 γ
    sigma: float = 0.02         # 价格波动率 σ（价格单位/√秒）
    A: float = 1.0              # 成交强度参数 A（次/秒）
    k: float = 50.0             # 成交强度衰减 k（1/价格单位）
    order_qty: float = 1.0      # 每次挂单数量（= 一个 lot Δ）
    max_inventory: float = 10.0  # 库存上限（单位：股）
    latency_z: float = 1.0      # 延迟加宽系数：δ += z·σ·√RTT
    alpha_weight: float = 1.0   # alpha 漂移的使用比例
    reference: str = "mid"      # "mid" 或 "microprice"
    post_only: bool = True
    extra: dict = field(default_factory=dict)


class BaseMarketMaker:
    name = "base_mm"

    def __init__(self, config: Optional[MarketMakerConfig] = None, microprice=None):
        self.cfg = config or MarketMakerConfig()
        self.microprice = microprice  # 可选：EmpiricalMicroprice 实例

    # --- 子类实现 ---
    def _offsets(self, q: float, tau: float, sigma: float, mu: float) -> Tuple[float, float]:
        raise NotImplementedError

    # --- 公共逻辑 ---
    def reference_price(self, book: BookSnapshot) -> float:
        if self.cfg.reference == "microprice" and self.microprice is not None:
            return self.microprice.price(book)
        return book.mid

    def quotes(
        self,
        book: BookSnapshot,
        state: StrategyState,
        alpha: Optional[AlphaSignal] = None,
        latency: Optional[LatencyProfile] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """返回 (bid_price, ask_price)；某侧为 None 表示不挂单。"""
        cfg = self.cfg
        s = self.reference_price(book)
        mu = alpha.drift_per_second(s) * cfg.alpha_weight if alpha else 0.0
        q_lots = state.inventory / cfg.order_qty
        db, da = self._offsets(q_lots, state.time_left, cfg.sigma, mu)
        if latency is not None:
            widen = cfg.latency_z * cfg.sigma * math.sqrt(latency.round_trip_s)
            db += widen
            da += widen
        bid = s - db
        ask = s + da
        tick = book.tick_size
        bid = math.floor(bid / tick + 1e-9) * tick
        ask = math.ceil(ask / tick - 1e-9) * tick
        if cfg.post_only:  # 不穿越对手价
            bid = min(bid, book.best_ask - tick)
            ask = max(ask, book.best_bid + tick)
        if ask <= bid:
            ask = bid + tick
        bid_out = None if state.inventory + cfg.order_qty > cfg.max_inventory + 1e-12 else round(bid, 10)
        ask_out = None if state.inventory - cfg.order_qty < -cfg.max_inventory - 1e-12 else round(ask, 10)
        return bid_out, ask_out

    def on_book(self, book, state, alpha=None, latency=None) -> List[ChildOrder]:
        bid, ask = self.quotes(book, state, alpha, latency)
        otype = OrderType.POST_ONLY if self.cfg.post_only else OrderType.LIMIT
        out = []
        if bid is not None:
            out.append(ChildOrder(book.symbol, Side.BUY, self.cfg.order_qty, bid, otype, book.ts, strategy=self.name))
        if ask is not None:
            out.append(ChildOrder(book.symbol, Side.SELL, self.cfg.order_qty, ask, otype, book.ts, strategy=self.name))
        return out

    def on_fill(self, fill: Fill, state: StrategyState) -> None:
        sgn = fill.side.sign
        state.inventory += sgn * fill.qty
        state.cash -= sgn * fill.qty * fill.price
