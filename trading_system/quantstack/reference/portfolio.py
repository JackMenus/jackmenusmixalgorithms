"""多空组合构造器（参考实现）：按分数排序取多空两腿，分数加权，带单票上限、总敞口目标与交易成本门槛。

跨学科库若提供更强的优化器（均值-方差 / 风险平价 / 鲁棒优化），实现 ``PortfolioConstructor`` 即可替换。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..contracts import MarketSnapshot, PortfolioSnapshot, RiskLimits, SignalBatch, TargetPosition


class LongShortConstructor:
    def __init__(
        self,
        top_k: int = 10,
        gross_target: float = 1.0,        # 总敞口 / NAV
        net_target: float = 0.0,          # 净敞口 / NAV
        max_weight: float = 0.08,
        cost_threshold_bps: float = 5.0,  # 期望 alpha 低于此成本估计不交易
        min_trade_weight: float = 0.001,
        default_participation: float = 0.1,
        lot_size: int = 1,
    ):
        self.top_k = top_k
        self.gross_target = gross_target
        self.net_target = net_target
        self.max_weight = max_weight
        self.cost_threshold_bps = cost_threshold_bps
        self.min_trade_weight = min_trade_weight
        self.default_participation = default_participation
        self.lot_size = lot_size

    def construct(self, signals: SignalBatch, portfolio: PortfolioSnapshot, market: Dict[str, MarketSnapshot], limits: RiskLimits) -> List[TargetPosition]:
        nav = portfolio.nav
        usable = [s for s in signals.signals if s.symbol in market and market[s.symbol].mid > 0 and abs(s.expected_return_bps) >= self.cost_threshold_bps]
        ranked = sorted(usable, key=lambda s: s.score, reverse=True)
        longs = [s for s in ranked if s.score > 0][: self.top_k]
        shorts = [s for s in ranked[::-1] if s.score < 0][: self.top_k] if limits.allow_short else []
        long_gross = self.gross_target * (1 + self.net_target) / 2 if shorts else self.gross_target * min(1.0, max(self.net_target, 0.0) + 0.5) * 2 / 2
        short_gross = self.gross_target * (1 - self.net_target) / 2 if shorts else 0.0
        if not shorts:
            long_gross = min(self.gross_target, 1.0)
        weights: Dict[str, float] = {}
        weights.update(_leg_weights(longs, long_gross, self.max_weight, +1))
        weights.update(_leg_weights(shorts, short_gross, self.max_weight, -1))
        sig_by_sym = signals.by_symbol()
        targets: List[TargetPosition] = []
        # 现有持仓若不在目标集合中 -> 平仓
        for sym, pos in portfolio.positions.items():
            if pos.quantity != 0 and sym not in weights and sym in market:
                targets.append(TargetPosition(symbol=sym, target_quantity=0, target_weight=0.0, urgency=0.4, max_participation=self.default_participation, reason="exit: 不再在目标集合"))
        for sym, w in weights.items():
            px = market[sym].mid
            qty = int(round(w * nav / px / self.lot_size)) * self.lot_size
            s = sig_by_sym[sym]
            cur_w = portfolio.quantity(sym) * px / nav if nav else 0.0
            if abs(w - cur_w) < self.min_trade_weight:
                continue
            # 紧急度：半衰期越短越急；置信度越高越急
            hl = float(s.metadata.get("half_life_bars", 3.0)) if s.metadata else 3.0
            urgency = float(np.clip(0.25 + 0.4 * s.confidence + 0.35 * min(1.0, 1.0 / max(hl, 0.5)), 0.05, 0.95))
            targets.append(
                TargetPosition(
                    symbol=sym,
                    target_quantity=qty,
                    target_weight=w,
                    urgency=urgency,
                    max_participation=min(self.default_participation, limits.max_participation),
                    benchmark="arrival",
                    alpha_bps=abs(s.expected_return_bps),
                    alpha_horizon_us=s.horizon_us,
                    reason=f"score={s.score:+.2f} er={s.expected_return_bps:+.1f}bps conf={s.confidence:.2f}",
                )
            )
        return targets


def _leg_weights(leg, gross: float, max_w: float, sign: int) -> Dict[str, float]:
    if not leg or gross <= 0:
        return {}
    raw = np.array([abs(s.score) for s in leg])
    raw = raw / raw.sum() * gross
    # 迭代裁剪单票上限并重新分配
    for _ in range(10):
        over = raw > max_w
        if not over.any():
            break
        excess = (raw[over] - max_w).sum()
        raw[over] = max_w
        under = ~over
        if under.any():
            raw[under] += excess * raw[under] / raw[under].sum()
        else:
            break
    return {s.symbol: float(sign * w) for s, w in zip(leg, raw)}
