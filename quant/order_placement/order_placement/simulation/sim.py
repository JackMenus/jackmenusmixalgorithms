"""轻量级仿真器：用于策略对比、参数校验与 RL 训练（非撮合级精度）。

- synthetic_books：生成带失衡→价格预测性的合成 L2 序列（用于微价格/OFI 测试）。
- MarketMakingSim：布朗中间价 + 泊松成交 λ(δ)=A·e^{−kδ}，支持 alpha 漂移与延迟。
- ExecutionSim：布朗中间价 + 线性永久/临时冲击 + 排队成交，评估执行缺口 (IS)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..interfaces import (
    AlphaSignal,
    BookSnapshot,
    Fill,
    LatencyProfile,
    OrderType,
    Side,
    StrategyState,
)


def make_book(symbol: str, ts: float, mid: float, tick: float, bid_q: float, ask_q: float,
              spread_ticks: int = 1, depth: int = 5, level_q: float = 10.0) -> BookSnapshot:
    half = spread_ticks * tick / 2
    bb = round((mid - half) / tick) * tick
    ba = bb + spread_ticks * tick
    bids = tuple((round(bb - i * tick, 10), bid_q if i == 0 else level_q) for i in range(depth))
    asks = tuple((round(ba + i * tick, 10), ask_q if i == 0 else level_q) for i in range(depth))
    return BookSnapshot(symbol, ts, bids, asks, tick)


def synthetic_books(n: int, tick: float = 0.01, seed: int = 0, symbol: str = "SIM") -> List[BookSnapshot]:
    """失衡度越高，下一跳越可能向上（与 prob_up_diffusion 同向），用于验证特征有效性。"""
    rng = np.random.default_rng(seed)
    mid = 100.0
    books = []
    bq, aq = 10.0, 10.0
    for i in range(n):
        books.append(make_book(symbol, float(i), mid, tick, bq, aq))
        bq = max(1.0, bq + rng.normal(0, 2))
        aq = max(1.0, aq + rng.normal(0, 2))
        p_up = 1 - 2 / math.pi * math.atan(aq / bq)
        if rng.random() < 0.3:  # 价格变动事件
            if rng.random() < p_up:
                mid += tick
            else:
                mid -= tick
            bq, aq = rng.uniform(5, 15), rng.uniform(5, 15)
    return books


@dataclass
class MMResult:
    pnl: np.ndarray
    inventory: np.ndarray
    n_fills: int
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class MarketMakingSim:
    s0: float = 100.0
    sigma: float = 0.02      # 价格单位/√秒
    A: float = 1.0
    k: float = 50.0
    dt: float = 0.1
    T: float = 600.0
    tick: float = 0.01
    alpha_kappa: float = 0.0  # 漂移 μ_t 服从 OU：dμ = −κμdt + ηdW
    alpha_eta: float = 0.0
    alpha_noise: float = 0.0  # 观测噪声（相对 μ 的标准差）
    latency: Optional[LatencyProfile] = None
    fee_maker: float = 0.0     # 每股，负数为返佣
    seed: int = 0

    def run(self, strategy) -> MMResult:
        rng = np.random.default_rng(self.seed)
        n = int(self.T / self.dt)
        s = self.s0
        mu = 0.0
        state = StrategyState(inventory=0.0, cash=0.0, t=0.0, horizon_end=self.T)
        lag = 0 if self.latency is None else int(math.ceil(self.latency.round_trip_s / 2 / self.dt))
        pending: List = [(None, None)] * lag
        pnl = np.zeros(n)
        inv = np.zeros(n)
        fills = 0
        for i in range(n):
            t = i * self.dt
            state.t = t
            book = make_book("SIM", t, s, self.tick, 10, 10)
            alpha = None
            if self.alpha_eta > 0:
                obs = mu * (1 + self.alpha_noise * rng.standard_normal())
                horizon = 1.0
                alpha = AlphaSignal("SIM", t, value=obs * horizon / s * 1e4, horizon_s=horizon)
            quotes = strategy.quotes(book, state, alpha, self.latency)
            pending.append(quotes)
            bid, ask = pending.pop(0)
            for px, side in ((bid, Side.BUY), (ask, Side.SELL)):
                if px is None:
                    continue
                delta = (s - px) if side is Side.BUY else (px - s)
                lam = self.A * math.exp(-self.k * max(delta, 0.0))
                if delta < 0 or rng.random() < 1 - math.exp(-lam * self.dt):
                    q = strategy.cfg.order_qty
                    strategy.on_fill(Fill("sim", "SIM", side, px, q, t), state)
                    state.cash -= self.fee_maker * q
                    fills += 1
            # 价格演化
            s += mu * self.dt + self.sigma * math.sqrt(self.dt) * rng.standard_normal()
            if self.alpha_eta > 0:
                mu += -self.alpha_kappa * mu * self.dt + self.alpha_eta * math.sqrt(self.dt) * rng.standard_normal()
            pnl[i] = state.cash + state.inventory * s
            inv[i] = state.inventory
        d = np.diff(pnl, prepend=0.0)
        stats = {
            "final_pnl": float(pnl[-1]),
            "pnl_std_per_step": float(d.std()),
            "sharpe_like": float(d.mean() / d.std() * math.sqrt(len(d))) if d.std() > 0 else 0.0,
            "mean_abs_inventory": float(np.abs(inv).mean()),
            "max_abs_inventory": float(np.abs(inv).max()),
            "n_fills": fills,
        }
        return MMResult(pnl, inv, fills, stats)


@dataclass
class ExecutionSim:
    s0: float = 100.0
    sigma: float = 0.02
    eta: float = 1e-4        # 临时冲击：每股价格让步 = η·(q/dt)
    gamma: float = 1e-5      # 永久冲击
    tick: float = 0.01
    dt: float = 1.0
    queue_size: float = 20.0
    depletion_rate: float = 5.0  # 最优价队列每秒被消耗的股数
    seed: int = 0

    def run(self, executor) -> Dict[str, float]:
        rng = np.random.default_rng(self.seed)
        parent = executor.parent
        s = self.s0
        arrival = s
        state = StrategyState(t=parent.start_ts, horizon_end=parent.end_ts)
        t = parent.start_ts
        working = {}  # order_id -> [order, queue_ahead]
        cost = 0.0
        steps = int(parent.duration / self.dt) + 1
        for _ in range(steps + 1):
            book = make_book(parent.symbol, t, s, self.tick, self.queue_size, self.queue_size)
            for o in executor.on_book(book, state):
                if o.order_type is OrderType.CANCEL:
                    working.pop(o.order_id, None)
                elif o.order_type in (OrderType.IOC, OrderType.MARKET):
                    q = o.qty
                    px = (book.best_ask if o.side is Side.BUY else book.best_bid) + o.side.sign * self.eta * q / self.dt
                    executor.on_fill(Fill(o.order_id, o.symbol, o.side, px, q, t, is_maker=False), state)
                    s += o.side.sign * self.gamma * q
                else:
                    working[o.order_id] = [o, self.queue_size]
            # 被动单排队成交
            consumed = rng.poisson(self.depletion_rate * self.dt)
            for oid, (o, ahead) in list(working.items()):
                rest = consumed - ahead
                if rest > 0:
                    q = min(rest, o.qty)
                    executor.on_fill(Fill(oid, o.symbol, o.side, o.price, q, t), state)
                    o.qty -= q
                    if o.qty <= 1e-12:
                        working.pop(oid)
                    else:
                        working[oid][1] = 0.0
                else:
                    working[oid][1] = ahead - consumed
            s += self.sigma * math.sqrt(self.dt) * rng.standard_normal()
            t += self.dt
            if executor.remaining() <= 1e-12:
                break
        done = executor.executed
        avg_px = -state.cash / state.inventory if parent.side is Side.BUY and state.inventory else (
            state.cash / -state.inventory if state.inventory else float("nan"))
        is_bps = parent.side.sign * (avg_px - arrival) / arrival * 1e4 if done else float("nan")
        return {"executed": done, "avg_price": avg_px, "arrival": arrival, "shortfall_bps": is_bps,
                "completion": done / parent.qty}
