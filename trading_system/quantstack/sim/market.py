"""合成行情。

两部分：
  * ``make_synthetic_panel``：日频面板（植入短期反转与量价背离两类可挖掘 alpha），格式与因子库 Panel 一致；
  * ``SyntheticMarket``：日内多场所 L1 行情过程（公允中间价几何布朗运动 + 永久/临时冲击 + 场所报价错位 + 延迟分布），
    实现 ``MarketDataFeed`` 协议，并为 ``VenueSimulator`` 提供撮合所需的真实延迟样本。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..contracts import US_PER_SEC, MarketSnapshot, Panel, VenueQuote

SESSION_US = 23_400 * US_PER_SEC  # 6.5 小时


# ----------------------------------------------------------------------------- 日频面板
def make_synthetic_panel(n_days: int = 400, n_assets: int = 60, seed: int = 0, signal_strength: float = 0.03) -> Panel:
    """生成带可挖掘信号的日频面板（与 factormine.data.make_synthetic_panel 同构）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    assets = [f"A{i:04d}" for i in range(n_assets)]
    market = rng.normal(0.0003, 0.01, n_days)
    beta = rng.uniform(0.6, 1.4, n_assets)
    idio = rng.uniform(0.01, 0.03, n_assets)
    base_vol = rng.lognormal(13.5, 0.5, n_assets)
    rets = np.zeros((n_days, n_assets))
    vols = np.zeros((n_days, n_assets))
    shock = rng.lognormal(0, 0.35, (n_days, n_assets))
    for t in range(n_days):
        alpha = np.zeros(n_assets)
        if t >= 20:
            past5 = rets[t - 5:t].sum(0)
            rev = -_zs(past5)
            vratio = vols[t - 1] / vols[t - 20:t].mean(0)
            vsig = -_zs(np.log(vratio))
            alpha = signal_strength * idio * (0.7 * rev + 0.5 * vsig)
        rets[t] = beta * market[t] + alpha + idio * rng.standard_normal(n_assets)
        vols[t] = base_vol * shock[t] * (1 + 8 * np.abs(rets[t]))
    close = 50 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    open_ = close / (1 + rets) * (1 + rng.normal(0, 0.002, close.shape))
    rng_spread = np.abs(rng.normal(0, 0.01, close.shape)) * close
    high = np.maximum(open_, close) + rng_spread
    low = np.minimum(open_, close) - rng_spread
    mk = lambda a: pd.DataFrame(a, index=dates, columns=assets)
    panel: Panel = {"open": mk(open_), "high": mk(high), "low": mk(low), "close": mk(close), "volume": mk(vols), "returns": mk(rets)}
    panel["vwap"] = (panel["high"] + panel["low"] + panel["close"]) / 3.0
    return panel


def _zs(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def panel_stats(panel: Panel, as_of=None, vol_window: int = 20, adv_window: int = 20) -> Dict[str, Dict[str, float]]:
    """从面板得到日内模拟需要的起点：最新价、日波动率、平均日成交量。"""
    close = panel["close"]
    volume = panel["volume"]
    if as_of is not None:
        close = close.loc[:as_of]
        volume = volume.loc[:as_of]
    rets = close.pct_change(fill_method=None)
    return {
        "price": close.iloc[-1].to_dict(),
        "volatility": rets.iloc[-vol_window:].std().fillna(0.02).to_dict(),
        "adv": volume.iloc[-adv_window:].mean().fillna(1e6).to_dict(),
    }


# ----------------------------------------------------------------------------- 日内多场所行情
@dataclass
class VenueSpec:
    name: str
    taker_fee: float = 0.0030
    maker_fee: float = -0.0020
    latency_median_us: float = 400.0     # 单程延迟中位数
    latency_sigma: float = 0.30          # 对数正态 σ
    tail_prob: float = 0.02              # 突刺概率
    tail_scale_us: float = 4000.0        # 突刺量级
    size_scale: float = 1.0              # 可见量相对规模
    dislocation_bps: float = 0.6         # 报价相对公允价的噪声
    quote_update_rate: float = 50.0      # 报价更新强度（次/秒）
    liquidity_share: float = 0.25        # 成交量份额
    hidden_liquidity: float = 0.2        # 隐藏流动性（相对可见量）


DEFAULT_VENUES: List[VenueSpec] = [
    VenueSpec("NYSE", 0.0030, -0.0020, 350.0, 0.25, 0.02, 3000.0, 1.2, 0.5, 60.0, 0.30, 0.2),
    VenueSpec("NASDAQ", 0.0030, -0.0025, 300.0, 0.30, 0.03, 4000.0, 1.0, 0.6, 80.0, 0.30, 0.15),
    VenueSpec("ARCA", 0.0028, -0.0020, 450.0, 0.30, 0.02, 3500.0, 0.8, 0.8, 50.0, 0.20, 0.2),
    VenueSpec("IEX", 0.0009, 0.0000, 850.0, 0.15, 0.01, 2000.0, 0.5, 0.3, 20.0, 0.20, 0.4),  # 350us speed bump
]


@dataclass
class _SymbolState:
    mid: float
    daily_vol: float
    adv: float
    perm_impact: float = 0.0     # 永久冲击（相对）
    temp_impact: float = 0.0     # 临时冲击（相对），指数衰减
    session_volume: float = 0.0
    last_ts: int = 0


class SyntheticMarket:
    """日内行情过程，实现 ``MarketDataFeed``。``snapshot`` 会把状态推进到请求时刻（惰性演化）。"""

    def __init__(
        self,
        prices: Dict[str, float],
        volatilities: Dict[str, float],
        adv: Dict[str, float],
        venues: Sequence[VenueSpec] = DEFAULT_VENUES,
        start_us: int = 0,
        tick: float = 0.01,
        seed: int = 0,
        gamma_perm: float = 1.0,        # 永久冲击：Δp/p = γ · σ_daily · q/ADV（线性）
        eta_temp: float = 5.0,          # 临时冲击（线性）：Δp/p = η · σ_daily · q/ADV，按 impact_halflife_us 衰减
        impact_halflife_us: int = 60 * US_PER_SEC,
        spread_bps_base: float = 2.0,
    ):
        self.venues = list(venues)
        self.venue_by_name = {v.name: v for v in self.venues}
        self.tick = tick
        self.start_us = int(start_us)
        self.gamma_perm = gamma_perm
        self.eta_temp = eta_temp
        self.impact_halflife_us = impact_halflife_us
        self.spread_bps_base = spread_bps_base
        self._rng = np.random.default_rng(seed)
        self._quote_cache: Dict[tuple, tuple] = {}
        self._state: Dict[str, _SymbolState] = {
            s: _SymbolState(mid=float(prices[s]), daily_vol=float(max(volatilities.get(s, 0.02), 1e-4)), adv=float(max(adv.get(s, 1e6), 1e3)), last_ts=self.start_us)
            for s in prices
        }

    # ------------------------------------------------------------------ 协议
    def symbols(self) -> List[str]:
        return list(self._state)

    def advance_to(self, ts_us: int) -> None:
        for st in self._state.values():
            dt = ts_us - st.last_ts
            if dt <= 0:
                continue
            frac = dt / SESSION_US
            sigma = st.daily_vol * math.sqrt(frac)
            st.mid *= math.exp(sigma * self._rng.standard_normal() - 0.5 * sigma * sigma)
            st.temp_impact *= math.exp(-dt * math.log(2) / self.impact_halflife_us)
            st.session_volume += st.adv * frac * self._rng.lognormal(0, 0.3)
            st.last_ts = ts_us

    def fair_price(self, symbol: str) -> float:
        st = self._state[symbol]
        return st.mid * (1.0 + st.perm_impact + st.temp_impact)

    def snapshot(self, symbol: str, ts_us: int) -> MarketSnapshot:
        self.advance_to(ts_us)
        st = self._state[symbol]
        fair = self.fair_price(symbol)
        spread_bps = self.spread_bps_base + 60.0 * st.daily_vol
        half = max(self.tick / 2.0, fair * spread_bps / 2e4)
        per_sec_vol = st.adv / (SESSION_US / US_PER_SEC)
        quotes = []
        for v in self.venues:
            key = (symbol, v.name)
            cached = self._quote_cache.get(key)
            # 场所报价只在「更新事件」时刷新（泊松强度 quote_update_rate），两次更新之间保持陈旧 —— 这正是延迟套利/陈旧报价风险的来源
            stale_px = cached is not None and abs(fair - cached[5]) > 0.5 * self.tick
            empty = cached is not None and (cached[3] <= 0 or cached[4] <= 0)
            if cached is None or stale_px or empty or self._rng.random() < 1.0 - math.exp(-v.quote_update_rate * (ts_us - cached[0]) / 1e6):
                disloc = self._rng.normal(0.0, v.dislocation_bps) * 1e-4 * fair
                bid = math.floor((fair + disloc - half) / self.tick) * self.tick
                ask = math.ceil((fair + disloc + half) / self.tick) * self.tick
                if ask - bid < self.tick - 1e-12:
                    ask = bid + self.tick
                size_mu = per_sec_vol * 5.0 * v.size_scale  # 约 5 秒成交量
                bs = int(max(100, self._rng.lognormal(math.log(size_mu + 1), 0.5)))
                as_ = int(max(100, self._rng.lognormal(math.log(size_mu + 1), 0.5)))
                cached = (ts_us, round(bid, 6), round(ask, 6), bs, as_, fair)
                self._quote_cache[key] = cached
            _, bid, ask, bs, as_, _ = cached
            quotes.append(
                VenueQuote(
                    venue=v.name, symbol=symbol, bid=bid, ask=ask, bid_size=bs, ask_size=as_,
                    ts_us=cached[0], taker_fee=v.taker_fee, maker_fee=v.maker_fee, quote_update_rate=v.quote_update_rate,
                )
            )
        return MarketSnapshot(
            symbol=symbol, ts_us=ts_us, quotes=quotes, last_price=round(fair, 6), adv=st.adv, volatility=st.daily_vol,
            session_volume=st.session_volume, extra={"fair": fair},
        )

    # ------------------------------------------------------------------ 撮合器回调
    def apply_trade(self, symbol: str, side: int, quantity: int, aggressor: bool = True) -> None:
        """记录一笔成交对公允价的影响。

        * 只有 **主动（taker）成交** 施加冲击：被动成交时对手方才是价格推动者，其影响已体现在价格走到我们挂单价这件事里；
        * 临时冲击按 **线性** 计入（Δp/p = η·σ_daily·q/ADV，按半衰期衰减），避免平方根律逐笔叠加时「切得越细冲击越大」的系统性偏差；
          整单的平方根律形态由永久冲击 + 临时冲击叠加近似；
        * 永久冲击线性：Δp/p = γ·σ_daily·q/ADV。
        """
        st = self._state[symbol]
        st.session_volume += quantity
        if not aggressor:
            return
        x = quantity / st.adv
        st.perm_impact += side * self.gamma_perm * st.daily_vol * x
        st.temp_impact += side * self.eta_temp * st.daily_vol * x

    def consume(self, symbol: str, venue: str, side: int, quantity: int) -> None:
        """某场所对手方可见量被吃掉 quantity（下一次报价更新前保持减少后的量）。"""
        key = (symbol, venue)
        c = self._quote_cache.get(key)
        if c is None:
            return
        ts, bid, ask, bs, as_, fair = c
        if side > 0:
            as_ = max(as_ - quantity, 0)
        else:
            bs = max(bs - quantity, 0)
        self._quote_cache[key] = (ts, bid, ask, bs, as_, fair)

    def latency_sample(self, venue: str) -> float:
        v = self.venue_by_name[venue]
        lat = v.latency_median_us * math.exp(v.latency_sigma * self._rng.standard_normal())
        if self._rng.random() < v.tail_prob:
            lat += self._rng.exponential(v.tail_scale_us)
        return float(lat)

    def venue_spec(self, venue: str) -> VenueSpec:
        return self.venue_by_name[venue]

    def rng(self) -> np.random.Generator:
        return self._rng
