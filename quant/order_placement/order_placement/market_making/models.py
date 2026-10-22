"""经典与前沿做市模型。

- AvellanedaStoikov：Avellaneda & Stoikov (2008)，有限期限 HJB 的近似解，带 alpha 漂移扩展
  （Cartea, Jaimungal & Penalva 2015, 第 10 章）。
- GLFT：Guéant, Lehalle & Fernandez-Tapia (2013) / Guéant (2017) 的无限期限渐近闭式解，
  库存有界，实盘中比 AS 更稳定；含漂移项。
- AdaptiveGLFT：在线估计 σ 与 (A, k) 并据库存动态调整 γ 的工程化变体
  （思路参考 Falces Marín et al. 2022 用 RL 调参 AS 的做法，此处用规则代替 RL）。
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Tuple

import numpy as np

from .base import BaseMarketMaker, MarketMakerConfig


class AvellanedaStoikov(BaseMarketMaker):
    """r = s + μτ − qΔγσ²τ；总价差 = γσ²τ + (2/γ)ln(1+γ/k)。"""

    name = "avellaneda_stoikov"

    def _offsets(self, q, tau, sigma, mu) -> Tuple[float, float]:
        g, k = self.cfg.gamma, self.cfg.k
        dq = self.cfg.order_qty
        r_shift = mu * tau - q * dq * g * sigma**2 * tau
        half = 0.5 * g * sigma**2 * tau + (1.0 / g) * math.log1p(g / k)
        return half - r_shift, half + r_shift


def glft_coefficients(gamma: float, sigma: float, A: float, k: float, lot: float = 1.0) -> Tuple[float, float]:
    """返回 GLFT 闭式解的 (c1, c2)：

    c1 = 1/(γΔ)·ln(1+γΔ/k)
    c2 = sqrt( γσ²/(2AΔk) · (1+γΔ/k)^(1 + k/(γΔ)) )
    """
    gd = gamma * lot
    c1 = math.log1p(gd / k) / gd
    c2 = math.sqrt(gamma * sigma**2 / (2.0 * A * lot * k) * (1.0 + gd / k) ** (1.0 + k / gd))
    return c1, c2


class GLFT(BaseMarketMaker):
    """δ_b(q) = c1 + (−μ/(γσ²) + (2q+1)/2)·Δ·c2，δ_a(q) = c1 + (μ/(γσ²) − (2q−1)/2)·Δ·c2。

    q 以 lot 计；与时间无关（无限期限渐近），适合 7×24 市场。
    """

    name = "glft"

    def _offsets(self, q, tau, sigma, mu) -> Tuple[float, float]:
        cfg = self.cfg
        c1, c2 = glft_coefficients(cfg.gamma, sigma, cfg.A, cfg.k, cfg.order_qty)
        drift_term = mu / (cfg.gamma * sigma**2) / cfg.order_qty if sigma > 0 else 0.0
        db = c1 + (-drift_term + (2 * q + 1) / 2.0) * c2 * cfg.order_qty
        da = c1 + (drift_term - (2 * q - 1) / 2.0) * c2 * cfg.order_qty
        return db, da


class AdaptiveGLFT(GLFT):
    """在线更新 σ（EWMA）并在库存接近上限时放大 γ 的 GLFT。"""

    name = "adaptive_glft"

    def __init__(self, config: Optional[MarketMakerConfig] = None, microprice=None,
                 vol_halflife: int = 200, gamma_inv_mult: float = 3.0):
        super().__init__(config, microprice)
        self._last_mid: Optional[float] = None
        self._last_ts: Optional[float] = None
        self._var: Optional[float] = None
        self._alpha = 1 - 0.5 ** (1.0 / vol_halflife)
        self.gamma_inv_mult = gamma_inv_mult
        self.base_gamma = self.cfg.gamma

    def update_vol(self, mid: float, ts: float) -> float:
        if self._last_mid is not None and ts > self._last_ts:
            r2 = (mid - self._last_mid) ** 2 / (ts - self._last_ts)
            self._var = r2 if self._var is None else (1 - self._alpha) * self._var + self._alpha * r2
        self._last_mid, self._last_ts = mid, ts
        return math.sqrt(self._var) if self._var else self.cfg.sigma

    def quotes(self, book, state, alpha=None, latency=None):
        sigma = self.update_vol(book.mid, book.ts)
        if self._var is not None:
            self.cfg.sigma = max(sigma, 1e-12)
        u = min(abs(state.inventory) / max(self.cfg.max_inventory, 1e-12), 1.0)
        self.cfg.gamma = self.base_gamma * (1.0 + (self.gamma_inv_mult - 1.0) * u**2)
        return super().quotes(book, state, alpha, latency)


def as_optimal_spread(gamma: float, sigma: float, tau: float, k: float) -> float:
    """AS 模型的最优总价差（便于校验/可视化）。"""
    return gamma * sigma**2 * tau + 2.0 / gamma * math.log1p(gamma / k)


def glft_quote_table(cfg: MarketMakerConfig, q_range=range(-5, 6)) -> np.ndarray:
    """生成 (q, δ_b, δ_a) 表，用于文档与参数检查。"""
    mm = GLFT(cfg)
    rows = [(q, *mm._offsets(q, 0.0, cfg.sigma, 0.0)) for q in q_range]
    return np.array(rows)
