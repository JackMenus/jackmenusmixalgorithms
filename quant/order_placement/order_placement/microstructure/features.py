"""限价订单簿 (LOB) 微结构特征。

包含：
- 加权中间价 / 失衡度 (imbalance)
- 单层与多层订单流失衡 OFI（Cont, Kukanov & Stoikov 2014；Xu, Gould & Howison 2019）
- 经验微价格 micro-price（Stoikov 2018 的经验近似）
- Kyle λ 价格冲击系数估计
- 基于中间价收益的已实现波动率
- 队列扩散近似下的“下一次价格上涨概率”（Cont & de Larrard 2013 的布朗近似）
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import BookSnapshot


def imbalance(book: BookSnapshot, levels: int = 1) -> float:
    """前 levels 档的买卖量失衡 I = (Vb - Va)/(Vb + Va) ∈ [-1, 1]。"""
    vb = sum(q for _, q in book.bids[:levels])
    va = sum(q for _, q in book.asks[:levels])
    tot = vb + va
    return 0.0 if tot <= 0 else (vb - va) / tot


def weighted_mid(book: BookSnapshot) -> float:
    """按对手方量加权的中间价：Pb·Va/(Vb+Va) 的对偶形式。"""
    (pb, qb), (pa, qa) = book.bids[0], book.asks[0]
    if qb + qa <= 0:
        return book.mid
    return (pb * qa + pa * qb) / (qb + qa)


def ofi_step(prev: BookSnapshot, cur: BookSnapshot, level: int = 0) -> float:
    """单步 OFI（Cont-Kukanov-Stoikov 2014，式 (3)），可指定档位实现多层 OFI。"""
    pb0, qb0 = prev.bids[level]
    pb1, qb1 = cur.bids[level]
    pa0, qa0 = prev.asks[level]
    pa1, qa1 = cur.asks[level]
    e = 0.0
    e += qb1 if pb1 >= pb0 else 0.0
    e -= qb0 if pb1 <= pb0 else 0.0
    e -= qa1 if pa1 <= pa0 else 0.0
    e += qa0 if pa1 >= pa0 else 0.0
    return e


def ofi_series(books: Sequence[BookSnapshot], levels: int = 1) -> np.ndarray:
    """返回形状 (n-1, levels) 的多层 OFI 序列（Xu-Gould-Howison 2019 MLOFI）。"""
    out = np.zeros((max(len(books) - 1, 0), levels))
    for i in range(1, len(books)):
        for lv in range(levels):
            out[i - 1, lv] = ofi_step(books[i - 1], books[i], lv)
    return out


def integrated_ofi(mlofi: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """积分 OFI（Cont-Cucuringu-Zhang 2023）：多层 OFI 投影到第一主成分。

    若未提供 weights，则用 PCA 第一主成分（按 L1 归一化）。
    """
    if mlofi.ndim != 2 or mlofi.shape[0] == 0:
        return np.zeros(0)
    if weights is None:
        x = mlofi - mlofi.mean(axis=0)
        cov = x.T @ x / max(len(x) - 1, 1)
        w_eig, v = np.linalg.eigh(cov)
        weights = v[:, -1]
        weights = weights * np.sign(weights.sum() or 1.0)
        weights = weights / np.abs(weights).sum()
    return mlofi @ weights


def kyle_lambda(price_changes: np.ndarray, signed_volume: np.ndarray) -> float:
    """OLS 估计 ΔP = λ · OFI + ε 中的 λ（价格冲击系数）。"""
    x = np.asarray(signed_volume, float)
    y = np.asarray(price_changes, float)
    denom = float(x @ x)
    return 0.0 if denom == 0 else float(x @ y) / denom


def realized_vol(mids: np.ndarray, dt: float) -> float:
    """以价格单位/√秒 计的已实现波动率 σ（做市模型中的 σ）。"""
    mids = np.asarray(mids, float)
    if len(mids) < 2 or dt <= 0:
        return 0.0
    return float(np.std(np.diff(mids), ddof=1) / np.sqrt(dt))


def prob_up_diffusion(bid_queue: float, ask_queue: float) -> float:
    """队列扩散近似下，卖一先于买一耗尽（价格上涨）的概率。

    将两档队列视为独立、同方差布朗运动，在第一象限中的调和函数解为
    P_up = 1 - (2/π)·arctan(ask/bid)。参见 Cont & de Larrard (2013) 的重流量极限。
    """
    if bid_queue <= 0 and ask_queue <= 0:
        return 0.5
    if bid_queue <= 0:
        return 0.0
    return float(1.0 - 2.0 / np.pi * np.arctan(ask_queue / bid_queue))


class EmpiricalMicroprice:
    """Stoikov (2018) 微价格的经验近似。

    微价格定义 P_micro = mid + g(I, S)，其中 g 为在失衡度 I 与价差 S 条件下
    中间价的长期期望调整。原论文通过马尔可夫链迭代求解 G* = Σ B^i G¹；
    此处采用直接回归：g(I,S) ≈ E[mid_{t+h} - mid_t | I_t, S_t]，在 h 足够大时等价于其极限。
    用对称化（把 (I,ΔM) 与 (-I,-ΔM) 同时入样）保证无偏。
    """

    def __init__(self, n_imb_bins: int = 10, max_spread_ticks: int = 3):
        self.n_imb_bins = n_imb_bins
        self.max_spread_ticks = max_spread_ticks
        self.table: Optional[np.ndarray] = None

    def _bin(self, imb: np.ndarray, spread_ticks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ib = np.clip(((imb + 1) / 2 * self.n_imb_bins).astype(int), 0, self.n_imb_bins - 1)
        sb = np.clip(np.rint(spread_ticks).astype(int), 1, self.max_spread_ticks) - 1
        return ib, sb

    def fit(self, books: Sequence[BookSnapshot], horizon_steps: int = 10) -> "EmpiricalMicroprice":
        n = len(books) - horizon_steps
        if n <= 0:
            raise ValueError("样本长度不足")
        imb = np.array([imbalance(b) for b in books[:n]])
        sp = np.array([b.spread_ticks for b in books[:n]])
        dm = np.array([books[i + horizon_steps].mid - books[i].mid for i in range(n)])
        imb = np.concatenate([imb, -imb])
        sp = np.concatenate([sp, sp])
        dm = np.concatenate([dm, -dm])
        ib, sb = self._bin(imb, sp)
        s = np.zeros((self.n_imb_bins, self.max_spread_ticks))
        c = np.zeros_like(s)
        np.add.at(s, (ib, sb), dm)
        np.add.at(c, (ib, sb), 1.0)
        self.table = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
        return self

    def adjustment(self, book: BookSnapshot) -> float:
        if self.table is None:
            # 未拟合时退化为加权中间价
            return weighted_mid(book) - book.mid
        ib, sb = self._bin(np.array([imbalance(book)]), np.array([book.spread_ticks]))
        return float(self.table[ib[0], sb[0]])

    def price(self, book: BookSnapshot) -> float:
        return book.mid + self.adjustment(book)


def feature_vector(book: BookSnapshot, prev: Optional[BookSnapshot] = None, levels: int = 3) -> Dict[str, float]:
    """一次性生成常用特征，供因子库或 RL 状态使用。"""
    f = {
        "mid": book.mid,
        "spread": book.spread,
        "spread_ticks": book.spread_ticks,
        "imbalance_l1": imbalance(book, 1),
        f"imbalance_l{levels}": imbalance(book, levels),
        "weighted_mid": weighted_mid(book),
        "prob_up": prob_up_diffusion(book.bids[0][1], book.asks[0][1]),
    }
    if prev is not None:
        lv = min(levels, len(book.bids), len(prev.bids), len(book.asks), len(prev.asks))
        for i in range(lv):
            f[f"ofi_l{i + 1}"] = ofi_step(prev, book, i)
    return f
