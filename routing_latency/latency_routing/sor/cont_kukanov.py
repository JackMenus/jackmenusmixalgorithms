"""Cont & Kukanov (2017) 多场所最优挂单拆分。

论文：R. Cont, A. Kukanov, "Optimal order placement in limit order markets",
Quantitative Finance 17(1), 2017.

问题：在时间窗 T 内需要成交 S 股。决策向量 X = (M, L_1..L_K)：
M 为立即市价（吃单）数量，L_k 为在场所 k 挂限价单的数量。场所 k 当前排在我们前面的队列为 Q_k，
窗口内该价位的流出量（成交+撤单）为随机变量 ξ_k。限价单成交量

    fill_k = min(L_k, (ξ_k - Q_k)^+)
    A(X, ξ) = M + Σ fill_k

成本（每股，以价格单位计）：

    V = (h + f) M - Σ (h + r_k) fill_k + λ_u (S - A)^+ + λ_o (A - S)^+

h 为半价差，f 为吃单费，r_k 为挂单返佣，λ_u / λ_o 为欠成交/超成交惩罚。
λ_u 越大越「急」，可由挂单策略库的 urgency 或因子库的 alpha 推出（见 ``urgency_to_penalty``）。

求解：论文给出凸性证明并用随机逼近（Robbins-Monro）。这里实现样本平均近似 (SAA) +
投影次梯度下降，样本 ξ 可以来自历史回放、Hawkes 过程模拟或简单的参数分布。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class CKResult:
    market_qty: float
    limit_qty: np.ndarray
    expected_cost: float          # 每股价格单位
    expected_fill: float
    shortfall_prob: float


def ck_cost(X: np.ndarray, S: float, Q: np.ndarray, xi: np.ndarray, h: float, f: float, r: np.ndarray, lam_u: float, lam_o: float) -> np.ndarray:
    """对每个样本返回总成本 V(X, ξ)；xi 形状 (N, K)。"""
    M, L = X[0], X[1:]
    fill = np.minimum(L, np.maximum(xi - Q, 0.0))
    A = M + fill.sum(1)
    return (h + f) * M - (fill * (h + r)).sum(1) + lam_u * np.maximum(S - A, 0) + lam_o * np.maximum(A - S, 0)


def solve_cont_kukanov(
    S: float,
    Q: Sequence[float],
    xi_samples: np.ndarray,
    half_spread: float,
    taker_fee: float,
    rebates: Sequence[float],
    lam_u: float,
    lam_o: float,
    iters: int = 400,
    step0: Optional[float] = None,
    x0: Optional[np.ndarray] = None,
) -> CKResult:
    Q = np.asarray(Q, dtype=float)
    r = np.asarray(rebates, dtype=float)
    xi = np.asarray(xi_samples, dtype=float)
    K = len(Q)
    if xi.shape[1] != K:
        raise ValueError("xi_samples 列数须等于场所数")
    h, f = half_spread, taker_fee
    X = np.array(x0, dtype=float) if x0 is not None else np.concatenate([[S * 0.0], np.full(K, S / K)])
    step0 = step0 if step0 is not None else S / 4.0
    scale = max(h + f, lam_u, lam_o, 1e-12)
    best_X, best_v = X.copy(), np.inf
    for t in range(1, iters + 1):
        M, L = X[0], X[1:]
        excess = xi - Q
        fill = np.minimum(L, np.maximum(excess, 0.0))
        A = M + fill.sum(1)
        under = (A < S).astype(float)
        over = (A > S).astype(float)
        pen = -lam_u * under + lam_o * over                     # dV/dA, 每个样本
        gM = (h + f) + pen.mean()
        active = (excess > L).astype(float)                       # d fill_k / d L_k
        gL = (active * (-(h + r)[None, :] + pen[:, None])).mean(0)
        g = np.concatenate([[gM], gL]) / scale
        X = np.clip(X - step0 / np.sqrt(t) * g, 0.0, S)
        if t % 20 == 0 or t == iters:
            v = ck_cost(X, S, Q, xi, h, f, r, lam_u, lam_o).mean()
            if v < best_v:
                best_v, best_X = v, X.copy()
    M, L = best_X[0], best_X[1:]
    fill = np.minimum(L, np.maximum(xi - Q, 0.0))
    A = M + fill.sum(1)
    return CKResult(
        market_qty=float(M),
        limit_qty=L,
        expected_cost=float(best_v) / S,
        expected_fill=float(np.minimum(A, S).mean()),
        shortfall_prob=float(np.mean(A < S - 1e-9)),
    )


def urgency_to_penalty(urgency: float, half_spread: float, taker_fee: float, alpha_per_share: float = 0.0) -> float:
    """把 0~1 的 urgency 与 alpha 映射成欠成交惩罚 λ_u。

    论文给出的边界：λ_u > h + f 时才会使用市价单。这里令
        λ_u = (h + f) * (0.5 + 2 * urgency) + max(alpha, 0)
    urgency≈0.25 时恰在边界；alpha 越强，错过成交的机会成本越高。
    """
    return (half_spread + taker_fee) * (0.5 + 2.0 * urgency) + max(alpha_per_share, 0.0)


def default_outflow_samples(queue_sizes: Sequence[float], turnover: Sequence[float], n: int = 2000, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """缺少历史数据时的默认流出量样本：Gamma(shape=1.5) 分布，均值 = queue * turnover。

    turnover 表示窗口内该价位被消耗的队列倍数（例如 1.2 表示平均流出 1.2 倍当前队列）。
    实盘请用历史回放或 Hawkes 模拟替换。
    """
    rng = rng or np.random.default_rng(0)
    mean = np.asarray(queue_sizes, dtype=float) * np.asarray(turnover, dtype=float)
    shape = 1.5
    return rng.gamma(shape, np.maximum(mean, 1e-9) / shape, size=(n, len(mean)))
