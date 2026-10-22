"""
信息几何：Fisher 度量、散度与镜像下降 / 在线组合
================================================

学科来源
--------
Rao (1945) 发现 Fisher 信息矩阵是参数空间上的黎曼度量；Amari (1985) 建立信息几何：
统计流形、对偶联络、Bregman 散度与自然梯度（Amari 1998）。
一元高斯族 (μ, σ) 的 Fisher–Rao 几何是双曲平面（Poincaré 上半平面），测地距离有闭式。

量化映射
--------
- **分布参数的“真实距离”**：比较两段时期 (μ, σ) 的 Fisher–Rao 距离，比欧氏差更合理（σ 小的时候同样的 Δμ 更显著）；
- **在线组合选择 = 简单形上的镜像下降**：Helmbold 等 (1998) 的指数梯度 EG、Cover (1991) 的通用组合，
  KL 作为 Bregman 散度 ⇒ 乘性更新，等价于概率单纯形上的自然梯度；
- **自然梯度**：优化策略参数（如神经网络策略）时按 Fisher 度量缩放梯度，收敛更稳；
- **散度作为距离**：KL/JS/α-散度用于策略去重、regime 比较、模型选择（与最优传输互补）。

核心公式
--------
Fisher 信息（高斯 μ, σ）：I = diag(1/σ², 2/σ²)
Fisher–Rao 距离：d = √2 · arccosh( 1 + [(μ₁−μ₂)²/2 + (σ₁−σ₂)²] / (2σ₁σ₂) )
Bregman 散度：D_F(x,y) = F(x) − F(y) − ⟨∇F(y), x−y⟩；F = Σ x log x ⇒ KL
EG 更新：w_{t+1,i} ∝ w_{t,i} exp(η x_{t,i} / ⟨w_t, x_t⟩)（x 为价格相对数）
通用组合：w_{t+1} = ∫ b S_t(b) dμ(b) / ∫ S_t(b) dμ(b)（对所有恒定再平衡组合按历史财富加权平均）

参考文献
--------
- Amari & Nagaoka (2000). Methods of Information Geometry. AMS.
- Amari (1998). Natural Gradient Works Efficiently in Learning. Neural Computation.
- Cover (1991). Universal Portfolios. Math. Finance.
- Helmbold, Schapire, Singer & Warmuth (1998). On-Line Portfolio Selection Using Multiplicative Updates. Math. Finance.
- Costa, Santos & Strapasson (2015). Fisher information distance: a geometrical reading. Discrete Appl. Math.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .._utils import as_2d, as_rng


# --------------------------------------------------------------------------- 度量与距离
def fisher_information_gaussian(sigma: float) -> np.ndarray:
    return np.diag([1.0 / sigma ** 2, 2.0 / sigma ** 2])


def fisher_rao_gaussian(mu1: float, s1: float, mu2: float, s2: float) -> float:
    """一元高斯之间的 Fisher–Rao 测地距离。"""
    arg = 1 + ((mu1 - mu2) ** 2 / 2 + (s1 - s2) ** 2) / (2 * s1 * s2)
    return float(np.sqrt(2) * np.arccosh(max(arg, 1.0)))


def kl_gaussian(mu1, s1, mu2, s2) -> float:
    return float(np.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2 * s2 ** 2) - 0.5)


def kl_divergence(p, q, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps; q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    m = 0.5 * (p / p.sum() + q / q.sum())
    return float(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))


def hellinger_distance(p, q) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p /= p.sum(); q /= q.sum()
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def alpha_divergence(p, q, alpha: float) -> float:
    """Amari α-散度（α→1 为 KL(p‖q)，α→−1 为 KL(q‖p)，α=0 为 Hellinger 型）。"""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p /= p.sum(); q /= q.sum()
    if abs(alpha - 1) < 1e-12:
        return kl_divergence(p, q)
    if abs(alpha + 1) < 1e-12:
        return kl_divergence(q, p)
    return float(4 / (1 - alpha ** 2) * (1 - np.sum(p ** ((1 - alpha) / 2) * q ** ((1 + alpha) / 2))))


def bregman_divergence(F: Callable, gradF: Callable, x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    return float(F(x) - F(y) - gradF(y) @ (x - y))


def natural_gradient_step(theta, grad, fisher, lr: float = 0.1, damping: float = 1e-6) -> np.ndarray:
    """自然梯度 θ ← θ + lr · F^{-1} ∇（最大化目标时）。"""
    F = np.asarray(fisher, dtype=float) + damping * np.eye(len(theta))
    return np.asarray(theta, dtype=float) + lr * np.linalg.solve(F, np.asarray(grad, dtype=float))


def rolling_fisher_rao(returns, window: int = 60, reference: tuple | None = None) -> dict:
    """滚动 (μ, σ) 与基准 (μ₀, σ₀) 的 Fisher–Rao 距离序列。"""
    r = np.asarray(returns, dtype=float)
    mu0, s0 = (r.mean(), r.std()) if reference is None else reference
    idx, d = [], []
    for end in range(window, len(r) + 1):
        seg = r[end - window:end]
        idx.append(end - 1); d.append(fisher_rao_gaussian(seg.mean(), seg.std() + 1e-12, mu0, s0))
    return dict(index=np.array(idx), distance=np.array(d))


# --------------------------------------------------------------------------- 在线组合
def exponentiated_gradient(price_relatives, eta: float = 0.05, w0=None) -> dict:
    """EG 在线组合（Helmbold 等 1998）。price_relatives (T, n)，x_{t,i} = P_{t,i}/P_{t−1,i}。

    返回 ``weights (T, n)``（每期用于下一期的权重）、``wealth (T,)``。
    """
    X = as_2d(price_relatives, "price_relatives")
    T, n = X.shape
    w = np.full(n, 1 / n) if w0 is None else np.asarray(w0, dtype=float)
    W = np.zeros((T, n)); wealth = np.zeros(T); S = 1.0
    for t in range(T):
        W[t] = w
        r = w @ X[t]; S *= r; wealth[t] = S
        w = w * np.exp(eta * X[t] / r); w /= w.sum()
    return dict(weights=W, wealth=wealth)


def constant_rebalanced(price_relatives, b) -> np.ndarray:
    """恒定再平衡组合（CRP）的财富路径。"""
    X = as_2d(price_relatives, "price_relatives")
    return np.cumprod(X @ np.asarray(b, dtype=float))


def universal_portfolio(price_relatives, n_samples: int = 2000, seed=None) -> dict:
    """Cover 通用组合（Dirichlet(1) 采样近似积分）。返回 ``weights``、``wealth``。"""
    rng = as_rng(seed)
    X = as_2d(price_relatives, "price_relatives")
    T, n = X.shape
    B = rng.dirichlet(np.ones(n), size=n_samples)          # (K, n)
    S = np.ones(n_samples)
    W = np.zeros((T, n)); wealth = np.zeros(T); S_up = 1.0
    for t in range(T):
        w = (S @ B) / S.sum()
        W[t] = w
        S_up *= w @ X[t]; wealth[t] = S_up
        S *= B @ X[t]
    return dict(weights=W, wealth=wealth)


def best_crp_in_hindsight(price_relatives) -> dict:
    """事后最优 CRP（基准）。"""
    from scipy.optimize import minimize
    X = as_2d(price_relatives, "price_relatives")
    n = X.shape[1]
    res = minimize(lambda b: -np.sum(np.log(X @ b)), np.full(n, 1 / n), bounds=[(0, 1)] * n,
                   constraints=[dict(type="eq", fun=lambda b: b.sum() - 1)], method="SLSQP")
    return dict(weights=res.x, wealth=float(np.exp(-res.fun)))
