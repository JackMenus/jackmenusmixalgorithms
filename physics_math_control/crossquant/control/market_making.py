"""
做市的随机最优控制：Avellaneda–Stoikov 与 Guéant–Lehalle–Fernandez-Tapia
=======================================================================

学科来源
--------
随机最优控制 / HJB 方程（Bellman 1957；Fleming–Soner）。做市商控制买卖报价距离 (δ^b, δ^a)，
成交到达是强度 λ(δ) = A e^{−kδ} 的泊松过程，库存 q 带来价格风险；最大化终端财富的指数效用。
HJB 方程经 Guéant 等的变量代换变成**线性 ODE 系统**（可用矩阵指数精确求解）。

量化映射
--------
- 报价：中间价附近对称/非对称挂单的距离，库存越多越向“去库存”方向偏移；
- 库存风险：γσ²(T−t) 项即“保留价格”对库存的惩罚；
- 与挂单策略库衔接：本模块给出 δ^b(q,t), δ^a(q,t)，挂单库负责落地成实际订单。

核心公式
--------
Avellaneda–Stoikov (2008) 近似解：
    保留价格  r(s,q,t) = s − q γ σ² (T − t)
    总价差    δ^a + δ^b = γσ²(T−t) + (2/γ) ln(1 + γ/k)
GLFT (2013) 精确解：令 α = kγσ²/2，η = A(1+γ/k)^{−(1+k/γ)}，库存限制 |q| ≤ Q，
    v'(t) = M v(t)，M_{qq} = αq²，M_{q,q±1} = −η，v(T) = 1  ⇒  v(t) = exp(−M(T−t))·1
    δ^b*(t,q) = (1/k) ln(v_q/v_{q+1}) + (1/γ) ln(1+γ/k),   δ^a*(t,q) = (1/k) ln(v_q/v_{q−1}) + (1/γ) ln(1+γ/k)
渐近（T→∞）解由 M 的最小特征值对应的 Perron 特征向量 f 给出：v_q ∝ f_q。
GLFT 闭式近似：
    δ^b ≈ (1/γ) ln(1+γ/k) + (2q+1)/2 · sqrt( γσ²/(2kA) (1+γ/k)^{1+k/γ} )
    δ^a ≈ (1/γ) ln(1+γ/k) − (2q−1)/2 · sqrt( γσ²/(2kA) (1+γ/k)^{1+k/γ} )

参考文献
--------
- Avellaneda & Stoikov (2008). High-frequency trading in a limit order book. QF.
- Guéant, Lehalle & Fernandez-Tapia (2013). Dealing with the inventory risk: a solution to the market making problem. MAFE. arXiv:1105.3115
- Cartea, Jaimungal & Penalva (2015). Algorithmic and High-Frequency Trading. CUP.
- Guéant (2017). Optimal market making. Applied Mathematical Finance（多资产/一般强度）。
- Asadulaev (2026). Zero-shot adaptation to order book dynamics. arXiv:2605.21707（AS/GLFT 的自适应扩展）。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy import linalg


# --------------------------------------------------------------------------- Avellaneda–Stoikov
def avellaneda_stoikov_quotes(s: float, q: float, t: float, T: float, gamma: float, sigma: float, k: float) -> dict:
    """AS (2008) 近似最优报价。返回 ``reservation``、``spread``、``bid``、``ask``、``delta_b``、``delta_a``。"""
    tau = max(T - t, 0.0)
    r = s - q * gamma * sigma ** 2 * tau
    spread = gamma * sigma ** 2 * tau + (2.0 / gamma) * np.log(1 + gamma / k)
    bid, ask = r - spread / 2, r + spread / 2
    return dict(reservation=r, spread=spread, bid=bid, ask=ask, delta_b=s - bid, delta_a=ask - s)


# --------------------------------------------------------------------------- GLFT
def glft_closed_form_quotes(q, gamma: float, sigma: float, k: float, A: float) -> dict:
    """GLFT (2013) 无限时域闭式近似报价距离（q 可为数组）。"""
    q = np.asarray(q, dtype=float)
    base = (1.0 / gamma) * np.log(1 + gamma / k)
    c = np.sqrt(gamma * sigma ** 2 / (2 * k * A) * (1 + gamma / k) ** (1 + k / gamma))
    delta_b = base + (2 * q + 1) / 2 * c
    delta_a = base - (2 * q - 1) / 2 * c
    return dict(delta_b=delta_b, delta_a=delta_a)


def _glft_matrix(gamma, sigma, k, A, Q):
    alpha = k * gamma * sigma ** 2 / 2
    eta = A * (1 + gamma / k) ** (-(1 + k / gamma))
    qs = np.arange(-Q, Q + 1)
    M = np.diag(alpha * qs.astype(float) ** 2)
    off = -eta * np.ones(len(qs) - 1)
    M += np.diag(off, 1) + np.diag(off, -1)
    return M, qs


def glft_exact_quotes(gamma: float, sigma: float, k: float, A: float, Q: int, T: float, t: float = 0.0) -> dict:
    """GLFT 有限时域精确解：v(t) = exp(−M(T−t))·1，报价距离由相邻 v_q 的对数比给出。

    返回 ``q``（−Q..Q）、``delta_b``（q=Q 处无买单，为 nan）、``delta_a``（q=−Q 处为 nan）、``v``（已按 e^{λ_min(T−t)} 归一）。
    """
    M, qs = _glft_matrix(gamma, sigma, k, A, Q)
    # 报价只依赖 v_q 之间的比值，先把 M 的最小特征值平移到 0 以避免长时域下的下溢
    shift = np.linalg.eigvalsh(M).min()
    v = linalg.expm(-(M - shift * np.eye(len(qs))) * (T - t)) @ np.ones(len(qs))
    base = (1.0 / gamma) * np.log(1 + gamma / k)
    delta_b = np.full(len(qs), np.nan); delta_a = np.full(len(qs), np.nan)
    delta_b[:-1] = (1.0 / k) * np.log(v[:-1] / v[1:]) + base
    delta_a[1:] = (1.0 / k) * np.log(v[1:] / v[:-1]) + base
    return dict(q=qs, delta_b=delta_b, delta_a=delta_a, v=v)


def glft_asymptotic_quotes(gamma: float, sigma: float, k: float, A: float, Q: int) -> dict:
    """GLFT 渐近（T→∞）解：用 M 的最小特征值的 Perron 特征向量 f 代替 v。"""
    M, qs = _glft_matrix(gamma, sigma, k, A, Q)
    w, V = np.linalg.eigh(M)
    f = V[:, np.argmin(w)]
    f = f * np.sign(f[np.argmax(np.abs(f))])
    f = np.maximum(f, 1e-300)
    base = (1.0 / gamma) * np.log(1 + gamma / k)
    delta_b = np.full(len(qs), np.nan); delta_a = np.full(len(qs), np.nan)
    delta_b[:-1] = (1.0 / k) * np.log(f[:-1] / f[1:]) + base
    delta_a[1:] = (1.0 / k) * np.log(f[1:] / f[:-1]) + base
    return dict(q=qs, delta_b=delta_b, delta_a=delta_a, f=f, lambda0=float(w.min()))


# --------------------------------------------------------------------------- 模拟
def simulate_market_making(quote_fn: Callable[[float, int, float], tuple], S0: float = 100.0, sigma: float = 2.0,
                           T: float = 1.0, dt: float = 0.005, A: float = 140.0, k: float = 1.5,
                           max_inventory: Optional[int] = None, seed=None) -> dict:
    """做市策略蒙特卡洛（AS 2008 的实验设置：S0=100, σ=2, T=1, dt=0.005, A=140, k=1.5）。

    quote_fn(s, q, t) -> (delta_b, delta_a)，返回买卖报价离中间价的距离；
    成交：买单以概率 A e^{−kδ_b} dt 被动成交（库存 +1），卖单同理。
    返回 ``pnl``、``inventory``（路径）、``cash``、``price``、``final_pnl``。
    """
    rng = np.random.default_rng(seed)
    n = int(round(T / dt))
    s = S0; q = 0; cash = 0.0
    price = np.zeros(n + 1); inv = np.zeros(n + 1, dtype=int); pnl = np.zeros(n + 1)
    price[0] = s
    for i in range(n):
        t = i * dt
        db, da = quote_fn(s, q, t)
        if max_inventory is not None:
            if q >= max_inventory:
                db = np.inf
            if q <= -max_inventory:
                da = np.inf
        p_buy = A * np.exp(-k * db) * dt if np.isfinite(db) else 0.0
        p_sell = A * np.exp(-k * da) * dt if np.isfinite(da) else 0.0
        if rng.random() < p_buy:
            q += 1; cash -= s - db
        if rng.random() < p_sell:
            q -= 1; cash += s + da
        s = s + sigma * np.sqrt(dt) * rng.standard_normal()
        price[i + 1] = s; inv[i + 1] = q; pnl[i + 1] = cash + q * s
    return dict(pnl=pnl, inventory=inv, cash=cash, price=price, final_pnl=float(cash + q * s), final_inventory=int(q))


def as_strategy(gamma: float, sigma: float, k: float, T: float) -> Callable:
    """生成 AS 报价函数供 ``simulate_market_making`` 使用。"""
    def quote(s, q, t):
        out = avellaneda_stoikov_quotes(s, q, t, T, gamma, sigma, k)
        return out["delta_b"], out["delta_a"]
    return quote


def symmetric_strategy(delta: float) -> Callable:
    """对称固定价差策略（基准）。"""
    return lambda s, q, t: (delta, delta)


def glft_strategy(gamma: float, sigma: float, k: float, A: float, Q: int) -> Callable:
    """GLFT 渐近最优报价函数（库存限制 ±Q）。"""
    sol = glft_asymptotic_quotes(gamma, sigma, k, A, Q)
    qs = sol["q"]

    def quote(s, q, t):
        q = int(np.clip(q, -Q, Q))
        i = int(np.where(qs == q)[0][0])
        db = sol["delta_b"][i] if not np.isnan(sol["delta_b"][i]) else np.inf
        da = sol["delta_a"][i] if not np.isnan(sol["delta_a"][i]) else np.inf
        return db, da
    return quote
