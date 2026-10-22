"""
线性二次调节（LQR）与 Gârleanu–Pedersen 动态交易
=================================================

学科来源
--------
最优控制（Kalman 1960、Bellman 动态规划）。线性系统 + 二次代价 ⇒ 最优反馈控制是状态的线性函数，
增益由（离散）Riccati 方程给出。LQR 是“可解的随机控制”范式，许多交易问题都可以凑成这个形状。

量化映射
--------
1. **有交易成本的动态组合（Gârleanu & Pedersen 2013）**：收益由均值回复的因子预测，
   二次交易成本 ⇒ 最优策略是“向目标组合（aim portfolio）部分调仓”：
       x_t = x_{t-1} + Λ^{-1} A_xx (aim_t − x_{t-1})
   aim_t 是把 Markowitz 组合按因子衰减速度打折后的加权平均：衰减快的因子权重更低。
2. **执行进度跟踪**：把“剩余仓位 − 目标进度”当作状态，交易速率当作控制，LQR 给出平滑的参与率调节
   （Almgren–Chriss 本质上也是 LQ 问题）。
3. **库存控制**：做市商库存偏离目标的二次惩罚 + 调仓成本 ⇒ LQR 反馈。

核心公式
--------
离散 LQR：min Σ_t x_tᵀQx_t + u_tᵀRu_t,  x_{t+1} = A x_t + B u_t
Riccati： P = Q + Aᵀ P A − Aᵀ P B (R + Bᵀ P B)^{-1} Bᵀ P A,   u_t = −K x_t,  K = (R + BᵀPB)^{-1} BᵀPA

Gârleanu–Pedersen（离散）：
目标 max E Σ_t (1−ρ)^{t+1}(x_tᵀ r_{t+1} − γ/2 x_tᵀΣx_t) − (1−ρ)^t/2 Δx_tᵀΛΔx_t,
r_{t+1} = B f_t + u_{t+1}, Δf_{t+1} = −Φ f_t + ε_{t+1}。
值函数 V = −½xᵀA_xx x + xᵀA_xf f + …，A_xx 满足不动点  A_xx = Λ M^{-1} J,
J = (1−ρ)(γΣ + A_xx), M = Λ + J；aim_t = (γΣ + A_xx)^{-1}(B + A_xf(I−Φ)) f_t。
单资产、Λ = λΣ 时 A_xx = aΣ，a 为二次方程 (1−ρ)a² + a[λρ + γ(1−ρ)] − λγ(1−ρ) = 0 的正根。

参考文献
--------
- Gârleanu & Pedersen (2013). Dynamic Trading with Predictable Returns and Transaction Costs. JF.
- Gârleanu & Pedersen (2016). Dynamic Portfolio Choice with Frictions. JET（连续时间版本）。
- Anderson & Moore (1990). Optimal Control: Linear Quadratic Methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def lqr_discrete(A, B, Q, R) -> dict:
    """无穷时域离散 LQR：返回 ``K``（反馈增益，u = −Kx）、``P``（Riccati 解）、闭环特征值。"""
    A, B, Q, R = (np.atleast_2d(np.asarray(m, dtype=float)) for m in (A, B, Q, R))
    from scipy import linalg  # 延迟导入，保持模块加载轻量

    P = linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eig = np.linalg.eigvals(A - B @ K)
    return dict(K=K, P=P, closed_loop_eigs=eig)


def finite_horizon_lqr(A, B, Q, R, Qf, T: int) -> dict:
    """有限时域 LQR（后向 Riccati 递推）。返回 ``Ks``（T 个增益矩阵）与 ``Ps``。"""
    A, B, Q, R, Qf = (np.atleast_2d(np.asarray(m, dtype=float)) for m in (A, B, Q, R, Qf))
    n, m = B.shape
    Ps = np.zeros((T + 1, n, n)); Ks = np.zeros((T, m, n))
    Ps[T] = Qf
    for t in range(T - 1, -1, -1):
        P = Ps[t + 1]
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Ks[t] = K
        Ps[t] = Q + A.T @ P @ (A - B @ K)
    return dict(Ks=Ks, Ps=Ps)


def simulate_lqr(A, B, Ks, x0, T: int, noise_cov=None, seed=None) -> dict:
    """用（时变或定常）增益闭环仿真。``Ks`` 可为单个矩阵或 (T,m,n)。返回状态与控制轨迹。"""
    rng = np.random.default_rng(seed)
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    n = A.shape[0]
    x = np.asarray(x0, dtype=float).copy()
    xs = np.zeros((T + 1, n)); us = np.zeros((T, B.shape[1]))
    xs[0] = x
    for t in range(T):
        K = Ks[t] if np.ndim(Ks) == 3 else Ks
        u = -K @ x
        w = rng.multivariate_normal(np.zeros(n), noise_cov) if noise_cov is not None else 0.0
        x = A @ x + B @ u + w
        us[t], xs[t + 1] = u, x
    return dict(x=xs, u=us)


# --------------------------------------------------------------------------- 执行进度跟踪（LQ 形式）
def execution_tracking_lqr(T: int, risk_aversion: float, cost: float, terminal_penalty: float = 1e4) -> dict:
    """把“剩余仓位 x_t”用交易速率 u_t 推向 0 的 LQ 问题：

    x_{t+1} = x_t − u_t,  代价 Σ_t risk·x_t² + cost·u_t² + terminal·x_T²。
    返回每期增益 K_t 与从 x_0 = 1 出发的归一化清算轨迹（可与 Almgren–Chriss 对比：两者形状一致）。
    """
    out = finite_horizon_lqr(A=[[1.0]], B=[[-1.0]], Q=[[risk_aversion]], R=[[cost]], Qf=[[terminal_penalty]], T=T)
    sim = simulate_lqr([[1.0]], [[-1.0]], out["Ks"], [1.0], T)
    return dict(Ks=out["Ks"][:, 0, 0], trajectory=sim["x"][:, 0], trades=sim["u"][:, 0])


# --------------------------------------------------------------------------- Gârleanu–Pedersen
def gp_single_asset_rate(gamma: float, lam: float, rho: float = 0.0) -> float:
    """单资产 GP 闭式解：返回 A_xx = aΣ 中的 a；交易速率 = a/λ（每期向 aim 靠近的比例）。"""
    b = lam * rho + gamma * (1 - rho)
    a = (-b + np.sqrt(b ** 2 + 4 * gamma * lam * (1 - rho) ** 2)) / (2 * (1 - rho))
    return float(a)


@dataclass
class GarleanuPedersen:
    """Gârleanu–Pedersen (2013) 多资产、多因子动态交易策略。

    参数
    ----
    Sigma : (n,n) 收益协方差
    Lambda: (n,n) 二次交易成本矩阵（成本 = ½ ΔxᵀΛΔx），常取 λΣ
    B     : (n,k) 因子载荷，E[r_{t+1}] = B f_t
    Phi   : (k,k) 因子均值回复矩阵，Δf = −Φ f + ε（对角元 φ_i ∈ (0,1)）
    gamma : 风险厌恶
    rho   : 折现率（折现因子 1−ρ）
    """

    Sigma: np.ndarray
    Lambda: np.ndarray
    B: np.ndarray
    Phi: np.ndarray
    gamma: float
    rho: float = 0.0

    def __post_init__(self):
        self.Sigma = np.atleast_2d(np.asarray(self.Sigma, dtype=float))
        self.Lambda = np.atleast_2d(np.asarray(self.Lambda, dtype=float))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=float))
        self.Phi = np.atleast_2d(np.asarray(self.Phi, dtype=float))
        self.n, self.k = self.B.shape
        self._solve()

    def _solve(self, tol: float = 1e-12, max_iter: int = 10000):
        n, rho = self.n, self.rho
        Axx = np.zeros((n, n))
        for _ in range(max_iter):
            J = (1 - rho) * (self.gamma * self.Sigma + Axx)
            M = self.Lambda + J
            Axx_new = self.Lambda @ np.linalg.solve(M, J)
            Axx_new = 0.5 * (Axx_new + Axx_new.T)
            if np.max(np.abs(Axx_new - Axx)) < tol:
                Axx = Axx_new
                break
            Axx = Axx_new
        self.A_xx = Axx
        J = (1 - rho) * (self.gamma * self.Sigma + Axx)
        M = self.Lambda + J
        W = np.linalg.solve(M, self.Lambda)              # W = M^{-1} Λ
        # 求 X = A_xfᵀ (k×n)： X − (1−ρ)(I−Φ)ᵀ X W = (1−ρ) Bᵀ W
        IPhi = np.eye(self.k) - self.Phi
        lhs = np.eye(self.k * n) - (1 - rho) * np.kron(W.T, IPhi.T)   # vec(AXB) = (Bᵀ⊗A)vec(X)
        rhs = ((1 - rho) * self.B.T @ W).reshape(-1, order="F")
        X = np.linalg.solve(lhs, rhs).reshape(self.k, n, order="F")
        self.A_xf = X.T
        self.trade_matrix = np.linalg.solve(self.Lambda, self.A_xx)   # Λ^{-1} A_xx
        self._aim_matrix = np.linalg.solve(self.gamma * self.Sigma + self.A_xx,
                                           self.B + self.A_xf @ IPhi)

    def markowitz(self, f) -> np.ndarray:
        """无成本的静态 Markowitz 组合 (γΣ)^{-1} B f。"""
        return np.linalg.solve(self.gamma * self.Sigma, self.B @ np.asarray(f, dtype=float))

    def aim(self, f) -> np.ndarray:
        """目标组合 aim_t = (γΣ + A_xx)^{-1}(B + A_xf(I−Φ)) f_t。"""
        return self._aim_matrix @ np.asarray(f, dtype=float)

    def step(self, x_prev, f) -> np.ndarray:
        """最优持仓 x_t = x_{t−1} + Λ^{-1}A_xx (aim_t − x_{t−1})。"""
        x_prev = np.asarray(x_prev, dtype=float)
        return x_prev + self.trade_matrix @ (self.aim(f) - x_prev)

    def simulate(self, f_path, x0=None) -> np.ndarray:
        """给定因子路径 (T,k)，返回持仓路径 (T,n)。"""
        f_path = np.atleast_2d(np.asarray(f_path, dtype=float))
        x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float)
        xs = np.zeros((len(f_path), self.n))
        for t, f in enumerate(f_path):
            x = self.step(x, f)
            xs[t] = x
        return xs
