"""
模型预测控制（MPC）与多期组合优化
=================================

学科来源
--------
过程控制（化工、炼油，1970s 起）。MPC 在每个时刻求解一段有限时域的开环最优控制问题，
只执行第一步，下一时刻用新观测重新求解（滚动时域 / receding horizon）。
它能自然处理约束（上下界、预算），是工业界使用最广的高级控制方法。

量化映射
--------
Boyd 等 (2017) *Multi-Period Trading via Convex Optimization* 把组合管理写成 MPC：
    max  Σ_{h=1}^{H} [ μ̂_hᵀ w_h − γ w_hᵀ Σ w_h − 交易成本(w_h − w_{h−1}) − 持有成本 ]
    s.t. 1ᵀw_h = 1（满仓）、w_h ≥ 0（可选多头约束）、‖w_h‖₁ ≤ L（杠杆）
每期只执行 w_1 − w_0。相比单期均值方差，MPC 会“预见”未来信号的衰减与成本，提前分摊调仓。
执行层同样可以用 MPC 做“进度跟踪 + 参与率上限 + 冲击成本”的滚动优化。

实现说明
--------
为避免依赖 cvxpy，这里用 ``scipy.optimize.minimize``(SLSQP) 求解；线性成本 κ|Δw| 用
sqrt(Δw² + ε²) 做光滑近似。规模（资产数 × 时域）在几百变量内很快；更大规模建议换 cvxpy/OSQP。

参考文献
--------
- Boyd, Busseti, Diamond, Kahn, Koh, Nystrup & Speth (2017). Multi-Period Trading via Convex Optimization. FnT Optimization.
- Rawlings, Mayne & Diehl (2017). Model Predictive Control: Theory, Computation, and Design.
- Herzog, Dondi & Geering (2007). Stochastic Model Predictive Control and Portfolio Optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import optimize


@dataclass
class MultiPeriodMPC:
    """多期组合 MPC（Boyd 等 2017 的简化实现）。

    参数
    ----
    Sigma : (n,n) 协方差
    gamma : 风险厌恶
    tc_linear : 线性交易成本系数（每单位换手），标量或 (n,)
    tc_quadratic : 二次交易成本系数（冲击），标量或 (n,)
    long_only : 是否禁止做空
    fully_invested : 是否要求 Σw = 1（否则 Σw ≤ 1，剩余为现金）
    leverage : ‖w‖₁ 上限
    discount : 未来各期目标的折现（<1 时更重视近期）
    smooth_eps : L1 光滑参数
    """

    Sigma: np.ndarray
    gamma: float = 1.0
    tc_linear: float | np.ndarray = 0.0
    tc_quadratic: float | np.ndarray = 0.0
    long_only: bool = True
    fully_invested: bool = True
    leverage: float = 1.0
    discount: float = 1.0
    smooth_eps: float = 1e-6

    def __post_init__(self):
        self.Sigma = np.atleast_2d(np.asarray(self.Sigma, dtype=float))
        self.n = self.Sigma.shape[0]
        self.kl = np.broadcast_to(np.asarray(self.tc_linear, dtype=float), (self.n,))
        self.kq = np.broadcast_to(np.asarray(self.tc_quadratic, dtype=float), (self.n,))

    def _objective(self, W, w0, mu_path):
        H = mu_path.shape[0]
        W = W.reshape(H, self.n)
        prev = w0
        total = 0.0
        for h in range(H):
            w = W[h]
            dw = w - prev
            disc = self.discount ** h
            total += disc * (mu_path[h] @ w - self.gamma * w @ self.Sigma @ w
                             - self.kl @ np.sqrt(dw ** 2 + self.smooth_eps ** 2) - self.kq @ dw ** 2)
            prev = w
        return -total

    def plan(self, w0, mu_path, w_init: Optional[np.ndarray] = None) -> dict:
        """求解整段时域的最优权重路径。

        参数
        ----
        w0 : (n,) 当前权重
        mu_path : (H, n) 未来 H 期的预期收益预测（第 0 行为下一期）

        返回 dict：``weights (H,n)``、``next_weights (n,)``（滚动时域下真正执行的那一步）、``trade``、``success``
        """
        w0 = np.asarray(w0, dtype=float)
        mu_path = np.atleast_2d(np.asarray(mu_path, dtype=float))
        H = mu_path.shape[0]
        x0 = np.tile(w0, H) if w_init is None else np.asarray(w_init).reshape(-1)
        cons = []
        for h in range(H):
            idx = slice(h * self.n, (h + 1) * self.n)
            if self.fully_invested:
                cons.append(dict(type="eq", fun=lambda W, idx=idx: np.sum(W[idx]) - 1.0))
            else:
                cons.append(dict(type="ineq", fun=lambda W, idx=idx: 1.0 - np.sum(W[idx])))
            if not self.long_only:
                cons.append(dict(type="ineq", fun=lambda W, idx=idx: self.leverage - np.sum(np.sqrt(W[idx] ** 2 + 1e-12))))
        bounds = [(0.0, self.leverage) if self.long_only else (-self.leverage, self.leverage)] * (H * self.n)
        res = optimize.minimize(self._objective, x0, args=(w0, mu_path), method="SLSQP",
                                bounds=bounds, constraints=cons, options=dict(maxiter=500, ftol=1e-10))
        W = res.x.reshape(H, self.n)
        return dict(weights=W, next_weights=W[0], trade=W[0] - w0, success=bool(res.success), objective=-res.fun)

    def run(self, w0, mu_forecaster: Callable[[int], np.ndarray], T: int, realized_returns=None) -> dict:
        """滚动时域回测：每期调用 ``mu_forecaster(t)`` 得到 (H,n) 预测，执行第一步。

        若给出 ``realized_returns (T,n)``，返回组合收益（扣除交易成本）。
        """
        w = np.asarray(w0, dtype=float)
        weights = np.zeros((T, self.n)); pnl = np.zeros(T)
        for t in range(T):
            out = self.plan(w, mu_forecaster(t))
            dw = out["next_weights"] - w
            w = out["next_weights"]
            weights[t] = w
            if realized_returns is not None:
                cost = self.kl @ np.abs(dw) + self.kq @ dw ** 2
                pnl[t] = w @ realized_returns[t] - cost
        return dict(weights=weights, pnl=pnl)
