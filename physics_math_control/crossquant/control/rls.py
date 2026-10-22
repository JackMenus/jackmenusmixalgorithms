"""
递推最小二乘（RLS）与自适应参数估计
===================================

学科来源
--------
自适应控制 / 系统辨识（Åström & Wittenmark）。带遗忘因子 λ 的 RLS 是“在线加权最小二乘”，
等价于状态噪声特殊结构下的 Kalman 滤波，但只有一个超参数，极易部署。

量化映射
--------
- 在线 β / 对冲比率、在线因子载荷；
- 自适应线性预测（下一期收益对特征的回归系数随时间漂移）；
- 延迟模型的在线回归（延迟对报文大小、时段等特征）。

公式
----
    e_t = y_t − θ_{t−1}ᵀ x_t
    K_t = P_{t−1} x_t / (λ + x_tᵀ P_{t−1} x_t)
    θ_t = θ_{t−1} + K_t e_t
    P_t = (P_{t−1} − K_t x_tᵀ P_{t−1}) / λ
"""
from __future__ import annotations

import numpy as np


class RecursiveLeastSquares:
    """带遗忘因子的 RLS。``forgetting`` 越小遗忘越快（有效窗口 ≈ 1/(1−λ)）。"""

    def __init__(self, dim: int, forgetting: float = 0.99, delta: float = 100.0):
        self.theta = np.zeros(dim); self.P = delta * np.eye(dim); self.lam = forgetting

    def update(self, x, y: float) -> float:
        """更新并返回先验预测误差 e_t。"""
        x = np.asarray(x, dtype=float)
        e = y - self.theta @ x
        Px = self.P @ x
        k = Px / (self.lam + x @ Px)
        self.theta = self.theta + k * e
        self.P = (self.P - np.outer(k, Px)) / self.lam
        return float(e)

    def predict(self, x) -> float:
        return float(self.theta @ np.asarray(x, dtype=float))


def adaptive_regression(X, y, forgetting: float = 0.99, delta: float = 100.0) -> dict:
    """对整段数据跑 RLS，返回 ``theta_path (T,d)`` 与 ``errors (T,)``。"""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    rls = RecursiveLeastSquares(X.shape[1], forgetting, delta)
    thetas = np.zeros_like(X); errs = np.zeros(len(y))
    for t in range(len(y)):
        errs[t] = rls.update(X[t], y[t]); thetas[t] = rls.theta
    return dict(theta_path=thetas, errors=errs)
