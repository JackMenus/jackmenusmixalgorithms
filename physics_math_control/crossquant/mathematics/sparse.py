"""
稀疏恢复与压缩感知：OMP、LASSO/ISTA 与稀疏指数复制
===================================================

学科来源
--------
信号处理与调和分析（Mallat & Zhang 1993 的匹配追踪；Chen, Donoho & Saunders 1998 的基追踪；
Candès–Romberg–Tao 与 Donoho 2006 的压缩感知：少量测量精确恢复稀疏信号）。
LASSO（Tibshirani 1996）是统计学中的同一思想。

量化映射
--------
- **稀疏指数复制 / 篓子**：用 k 只股票跟踪指数（交易成本低、可实施）；
- **稀疏因子暴露 / alpha 组合**：在成百个候选因子里选少数（去除共线性）；
- **稀疏对冲**：用少量流动性好的工具对冲一个复杂头寸；
- **订单簿形状的稀疏表示**：少数基函数描述深度曲线。

参考文献
--------
- Tropp & Gilbert (2007). Signal Recovery From Random Measurements Via Orthogonal Matching Pursuit. IEEE IT.
- Beck & Teboulle (2009). A Fast Iterative Shrinkage-Thresholding Algorithm (FISTA). SIAM J. Imaging Sci.
- Brodie, Daubechies, De Mol, Giannone & Loris (2009). Sparse and stable Markowitz portfolios. PNAS.
- Benidis, Feng & Palomar (2018). Sparse Portfolios for High-Dimensional Financial Index Tracking. IEEE TSP.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def omp(A, y, k: int, tol: float = 1e-10) -> dict:
    """正交匹配追踪：选 k 个原子使 ‖y − A x‖ 最小。返回 ``coef``（稠密向量）、``support``、``residual_norm``。"""
    A = np.asarray(A, dtype=float); y = np.asarray(y, dtype=float)
    norms = np.linalg.norm(A, axis=0) + 1e-12
    r = y.copy(); support = []
    for _ in range(k):
        corr = np.abs(A.T @ r) / norms
        corr[support] = -np.inf
        j = int(np.argmax(corr)); support.append(j)
        coef_s, *_ = np.linalg.lstsq(A[:, support], y, rcond=None)
        r = y - A[:, support] @ coef_s
        if np.linalg.norm(r) < tol:
            break
    coef = np.zeros(A.shape[1]); coef[support] = coef_s
    return dict(coef=coef, support=np.array(support), residual_norm=float(np.linalg.norm(r)))


def lasso_ista(A, y, lam: float, n_iter: int = 2000, fista: bool = True, tol: float = 1e-9) -> np.ndarray:
    """LASSO  min ½‖y − Ax‖² + λ‖x‖₁ 的 ISTA/FISTA 求解。"""
    A = np.asarray(A, dtype=float); y = np.asarray(y, dtype=float)
    L = np.linalg.norm(A, 2) ** 2
    x = np.zeros(A.shape[1]); z = x.copy(); t = 1.0
    for _ in range(n_iter):
        grad = A.T @ (A @ z - y)
        x_new = z - grad / L
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - lam / L, 0.0)
        if fista:
            t_new = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
            z = x_new + (t - 1) / t_new * (x_new - x)
            t = t_new
        else:
            z = x_new
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return x


def sparse_index_tracking(asset_returns, index_returns, k: int, long_only: bool = True, sum_to_one: bool = True) -> dict:
    """稀疏指数复制：OMP 选 k 只成分，再在约束下做最小跟踪误差权重。

    返回 ``weights``（全维）、``support``、``tracking_error``（年化前的每期标准差）。
    """
    R = np.asarray(asset_returns, dtype=float); y = np.asarray(index_returns, dtype=float)
    sel = omp(R, y, k)["support"]
    Rs = R[:, sel]
    n = len(sel)

    def obj(w):
        e = Rs @ w - y
        return e @ e
    cons = [dict(type="eq", fun=lambda w: w.sum() - 1)] if sum_to_one else []
    bounds = [(0, 1)] * n if long_only else None
    res = minimize(obj, np.full(n, 1 / n), bounds=bounds, constraints=cons, method="SLSQP")
    w = np.zeros(R.shape[1]); w[sel] = res.x
    te = np.std(R @ w - y)
    return dict(weights=w, support=sel, tracking_error=float(te))
