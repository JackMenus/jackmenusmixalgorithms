"""
Koopman 算子、动态模态分解（DMD）与奇异谱分析（SSA）
=====================================================

学科来源
--------
Koopman (1931) 证明非线性动力系统可以在可观测函数空间上表示为线性算子；Schmid (2010) 在流体力学中
提出 DMD，从快照数据直接提取该算子的有限维近似（Rowley 等 2009 建立与 Koopman 的联系）；
Hankel/时延 DMD（Brunton 等 2017，HAVOK）用 Takens 嵌入处理只有标量观测的情形。
SSA（Broomhead & King 1986；Golyandina 等 2001）是 Hankel 矩阵 SVD 的信号分解版本。

量化映射
--------
- **线性可解释的多资产预测**：DMD 的特征值给出各模态的增长率与频率，模态给出横截面结构（Mann & Kutz 2016）；
- **趋势/周期分解**：SSA 把价格分成趋势 + 振荡 + 噪声，比移动平均延迟小；
- **Koopman 式特征**：时延嵌入 + 线性算子 = 可解释的“动力学因子”，近年与 Transformer 结合（DeepKoopFormer 2025 等）。

核心公式
--------
X' ≈ A X；经 X = UΣVᵀ 截断：Ã = Uᵀ X' V Σ^{-1}，特征对 (λ, w)，模态 Φ = X' V Σ^{-1} W；
预测 x_t ≈ Φ diag(λ^t) b，b = Φ⁺ x_0。
SSA：轨迹矩阵（Hankel）→ SVD → 分组 → 对角平均重构。

参考文献
--------
- Koopman (1931). Hamiltonian systems and transformation in Hilbert space. PNAS.
- Schmid (2010). Dynamic mode decomposition of numerical and experimental data. JFM.
- Tu, Rowley, Luchtenburg, Brunton & Kutz (2014). On dynamic mode decomposition: Theory and applications. JCD.
- Mann & Kutz (2016). Dynamic mode decomposition for financial trading strategies. QF. arXiv:1508.04487
- Brunton, Brunton, Proctor, Kaiser & Kutz (2017). Chaos as an intermittently forced linear system. Nat. Commun.
- Forootani, Khosravi & Barati (2025). DeepKoopFormer. arXiv:2508.02616；(2026) Learnable Koopman-Enhanced Transformer. arXiv:2602.02592
"""
from __future__ import annotations

import numpy as np


def dmd(X, rank: int | None = None, dt: float = 1.0) -> dict:
    """精确 DMD（Tu 等 2014）。X 形状 (n_features, T)。

    返回 ``eigs``（离散特征值）、``omega``（连续：log λ/dt）、``modes (n, r)``、``amplitudes``、``A_tilde``。
    """
    X = np.asarray(X, dtype=float)
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    r = len(s) if rank is None else min(rank, len(s))
    U, s, V = U[:, :r], s[:r], Vh[:r].T
    A_tilde = U.T @ X2 @ V / s
    eigs, W = np.linalg.eig(A_tilde)
    modes = X2 @ V / s @ W
    b = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]
    return dict(eigs=eigs, omega=np.log(eigs.astype(complex)) / dt, modes=modes, amplitudes=b, A_tilde=A_tilde, rank=r)


def dmd_predict(model: dict, n_steps: int, x0=None) -> np.ndarray:
    """由 DMD 模型外推 n_steps 步（从 x0 或拟合时的初值出发），返回实部 (n_features, n_steps+1)。"""
    Phi, lam = model["modes"], model["eigs"]
    b = model["amplitudes"] if x0 is None else np.linalg.lstsq(Phi, np.asarray(x0, dtype=float), rcond=None)[0]
    t = np.arange(n_steps + 1)
    dyn = (lam[:, None] ** t[None, :]) * b[:, None]
    return (Phi @ dyn).real


def hankel_matrix(x, delay: int) -> np.ndarray:
    """标量/多维序列的 Hankel（轨迹）矩阵：(delay·d, T−delay+1)。"""
    X = np.asarray(x, dtype=float)
    X = X[:, None] if X.ndim == 1 else X
    T, d = X.shape
    cols = T - delay + 1
    return np.vstack([X[i:i + cols].T for i in range(delay)])


def hankel_dmd(x, delay: int = 10, rank: int | None = None) -> dict:
    """时延嵌入 + DMD（HAVOK 风格）。返回 DMD 模型及 ``predict(h)``（返回未来 h 步的最后一行观测）。"""
    X = np.asarray(x, dtype=float)
    H = hankel_matrix(X, delay)
    model = dmd(H, rank)
    d = 1 if X.ndim == 1 else X.shape[1]

    def predict(h: int):
        fut = dmd_predict(model, h, x0=H[:, -1])
        return fut[-d:, 1:].T.squeeze()
    model["predict"] = predict
    model["hankel"] = H
    return model


def ssa(x, window: int, n_components: int | None = None, groups: list | None = None) -> dict:
    """奇异谱分析。返回 ``components (k, T)``（对角平均重构）、``singular_values``、``trend``（第一分量）。"""
    x = np.asarray(x, dtype=float)
    T = len(x); L = window; K = T - L + 1
    H = np.column_stack([x[i:i + L] for i in range(K)])
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    k = len(s) if n_components is None else min(n_components, len(s))
    groups = [[i] for i in range(k)] if groups is None else groups
    comps = []
    for g in groups:
        Hg = sum(s[i] * np.outer(U[:, i], Vh[i]) for i in g)
        comps.append(_diagonal_average(Hg, T))
    comps = np.array(comps)
    return dict(components=comps, singular_values=s, trend=comps[0], contribution=s ** 2 / np.sum(s ** 2))


def _diagonal_average(H, T):
    L, K = H.shape
    out = np.zeros(T); cnt = np.zeros(T)
    for i in range(L):
        out[i:i + K] += H[i]; cnt[i:i + K] += 1
    return out / cnt


def ssa_forecast(x, window: int, n_components: int, horizon: int) -> np.ndarray:
    """SSA 递推预测（Golyandina 等 2001 的 R-forecasting）。"""
    x = np.asarray(x, dtype=float)
    T = len(x); L = window; K = T - L + 1
    H = np.column_stack([x[i:i + L] for i in range(K)])
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    P = U[:, :n_components]
    pi = P[-1]; nu2 = pi @ pi
    R = (P[:-1] @ pi) / (1 - nu2)
    rec = ssa(x, window, groups=[list(range(n_components))])["components"][0]
    series = list(rec)
    for _ in range(horizon):
        series.append(R @ np.array(series[-(L - 1):]))
    return np.array(series[T:])
