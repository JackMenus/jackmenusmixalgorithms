"""
流形学习与谱聚类：扩散映射
==========================

学科来源
--------
调和分析与谱图论（Coifman & Lafon 2006 的扩散映射；Belkin & Niyogi 2003 的 Laplacian 特征映射；
Ng, Jordan & Weiss 2002 的谱聚类）。核心：数据上的随机游走的（图拉普拉斯）特征函数给出内在坐标。

量化映射
--------
- **板块 / 风格发现**：相关矩阵 → 亲和图 → 谱聚类，比行业分类更贴近实际共动；
- **低维市场状态**：扩散映射把高维横截面压成几个“扩散坐标”，作为 regime 变量；
- **因子相似度图**：去除冗余因子、构造正交化顺序。

参考文献
--------
- Coifman & Lafon (2006). Diffusion maps. Applied and Computational Harmonic Analysis.
- Ng, Jordan & Weiss (2002). On Spectral Clustering: Analysis and an algorithm. NeurIPS.
- von Luxburg (2007). A tutorial on spectral clustering. Statistics and Computing.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2


def gaussian_affinity(X, epsilon: float | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    epsilon = np.median(D2[D2 > 0]) if epsilon is None else epsilon
    return np.exp(-D2 / epsilon)


def diffusion_maps(X, n_components: int = 2, epsilon: float | None = None, t: int = 1, alpha: float = 0.0) -> dict:
    """扩散映射。alpha=0 为经典归一化（图拉普拉斯），alpha=1 近似 Laplace–Beltrami。返回 ``embedding``、``eigenvalues``。"""
    K = gaussian_affinity(X, epsilon)
    d = K.sum(axis=1)
    if alpha > 0:
        K = K / np.outer(d ** alpha, d ** alpha); d = K.sum(axis=1)
    Ms = K / np.sqrt(np.outer(d, d))
    w, V = np.linalg.eigh(Ms)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    psi = V / np.sqrt(d)[:, None]
    emb = psi[:, 1:n_components + 1] * (w[1:n_components + 1] ** t)[None, :]
    return dict(embedding=emb, eigenvalues=w[:n_components + 1])


def spectral_clustering(affinity, k: int, seed=None) -> np.ndarray:
    """Ng–Jordan–Weiss 归一化谱聚类，返回标签。"""
    A = np.asarray(affinity, dtype=float).copy(); np.fill_diagonal(A, 0.0)
    d = A.sum(axis=1) + 1e-12
    L = np.eye(len(A)) - A / np.sqrt(np.outer(d, d))
    w, V = np.linalg.eigh(L)
    U = V[:, :k]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    _, labels = kmeans2(U, k, minit="++", seed=rng)
    return labels


def correlation_affinity(corr, epsilon: float | None = None) -> np.ndarray:
    """由相关矩阵构造亲和矩阵 exp(−d²/ε)，d = sqrt(2(1−ρ))。"""
    corr = np.clip(np.asarray(corr, dtype=float), -1, 1)
    D2 = 2 * (1 - corr)
    epsilon = np.median(D2[D2 > 0]) if epsilon is None else epsilon
    return np.exp(-D2 / epsilon)


def asset_clusters(corr, k: int, seed=None) -> np.ndarray:
    """相关矩阵 → 谱聚类标签（板块/风格发现）。"""
    return spectral_clustering(correlation_affinity(corr), k, seed)
