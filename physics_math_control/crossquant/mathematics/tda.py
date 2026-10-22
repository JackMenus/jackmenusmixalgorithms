"""
拓扑数据分析（TDA）：Vietoris–Rips 持续同调与持续景观
======================================================

学科来源
--------
代数拓扑（Poincaré 的同调群）与计算拓扑（Edelsbrunner, Letscher & Zomorodian 2002 的持续同调；
Carlsson 2009）。在点云上按尺度 ε 增长的 Vietoris–Rips 复形里追踪“连通分支”（H₀）和“环”（H₁）
的出生与消亡，持续图 (birth, death) 是对数据形状的稳定、坐标无关的描述（Cohen-Steiner 等 2007 的稳定性定理）。

量化映射
--------
- **崩盘预警**（Gidea & Katz 2018）：对若干指数的滑动窗口收益点云计算 H₁ 持续景观的 L¹/L² 范数，
  在 2000、2008 危机前范数显著上升；
- **市场结构变化**：相关结构变化会改变点云形状，比协方差统计更稳健于异常值；
- **策略/因子的拓扑特征**：持续图的统计量作为非线性特征。

算法
----
H₀：按边长排序的 Kruskal/并查集，合并时记录死亡时间；
H₁：三角形（按最长边过滤值排序）的边界矩阵在 Z₂ 上做标准列约简（Python 大整数作位集），
    约简后非零列的最低位边（正边，创造环）与该三角形配对 ⇒ (birth = 边长, death = 三角过滤值)。
持续景观 λ_k(t) = k-th max{ min(t − b, d − t)₊ }，L^p 范数在网格上数值积分。

参考文献
--------
- Edelsbrunner & Harer (2010). Computational Topology: An Introduction. AMS.
- Bubenik (2015). Statistical topological data analysis using persistence landscapes. JMLR.
- Gidea & Katz (2018). Topological data analysis of financial time series: Landscapes of crashes. Physica A.
- (2025). TDA for stock market crash detection. Computers 14(10):408.
"""
from __future__ import annotations

import numpy as np


def _distance_matrix(points):
    P = np.asarray(points, dtype=float)
    return np.sqrt(np.maximum(np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=2), 0.0))


def vietoris_rips_persistence(points, max_dim: int = 1, max_edge: float | None = None) -> dict:
    """点云 (n, d) 的 Vietoris–Rips 持续同调（H₀ 与可选 H₁）。

    返回 dict {0: (k,2) 数组, 1: (k,2) 数组}；无限持续的 H₀ 分量死亡时间记为 ``max_edge``（缺省为最大边长）。
    n ≲ 80 时速度可接受（三角形数 O(n³)）。
    """
    D = _distance_matrix(points)
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    lengths = D[iu]
    max_edge = float(lengths.max()) if max_edge is None else max_edge
    order = np.argsort(lengths, kind="stable")
    edges = [(iu[0][k], iu[1][k], lengths[k]) for k in order if lengths[k] <= max_edge]
    # ---- H0：并查集
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    h0 = []
    edge_index = {}
    positive_edges = []
    for idx, (i, j, w) in enumerate(edges):
        edge_index[(i, j)] = idx
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            h0.append((0.0, w))
        else:
            positive_edges.append(idx)
    n_comp = len(set(find(a) for a in range(n)))
    h0 += [(0.0, max_edge)] * n_comp
    diagrams = {0: np.array(h0) if h0 else np.zeros((0, 2))}
    if max_dim < 1:
        return diagrams
    # ---- H1：三角形边界约简
    tri = []
    for a in range(n):
        for b in range(a + 1, n):
            dab = D[a, b]
            if dab > max_edge:
                continue
            for c in range(b + 1, n):
                f = max(dab, D[a, c], D[b, c])
                if f <= max_edge:
                    tri.append((f, a, b, c))
    tri.sort(key=lambda t: t[0])
    pivots = {}
    paired_edges = set()
    h1 = []
    for f, a, b, c in tri:
        col = (1 << edge_index[(a, b)]) | (1 << edge_index[(a, c)]) | (1 << edge_index[(b, c)])
        while col:
            low = col.bit_length() - 1
            if low in pivots:
                col ^= pivots[low]
            else:
                pivots[low] = col
                birth = edges[low][2]
                if f > birth + 1e-12:
                    h1.append((birth, f))
                paired_edges.add(low)
                break
    for e in positive_edges:
        if e not in paired_edges:
            h1.append((edges[e][2], max_edge))
    diagrams[1] = np.array(h1) if h1 else np.zeros((0, 2))
    return diagrams


def persistence_landscape(diagram, k_max: int = 3, grid=None) -> tuple[np.ndarray, np.ndarray]:
    """持续景观 λ_1..λ_{k_max} 在网格上的取值，返回 (grid, landscapes (k_max, len(grid)))。"""
    dg = np.asarray(diagram, dtype=float).reshape(-1, 2)
    if grid is None:
        hi = dg[:, 1].max() if len(dg) else 1.0
        grid = np.linspace(0, hi, 200)
    if len(dg) == 0:
        return grid, np.zeros((k_max, len(grid)))
    tents = np.maximum(np.minimum(grid[None, :] - dg[:, :1], dg[:, 1:] - grid[None, :]), 0.0)  # (k_pts, grid)
    tents_sorted = -np.sort(-tents, axis=0)
    L = np.zeros((k_max, len(grid)))
    m = min(k_max, len(dg))
    L[:m] = tents_sorted[:m]
    return grid, L


def landscape_norm(diagram, p: float = 1.0, k_max: int = 3, grid=None) -> float:
    """持续景观的 L^p 范数（Gidea–Katz 的崩盘指标）。"""
    grid, L = persistence_landscape(diagram, k_max, grid)
    val = np.trapezoid(np.abs(L) ** p, grid, axis=1).sum()
    return float(val ** (1 / p))


def total_persistence(diagram, p: float = 1.0) -> float:
    dg = np.asarray(diagram, dtype=float).reshape(-1, 2)
    pers = dg[:, 1] - dg[:, 0]
    pers = pers[np.isfinite(pers)]
    return float(np.sum(pers ** p) ** (1 / p)) if len(pers) else 0.0


def tda_crash_indicator(returns, window: int = 50, step: int = 1, p_list=(1.0, 2.0), max_edge: float | None = None) -> dict:
    """滑动窗口点云（每行一个时刻、每列一个指数的收益）的 H₁ 持续景观范数时间序列。"""
    R = np.asarray(returns, dtype=float)
    R = R[:, None] if R.ndim == 1 else R
    idx = []; norms = {p: [] for p in p_list}
    for end in range(window, len(R) + 1, step):
        seg = R[end - window:end]
        dg = vietoris_rips_persistence(seg, max_dim=1, max_edge=max_edge)[1]
        idx.append(end - 1)
        for p in p_list:
            norms[p].append(landscape_norm(dg, p))
    return dict(index=np.array(idx), **{f"L{int(p)}": np.array(v) for p, v in norms.items()})
