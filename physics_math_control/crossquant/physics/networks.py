"""
复杂网络与层次结构：最小生成树、层次风险平价
=============================================

学科来源
--------
统计物理的复杂网络（Mantegna 1999 用相关距离 d = sqrt(2(1−ρ)) 构造股票的最小生成树，
发现其拓扑与行业分类一致）；Onnela 等 (2003) 发现危机时 MST 收缩（“树长”变短）。
层次风险平价（López de Prado 2016）把层次聚类的树结构用于资金分配，回避协方差求逆。

量化映射
--------
- **风险分配**：HRP 权重样本外比均值方差更稳；
- **系统性风险监控**：MST 归一化树长、最大特征值吸收比、网络密度；
- **板块发现 / 相关簇**：树的分支 = 隐含行业；
- **对冲篮子选择**：树上距离近的资产适合配对/替代。

参考文献
--------
- Mantegna (1999). Hierarchical structure in financial markets. EPJB.
- Onnela, Chakraborti, Kaski, Kertész & Kanto (2003). Dynamics of market correlations: Taxonomy and portfolio analysis. PRE.
- López de Prado (2016). Building Diversified Portfolios that Outperform Out of Sample. JPM.
- Tumminello, Aste, Di Matteo & Mantegna (2005). A tool for filtering information in complex systems (PMFG). PNAS.
- Maillard, Roncalli & Teïletche (2010). The properties of equally weighted risk contribution portfolios. JPM.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .._utils import cov_to_corr


def correlation_distance(corr) -> np.ndarray:
    """Mantegna 距离 d_ij = sqrt(2(1 − ρ_ij))，是一个合法的欧氏度量。"""
    corr = np.clip(np.asarray(corr, dtype=float), -1, 1)
    d = np.sqrt(np.maximum(2 * (1 - corr), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def minimum_spanning_tree(dist) -> dict:
    """Prim 算法求 MST。返回 ``edges``（(i, j, w) 列表）、``adjacency``（权重矩阵）、``total_length``、``degree``。"""
    D = np.asarray(dist, dtype=float)
    n = D.shape[0]
    in_tree = np.zeros(n, dtype=bool); in_tree[0] = True
    best = D[0].copy(); parent = np.zeros(n, dtype=int)
    edges = []
    adj = np.zeros((n, n))
    for _ in range(n - 1):
        cand = np.where(~in_tree, best, np.inf)
        j = int(np.argmin(cand)); i = int(parent[j])
        edges.append((i, j, float(D[i, j])))
        adj[i, j] = adj[j, i] = D[i, j]
        in_tree[j] = True
        update = (D[j] < best) & ~in_tree
        best[update] = D[j][update]; parent[update] = j
    degree = (adj > 0).sum(axis=1)
    return dict(edges=edges, adjacency=adj, total_length=float(sum(e[2] for e in edges)),
                normalized_length=float(sum(e[2] for e in edges) / (n - 1)), degree=degree)


def tree_closeness_centrality(adjacency) -> np.ndarray:
    """树上的接近中心性（基于跳数 BFS）。"""
    A = np.asarray(adjacency) > 0
    n = A.shape[0]
    cent = np.zeros(n)
    for s in range(n):
        dist = np.full(n, -1); dist[s] = 0; queue = [s]
        while queue:
            u = queue.pop(0)
            for v in np.where(A[u])[0]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1; queue.append(v)
        cent[s] = (n - 1) / dist[dist > 0].sum() if n > 1 else 0
    return cent


# --------------------------------------------------------------------------- HRP
def _quasi_diag(link):
    return hierarchy.leaves_list(link)


def _ivp(cov):
    iv = 1.0 / np.diag(cov)
    return iv / iv.sum()


def _cluster_var(cov, idx):
    sub = cov[np.ix_(idx, idx)]
    w = _ivp(sub)
    return float(w @ sub @ w)


def hierarchical_risk_parity(cov, linkage_method: str = "single") -> dict:
    """层次风险平价（HRP）。返回 ``weights``、``order``（准对角化顺序）、``linkage``。"""
    cov = np.asarray(cov, dtype=float)
    corr = cov_to_corr(cov)
    dist = correlation_distance(corr)
    link = hierarchy.linkage(squareform(dist, checks=False), method=linkage_method)
    order = list(_quasi_diag(link))
    w = np.ones(cov.shape[0])
    clusters = [order]
    while clusters:
        new = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            c1, c2 = c[:half], c[half:]
            v1, v2 = _cluster_var(cov, c1), _cluster_var(cov, c2)
            a = 1 - v1 / (v1 + v2)
            w[c1] *= a; w[c2] *= 1 - a
            new += [c1, c2]
        clusters = new
    return dict(weights=w / w.sum(), order=np.array(order), linkage=link)


def inverse_variance_weights(cov) -> np.ndarray:
    return _ivp(np.asarray(cov, dtype=float))


def risk_parity_weights(cov, budget=None) -> np.ndarray:
    """等风险贡献（ERC）组合：求 w 使每个资产的风险贡献 w_i (Σw)_i 相等（或按 budget 比例）。"""
    cov = np.asarray(cov, dtype=float); n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, dtype=float) / np.sum(budget)

    def obj(w):
        port_var = w @ cov @ w
        rc = w * (cov @ w) / port_var
        return np.sum((rc - b) ** 2)
    res = minimize(obj, np.full(n, 1.0 / n), bounds=[(1e-6, 1)] * n,
                   constraints=[dict(type="eq", fun=lambda w: w.sum() - 1)], method="SLSQP", options=dict(ftol=1e-14, maxiter=1000))
    return res.x / res.x.sum()


def risk_contributions(w, cov) -> np.ndarray:
    """各资产风险贡献占比。"""
    w = np.asarray(w, dtype=float); cov = np.asarray(cov, dtype=float)
    return w * (cov @ w) / (w @ cov @ w)


def correlation_clusters(corr, n_clusters: int, linkage_method: str = "average") -> np.ndarray:
    """按相关距离做层次聚类，返回每个资产的簇标签（1..n_clusters）。"""
    dist = correlation_distance(corr)
    link = hierarchy.linkage(squareform(dist, checks=False), method=linkage_method)
    return hierarchy.fcluster(link, n_clusters, criterion="maxclust")
