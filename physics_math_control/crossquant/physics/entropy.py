"""
熵、互信息与传递熵：信息论工具箱
=================================

学科来源
--------
统计物理（Boltzmann–Gibbs 熵）与信息论（Shannon 1948）。Rényi、Tsallis 熵是对非广延系统的推广；
传递熵（Schreiber 2000）用条件互信息度量“X 的过去对 Y 的未来的额外预测力”，是方向性的、非线性的
Granger 因果；排列熵（Bandt & Pompe 2002）、样本熵（Richman & Moorman 2000）来自生理信号的复杂度分析。

量化映射
--------
- **因子筛选**：互信息（KSG 估计）衡量因子与未来收益的非线性依赖，比 IC 更通用；
- **领先滞后 / 信息流网络**：传递熵矩阵 TE_{i→j} 给出资产间信息流方向（Dimpfl & Peter 2013；Marschinski & Kantz 2002）；
- **市场效率 / 可预测性**：排列熵、样本熵越低，序列越“有结构”；
- **订单流复杂度**：订单类型序列的熵率变化可提示算法交易者行为切换。
- **最大熵原理**：在仅知部分矩的情况下选最无偏的分布（Jaynes 1957）——分布估计与组合多样化。

核心公式
--------
H(X) = −Σ p log p；   H_α = log Σ p^α /(1−α)；   S_q = (1 − Σ p^q)/(q−1)
I(X;Y) = Σ p(x,y) log [p(x,y)/(p(x)p(y))]；KSG：I = ψ(k) + ψ(N) − ⟨ψ(n_x+1) + ψ(n_y+1)⟩
TE_{X→Y} = Σ p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) log [ p(y_{t+1}|y_t^{(k)}, x_t^{(l)}) / p(y_{t+1}|y_t^{(k)}) ]
有效传递熵 ETE = TE − mean(TE 打乱 X 后)（消除有限样本偏差，Marschinski & Kantz 2002）

参考文献
--------
- Shannon (1948). A Mathematical Theory of Communication.
- Schreiber (2000). Measuring Information Transfer. PRL.
- Kraskov, Stögbauer & Grassberger (2004). Estimating mutual information. PRE.
- Bandt & Pompe (2002). Permutation Entropy: A Natural Complexity Measure for Time Series. PRL.
- Marschinski & Kantz (2002). Analysing the information flow between financial time series. EPJB.
- Dimpfl & Peter (2013). Using transfer entropy to measure information flows between financial markets. SNDE.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from .._utils import as_1d, as_rng


# --------------------------------------------------------------------------- 基本熵
def shannon_entropy(p, base: float = np.e) -> float:
    p = np.asarray(p, dtype=float); p = p[p > 0]; p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(base))


def renyi_entropy(p, alpha: float, base: float = np.e) -> float:
    p = np.asarray(p, dtype=float); p = p[p > 0]; p = p / p.sum()
    if abs(alpha - 1) < 1e-12:
        return shannon_entropy(p, base)
    return float(np.log(np.sum(p ** alpha)) / (1 - alpha) / np.log(base))


def tsallis_entropy(p, q: float) -> float:
    p = np.asarray(p, dtype=float); p = p[p > 0]; p = p / p.sum()
    if abs(q - 1) < 1e-12:
        return shannon_entropy(p)
    return float((1 - np.sum(p ** q)) / (q - 1))


def discretize(x, bins: int = 6, method: str = "quantile") -> np.ndarray:
    """把连续序列离散为 0..bins−1 的符号序列（method ∈ {'quantile','uniform','sign'}）。"""
    x = as_1d(x, "x")
    if method == "sign":
        return (x > 0).astype(int)
    if method == "quantile":
        edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    else:
        edges = np.linspace(x.min(), x.max(), bins + 1)[1:-1]
    return np.searchsorted(edges, x, side="right")


def _joint_counts(*cols):
    arr = np.stack(cols, axis=1)
    _, counts = np.unique(arr, axis=0, return_counts=True)
    return counts / counts.sum()


def entropy_discrete(*cols) -> float:
    """离散符号序列（可多列联合）的 Shannon 熵（nats）。"""
    return shannon_entropy(_joint_counts(*cols))


def mutual_information_binned(x, y, bins: int = 6) -> float:
    """分箱互信息 I(X;Y) = H(X) + H(Y) − H(X,Y)。"""
    xs, ys = discretize(x, bins), discretize(y, bins)
    return entropy_discrete(xs) + entropy_discrete(ys) - entropy_discrete(xs, ys)


def mutual_information_ksg(x, y, k: int = 3) -> float:
    """Kraskov–Stögbauer–Grassberger (2004) 算法 1 的 k 近邻互信息估计（nats），适合连续变量。"""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    X = x[:, None] if x.ndim == 1 else x
    Y = y[:, None] if y.ndim == 1 else y
    N = len(X)
    Z = np.hstack([X, Y])
    tree = cKDTree(Z)
    d, _ = tree.query(Z, k=k + 1, p=np.inf)
    eps = d[:, -1]
    tx, ty = cKDTree(X), cKDTree(Y)
    nx = np.array([len(tx.query_ball_point(X[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(N)])
    ny = np.array([len(ty.query_ball_point(Y[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(N)])
    return float(digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


# --------------------------------------------------------------------------- 复杂度
def permutation_entropy(x, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """Bandt–Pompe 排列熵：序数模式分布的 Shannon 熵，normalize=True 时除以 log(order!)。"""
    x = as_1d(x, "x")
    n = len(x) - (order - 1) * delay
    idx = np.arange(order) * delay
    patterns = np.argsort(x[np.arange(n)[:, None] + idx], axis=1)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    H = shannon_entropy(counts)
    n_patterns = float(np.prod(np.arange(1, order + 1)))
    return float(H / np.log(n_patterns)) if normalize else float(H)


def sample_entropy(x, m: int = 2, r: float | None = None) -> float:
    """样本熵 SampEn(m, r) = −log( A/B )，A/B 为长度 m+1 / m 的模板匹配数（切比雪夫距离 < r）。"""
    x = as_1d(x, "x")
    r = 0.2 * np.std(x) if r is None else r
    N = len(x)

    def count(mm):
        templates = np.array([x[i:i + mm] for i in range(N - m)])
        tree = cKDTree(templates)
        pairs = tree.count_neighbors(tree, r, p=np.inf) - len(templates)
        return pairs
    B = count(m); A = count(m + 1)
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.inf


def approximate_entropy(x, m: int = 2, r: float | None = None) -> float:
    """近似熵 ApEn(m, r)（Pincus 1991）。"""
    x = as_1d(x, "x")
    r = 0.2 * np.std(x) if r is None else r
    N = len(x)

    def phi(mm):
        templates = np.array([x[i:i + mm] for i in range(N - mm + 1)])
        tree = cKDTree(templates)
        C = np.array([len(tree.query_ball_point(t, r, p=np.inf)) for t in templates]) / len(templates)
        return np.mean(np.log(C))
    return float(phi(m) - phi(m + 1))


def spectral_entropy(x, normalize: bool = True) -> float:
    """功率谱的归一化 Shannon 熵（白噪声 → 1）。"""
    x = as_1d(x, "x")
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2
    psd = psd[1:]
    H = shannon_entropy(psd)
    return float(H / np.log(len(psd)) if normalize else H)


# --------------------------------------------------------------------------- 传递熵
def transfer_entropy(x, y, k: int = 1, l: int = 1, bins: int = 4, method: str = "quantile") -> float:
    """分箱传递熵 TE_{X→Y}（nats）：X 的过去 l 期对 Y 下一期的额外信息，Y 自身历史 k 期。

    TE = H(y_{t+1}, y^{(k)}) − H(y^{(k)}) − H(y_{t+1}, y^{(k)}, x^{(l)}) + H(y^{(k)}, x^{(l)})
    """
    xs = discretize(x, bins, method); ys = discretize(y, bins, method)
    n = len(xs); m = max(k, l)
    T = n - m
    y_next = ys[m:]
    y_hist = [ys[m - i - 1:n - i - 1] for i in range(k)]
    x_hist = [xs[m - i - 1:n - i - 1] for i in range(l)]
    H_ynext_yh = entropy_discrete(y_next, *y_hist)
    H_yh = entropy_discrete(*y_hist)
    H_all = entropy_discrete(y_next, *y_hist, *x_hist)
    H_yh_xh = entropy_discrete(*y_hist, *x_hist)
    return float(H_ynext_yh - H_yh - H_all + H_yh_xh)


def effective_transfer_entropy(x, y, k: int = 1, l: int = 1, bins: int = 4, n_shuffle: int = 50, seed=None) -> dict:
    """有效传递熵 ETE = TE − mean(打乱 X 的 TE)，并给出置换 p 值与 z 分数。"""
    rng = as_rng(seed)
    x = as_1d(x, "x"); y = as_1d(y, "y")
    te = transfer_entropy(x, y, k, l, bins)
    null = np.array([transfer_entropy(rng.permutation(x), y, k, l, bins) for _ in range(n_shuffle)])
    z = (te - null.mean()) / (null.std() + 1e-12)
    return dict(TE=te, ETE=float(te - null.mean()), z=float(z), p_value=float(np.mean(null >= te)), null=null)


def transfer_entropy_matrix(returns, k: int = 1, l: int = 1, bins: int = 4) -> np.ndarray:
    """多资产传递熵矩阵，M[i, j] = TE_{i→j}（信息从 i 流向 j）。"""
    R = np.asarray(returns, dtype=float)
    n = R.shape[1]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                M[i, j] = transfer_entropy(R[:, i], R[:, j], k, l, bins)
    return M


def max_entropy_weights(mu, Sigma, target_return: float, n_iter: int = 500, lr: float = 0.1) -> np.ndarray:
    """最大熵组合：在给定目标收益下最大化权重熵 −Σ w log w（Bera & Park 2008），镜像下降求解。"""
    mu = np.asarray(mu, dtype=float); n = len(mu)
    w = np.full(n, 1.0 / n)
    lam = 0.0
    for _ in range(n_iter):
        grad = -np.log(w) - 1 + lam * mu
        w = w * np.exp(lr * grad); w /= w.sum()
        lam += lr * (target_return - mu @ w)
    return w
