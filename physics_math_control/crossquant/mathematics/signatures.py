"""
路径签名（Path Signature）与粗糙路径
====================================

学科来源
--------
Chen (1957) 的迭代积分，Lyons (1998) 的粗糙路径理论。路径 X:[0,T]→R^d 的签名
S(X) = (1, ∫dX, ∬dX⊗dX, …) 是路径的“非交换幂级数”：它对路径是 **唯一刻画**（在树状等价意义下），
对连续泛函是 **通用逼近器**（任意路径泛函可由签名的线性泛函逼近），且天然处理不等间隔、不同长度的路径。
Chen 恒等式 S(X∗Y) = S(X) ⊗ S(Y) 让分段线性路径的签名可以精确递推计算。

量化映射
--------
- **路径特征**：对滚动窗口的价格/成交量多维路径取截断签名，作为因子挖掘的非线性特征（Levin, Lyons & Ni 2013）；
  二阶项的反对称部分 = Lévy 面积，刻画两条序列的“领先–滞后旋转”；
- **签名交易（Futter, Horvath & Wiese 2023）**：持仓 ξ_t = ⟨ℓ, S(X)_{0,t}⟩ 是签名的线性泛函，
  均值方差最优的 ℓ 有闭式解 —— 把 Markowitz 从“静态权重”推广到“路径依赖策略”；
- **配对交易**（Guo 等 2025）：把签名分解成两个指标做双重过滤；
- **市场模拟器评价**：签名 MMD 比较生成路径与真实路径的分布。

核心公式
--------
分段线性路径一段增量 Δ 的签名：exp(Δ) 的 k 阶项 = Δ^{⊗k}/k!
Chen 恒等式（k 阶）：(S(X)⊗S(Y))_k = Σ_{i=0}^k S(X)_i ⊗ S(Y)_{k−i}
对数签名：log S = Σ_{k≥1} (−1)^{k+1}(S−1)^{⊗k}/k（李代数元素，维数更小、无冗余）
签名交易（单资产，简化）：PnL = Σ_t ξ_t ΔX_{t+1} = ℓᵀ Σ_t Z_t ΔX_{t+1} =: ℓᵀY，
    ℓ* = (2λ)^{-1} (Cov(Y) + ridge·I)^{-1} E[Y]

参考文献
--------
- Lyons (1998). Differential equations driven by rough signals. Rev. Mat. Iberoam.
- Levin, Lyons & Ni (2013). Learning from the past, predicting the statistics for the future, learning an evolving system. arXiv:1309.0260
- Chevyrev & Kormilitzin (2016). A Primer on the Signature Method in Machine Learning. arXiv:1603.03788
- Futter, Horvath & Wiese (2023). Signature Trading: A Path-Dependent Extension of the Mean-Variance Framework with Exogenous Signals. arXiv:2308.15135
- Guo, Jin, Kuang, Qian & Wang (2025). Signature Decomposition Method Applying to Pair Trading. arXiv:2505.05332
- Kidger & Lyons (2021). Signatory: differentiable computations of the signature and logsignature transforms.
"""
from __future__ import annotations

from math import factorial

import numpy as np

from .._utils import as_2d


# --------------------------------------------------------------------------- 张量代数
def _tensor_exp(delta: np.ndarray, depth: int) -> list:
    """一段线性增量的签名 exp(Δ)：返回 [level1, ..., level_depth]。"""
    levels = [delta.copy()]
    power = delta.copy()                      # Δ^{⊗k}（未归一）
    for k in range(2, depth + 1):
        power = np.multiply.outer(power, delta)
        levels.append(power / factorial(k))   # Δ^{⊗k}/k!
    return levels


def tensor_product(A: list, B: list, depth: int) -> list:
    """截断张量代数乘法（含隐含的 0 阶 1）：(A⊗B)_k = A_k + B_k + Σ_{i=1}^{k−1} A_i ⊗ B_{k−i}。"""
    out = []
    for k in range(1, depth + 1):
        term = A[k - 1] + B[k - 1]
        for i in range(1, k):
            term = term + np.multiply.outer(A[i - 1], B[k - i - 1])
        out.append(term)
    return out


def signature(path, depth: int = 3) -> list:
    """分段线性路径 (T, d) 的截断签名，返回 [level1 (d,), level2 (d,d), ..., level_depth]。"""
    X = as_2d(path, "path")
    d = X.shape[1]
    sig = [np.zeros((d,) * k) for k in range(1, depth + 1)]
    for t in range(1, len(X)):
        delta = X[t] - X[t - 1]
        sig = tensor_product(sig, _tensor_exp(delta, depth), depth) if t > 1 else _tensor_exp(delta, depth)
    return sig


def signature_flat(path, depth: int = 3, include_one: bool = False) -> np.ndarray:
    """展平后的签名向量（维数 d + d² + … + d^depth，可选前置常数 1）。"""
    levels = signature(path, depth)
    vec = np.concatenate([lv.reshape(-1) for lv in levels])
    return np.concatenate([[1.0], vec]) if include_one else vec


def log_signature(sig_levels: list) -> list:
    """由截断签名计算对数签名（张量对数级数），返回同样分层的列表。"""
    depth = len(sig_levels)
    X = sig_levels  # S − 1 的各阶
    result = [lv.copy() for lv in X]
    power = [lv.copy() for lv in X]
    for k in range(2, depth + 1):
        power = _tensor_product_no_unit(power, X, depth)
        coef = (-1) ** (k + 1) / k
        for i in range(depth):
            result[i] = result[i] + coef * power[i]
    return result


def _tensor_product_no_unit(A, B, depth):
    out = []
    for k in range(1, depth + 1):
        term = np.zeros_like(A[k - 1])
        for i in range(1, k):
            term = term + np.multiply.outer(A[i - 1], B[k - i - 1])
        out.append(term)
    return out


def levy_area(path) -> np.ndarray:
    """二阶签名的反对称部分 A_ij = ½(S_ij − S_ji)（Lévy 面积矩阵）。"""
    S2 = signature(path, 2)[1]
    return 0.5 * (S2 - S2.T)


# --------------------------------------------------------------------------- 路径变换
def add_time(path, t=None) -> np.ndarray:
    """时间增广：在第 0 列加入时间（使签名区分“走多快”）。"""
    X = as_2d(path, "path")
    t = np.linspace(0, 1, len(X)) if t is None else np.asarray(t, dtype=float)
    return np.hstack([t[:, None], X])


def lead_lag(path) -> np.ndarray:
    """Lead–lag 变换：(X^lead, X^lag) 的二维嵌入，其 Lévy 面积编码二次变差/波动率。"""
    X = as_2d(path, "path")
    T, d = X.shape
    out = np.zeros((2 * T - 1, 2 * d))
    for i in range(T):
        out[2 * i, :d] = X[i]; out[2 * i, d:] = X[i]
        if i < T - 1:
            out[2 * i + 1, :d] = X[i + 1]; out[2 * i + 1, d:] = X[i]
    return out


def basepoint(path) -> np.ndarray:
    """在路径前加原点，使签名对平移敏感（否则签名只看增量）。"""
    X = as_2d(path, "path")
    return np.vstack([np.zeros((1, X.shape[1])), X])


# --------------------------------------------------------------------------- 特征与签名交易
def sliding_signature_features(series, window: int, depth: int = 3, step: int = 1, transform=None,
                               normalize: bool = True) -> dict:
    """滚动窗口签名特征矩阵。series (T,) 或 (T,d)；transform 可为 add_time / lead_lag 等。

    返回 ``features (n_windows, n_feat)``、``index``（窗口末端位置）。normalize=True 时窗口内先减起点。
    """
    X = as_2d(series, "series")
    feats, idx = [], []
    for end in range(window, len(X) + 1, step):
        seg = X[end - window:end]
        if normalize:
            seg = seg - seg[0]
        if transform is not None:
            seg = transform(seg)
        feats.append(signature_flat(seg, depth)); idx.append(end - 1)
    return dict(features=np.array(feats), index=np.array(idx))


class SignatureTrading:
    """简化版签名交易（Futter–Horvath–Wiese 2023）：持仓是滚动窗口签名的线性泛函。

    参数
    ----
    window : 签名所用的回看窗口长度
    depth : 截断阶数
    risk_aversion : λ（均值方差权衡）
    ridge : 协方差正则化
    transform : 路径变换（缺省时间增广）
    """

    def __init__(self, window: int = 20, depth: int = 2, risk_aversion: float = 1.0, ridge: float = 1e-6, transform=add_time):
        self.window, self.depth, self.lam, self.ridge, self.transform = window, depth, risk_aversion, ridge, transform
        self.ell = None

    def _features(self, x):
        out = sliding_signature_features(x, self.window, self.depth, transform=self.transform)
        Z = np.hstack([np.ones((len(out["features"]), 1)), out["features"]])
        return Z, out["index"]

    def fit(self, x) -> "SignatureTrading":
        """x : 价格/对数价格序列 (T,) 或多维 (T,d)（交易第 0 列资产）。"""
        X = as_2d(x, "x")
        Z, idx = self._features(X)
        dX = X[1:, 0] - X[:-1, 0]
        Z, idx = Z[:-1], idx[:-1]                    # 持仓在 t，收益在 t→t+1
        Y = Z * dX[idx][:, None]                     # Y_t = Z_t ΔX_{t+1}
        mean = Y.mean(axis=0); cov = np.cov(Y, rowvar=False) + self.ridge * np.eye(Z.shape[1])
        self.ell = np.linalg.solve(cov, mean) / (2 * self.lam)
        self.feature_scale = np.abs(Z).mean(axis=0)
        return self

    def positions(self, x) -> dict:
        """返回 ``position``（与窗口末端对齐）、``index``。"""
        X = as_2d(x, "x")
        Z, idx = self._features(X)
        return dict(position=Z @ self.ell, index=idx)

    def backtest(self, x, cost: float = 0.0) -> dict:
        X = as_2d(x, "x")
        pos = self.positions(X)
        p, idx = pos["position"][:-1], pos["index"][:-1]
        dX = X[1:, 0] - X[:-1, 0]
        pnl = p * dX[idx] - cost * np.abs(np.diff(np.concatenate([[0.0], p])))
        sharpe = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
        return dict(pnl=pnl, position=p, index=idx, sharpe=float(sharpe))
