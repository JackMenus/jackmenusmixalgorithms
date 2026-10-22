"""
Hurst 指数、DFA 与多重分形：长记忆与标度律
==========================================

学科来源
--------
水文学（Hurst 1951 研究尼罗河水位的重标极差 R/S）、分形几何（Mandelbrot 的分形布朗运动 fBm）、
统计物理的标度理论（Peng 等 1994 为 DNA 序列提出 DFA，去趋势涨落分析）。
核心思想：自相似过程的涨落随尺度按幂律 F(s) ∝ s^H 增长，H 刻画记忆性。

量化映射
--------
- H ≈ 0.5：随机游走；H > 0.5：趋势持续（动量有效）；H < 0.5：反持续（均值回复有效）。
  滚动 Hurst 可作为“趋势/震荡 regime”判别器，决定用动量还是反转因子。
- **多重分形宽度 Δh**：大小涨落标度不同，宽度越大市场越“不均匀”，与危机前后状态相关。
- **方差比检验**（Lo–MacKinlay 1988）：VR(k) ≠ 1 直接给出可预测性的统计显著性。
- **粗糙波动率**：对数波动率的 H ≈ 0.1（Gatheral 等 2018），见 ``rough_volatility``。

核心公式
--------
R/S：  E[R(n)/S(n)] ∝ n^H
DFA：  Y(i) = Σ_{k≤i}(x_k − x̄)；分段多项式去趋势后 F(s) = sqrt(mean residual²) ∝ s^α（对增量序列 α = H）
MF-DFA： F_q(s) = (mean_ν F²(s,ν)^{q/2})^{1/q} ∝ s^{h(q)}，Δh = h(q_min) − h(q_max)
广义 Hurst（Di Matteo）： K_q(τ) = E|X(t+τ) − X(t)|^q ∝ τ^{q H(q)}
方差比： VR(k) = Var(r_t + … + r_{t−k+1}) / (k Var(r_t))

参考文献
--------
- Hurst (1951). Long-term storage capacity of reservoirs. Trans. ASCE.
- Peng et al. (1994). Mosaic organization of DNA nucleotides. PRE（DFA）。
- Kantelhardt et al. (2002). Multifractal detrended fluctuation analysis of nonstationary time series. Physica A.
- Di Matteo (2007). Multi-scaling in finance. QF.
- Lo & MacKinlay (1988). Stock market prices do not follow random walks. RFS.
"""
from __future__ import annotations

import numpy as np

from .._utils import as_1d


def rescaled_range_hurst(x, min_window: int = 8, max_window: int | None = None, n_scales: int = 12) -> dict:
    """经典 R/S 分析（输入为增量/收益序列）。返回 ``H``、``scales``、``rs``。"""
    x = as_1d(x, "x")
    n = len(x)
    max_window = n // 2 if max_window is None else max_window
    scales = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), n_scales).astype(int))
    rs = []
    for s in scales:
        m = n // s
        vals = []
        for i in range(m):
            seg = x[i * s:(i + 1) * s]
            y = np.cumsum(seg - seg.mean())
            R = y.max() - y.min(); S = seg.std(ddof=1)
            if S > 0:
                vals.append(R / S)
        rs.append(np.mean(vals))
    rs = np.array(rs)
    H = np.polyfit(np.log(scales), np.log(rs), 1)[0]
    return dict(H=float(H), scales=scales, rs=rs)


def dfa(x, scales=None, order: int = 1, min_scale: int = 8, n_scales: int = 15) -> dict:
    """去趋势涨落分析（DFA-m）。输入增量序列（收益），返回 ``alpha``（= H）、``scales``、``F``。

    对价格（已积分的序列）输入时得到 α = H + 1。
    """
    x = as_1d(x, "x")
    n = len(x)
    y = np.cumsum(x - x.mean())
    if scales is None:
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(n // 4), n_scales).astype(int))
    F = []
    for s in scales:
        m = n // s
        segs = y[:m * s].reshape(m, s)
        segs_rev = y[n - m * s:].reshape(m, s)[::-1]
        t = np.arange(s)
        res = []
        for block in (segs, segs_rev):
            coef = np.polyfit(t, block.T, order)
            fit = np.polyval(coef, t[:, None]).T if order > 0 else np.broadcast_to(coef.T, block.shape)
            res.append(np.mean((block - fit) ** 2, axis=1))
        F.append(np.sqrt(np.mean(np.concatenate(res))))
    F = np.array(F)
    alpha = np.polyfit(np.log(scales), np.log(F), 1)[0]
    return dict(alpha=float(alpha), H=float(alpha), scales=np.asarray(scales), F=F)


def mfdfa(x, q_list=(-5, -3, -1, 0, 1, 3, 5), scales=None, order: int = 1, min_scale: int = 16, n_scales: int = 12) -> dict:
    """多重分形 DFA。返回 ``q``、``h``（广义 Hurst h(q)）、``width``（Δh）、``F_q (len q, len scales)``。"""
    x = as_1d(x, "x")
    n = len(x)
    y = np.cumsum(x - x.mean())
    q_list = np.asarray(q_list, dtype=float)
    if scales is None:
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(n // 4), n_scales).astype(int))
    Fq = np.zeros((len(q_list), len(scales)))
    for j, s in enumerate(scales):
        m = n // s
        t = np.arange(s)
        F2 = []
        for block in (y[:m * s].reshape(m, s), y[n - m * s:].reshape(m, s)[::-1]):
            coef = np.polyfit(t, block.T, order)
            fit = np.polyval(coef, t[:, None]).T
            F2.append(np.mean((block - fit) ** 2, axis=1))
        F2 = np.concatenate(F2); F2 = F2[F2 > 0]
        for i, q in enumerate(q_list):
            Fq[i, j] = np.exp(0.5 * np.mean(np.log(F2))) if q == 0 else np.mean(F2 ** (q / 2)) ** (1 / q)
    h = np.array([np.polyfit(np.log(scales), np.log(Fq[i]), 1)[0] for i in range(len(q_list))])
    return dict(q=q_list, h=h, width=float(h.min() - h.max()) * -1, F_q=Fq, scales=np.asarray(scales))


def generalized_hurst(x_level, q: float = 1.0, max_tau: int = 20) -> dict:
    """广义 Hurst 指数（Di Matteo）：输入为水平序列（价格/对数价格）。返回 ``H``、``K_q``。"""
    x = as_1d(x_level, "x_level")
    taus = np.arange(1, max_tau + 1)
    K = np.array([np.mean(np.abs(x[tau:] - x[:-tau]) ** q) for tau in taus])
    slope = np.polyfit(np.log(taus), np.log(K), 1)[0]
    return dict(H=float(slope / q), K_q=K, taus=taus)


def variance_ratio(returns, k: int) -> dict:
    """Lo–MacKinlay 方差比 VR(k) 与同方差假设下的 z 统计量（重叠样本、无偏修正）。"""
    r = as_1d(returns, "returns")
    T = len(r)
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (T - 1)
    rk = np.convolve(r, np.ones(k), "valid")
    m = k * (T - k + 1) * (1 - k / T)
    vark = np.sum((rk - k * mu) ** 2) / m
    vr = vark / var1
    z = (vr - 1) / np.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k * T))
    return dict(VR=float(vr), z=float(z))


def rolling_hurst(returns, window: int = 250, step: int = 5, method: str = "dfa") -> dict:
    """滚动窗口 Hurst（regime 指示器）。返回 ``index``（窗口末端位置）、``H``。"""
    r = as_1d(returns, "returns")
    idx, H = [], []
    for end in range(window, len(r) + 1, step):
        seg = r[end - window:end]
        h = dfa(seg)["H"] if method == "dfa" else rescaled_range_hurst(seg)["H"]
        idx.append(end - 1); H.append(h)
    return dict(index=np.array(idx), H=np.array(H))
