"""
分形布朗运动与粗糙波动率（Rough Volatility）
===========================================

学科来源
--------
Kolmogorov (1940) 在湍流研究中引入分形布朗运动 fBm（Mandelbrot & Van Ness 1968 命名），
Hurst 指数 H 控制路径粗糙度与长记忆。Gatheral, Jaisson & Rosenbaum (2018) 发现对数已实现波动率
的 H ≈ 0.1——比布朗运动（H=0.5）粗糙得多，由此诞生 rough Bergomi / rough Heston 等模型，
它们能同时拟合 ATM 偏斜的幂律衰减与波动率时间序列特性。

量化映射
--------
- **波动率预测**：rough 模型的预测公式（fBm 条件期望）在 RV 预测上优于 HAR 等基准；
- **期权定价 / 波动率曲面**：rBergomi 复现短期偏斜 ∝ T^{H−1/2}；
- **微观结构解释**：Hawkes 近临界 + 买卖不对称 ⇒ 宏观极限为 rough Heston（El Euch–Rosenbaum）；
- **H 作为状态变量**：对数波动率的 H 估计可监控市场是否“更粗糙”。

核心公式
--------
fBm 协方差：  E[B^H_t B^H_s] = ½(t^{2H} + s^{2H} − |t−s|^{2H})
Davies–Harte：fGn 自协方差 γ(k) 嵌入 2n 维循环矩阵，FFT 对角化精确采样，O(n log n)
H 估计（GJR 2018）：m(q,Δ) = ⟨|log σ_{t+Δ} − log σ_t|^q⟩ ∝ Δ^{ζ_q}，ζ_q ≈ qH
rBergomi：V_t = ξ₀ exp(η Y_t − ½η² t^{2H})，Y_t = √(2H) ∫₀^t (t−s)^{H−1/2} dW_s；
         dS/S = √V (ρ dW + √(1−ρ²) dW⊥)。Y 用混合格式（Bennedsen–Lunde–Pakkanen 2017，κ=1）模拟。

参考文献
--------
- Mandelbrot & Van Ness (1968). Fractional Brownian Motions, Fractional Noises and Applications. SIAM Rev.
- Davies & Harte (1987). Tests for Hurst effect. Biometrika.
- Gatheral, Jaisson & Rosenbaum (2018). Volatility is rough. QF.
- Bayer, Friz & Gatheral (2016). Pricing under rough volatility. QF.
- Bennedsen, Lunde & Pakkanen (2017). Hybrid scheme for Brownian semistationary processes. Finance Stoch.
- El Euch & Rosenbaum (2019). The characteristic function of rough Heston models. Math. Finance.
- Rosenbaum & Zhang (2022). Deep calibration of the quadratic rough Heston model. Risk.
- Fan, Wang & Ye (2026). On options-driven realized volatility forecasting: Information gains via rough volatility model. arXiv:2604.02743.
"""
from __future__ import annotations

import numpy as np

from .._utils import as_1d, as_rng


def fgn_autocov(k, H: float) -> np.ndarray:
    """分形高斯噪声自协方差 γ(k) = ½(|k+1|^{2H} − 2|k|^{2H} + |k−1|^{2H})。"""
    k = np.abs(np.asarray(k, dtype=float))
    return 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H))


def fbm_davies_harte(n: int, H: float, T: float = 1.0, seed=None, n_paths: int = 1) -> np.ndarray:
    """Davies–Harte 循环嵌入精确模拟 fBm，返回 (n+1,) 或 (n_paths, n+1) 的路径（B_0 = 0，时间 [0, T]）。"""
    rng = as_rng(seed)
    m = 2 * n
    r = fgn_autocov(np.arange(n + 1), H)
    c = np.concatenate([r[:n], [r[n]], r[n - 1:0:-1]])  # 长度 2n，对称
    lam = np.fft.fft(c).real
    if lam.min() < -1e-10:
        raise ValueError("循环嵌入特征值为负，增大 n 或调整 H")
    lam = np.maximum(lam, 0.0)
    out = np.zeros((n_paths, n + 1))
    for p in range(n_paths):
        W = np.zeros(m, dtype=complex)
        W[0] = rng.standard_normal()
        W[n] = rng.standard_normal()
        u = rng.standard_normal(n - 1); v = rng.standard_normal(n - 1)
        W[1:n] = (u + 1j * v) / np.sqrt(2)
        W[n + 1:] = np.conj(W[1:n][::-1])
        Y = np.sqrt(lam) * W
        fgn = (np.sqrt(m) * np.fft.ifft(Y)).real[:n]
        out[p, 1:] = np.cumsum(fgn) * (T / n) ** H
    return out[0] if n_paths == 1 else out


def hurst_from_realized_variance(log_vol, q_list=(0.5, 1.0, 1.5, 2.0, 3.0), lags=None) -> dict:
    """Gatheral–Jaisson–Rosenbaum 方法：对 log σ 序列估计 H。返回 ``H``、``zeta_q``、``q``。"""
    x = as_1d(log_vol, "log_vol")
    lags = np.arange(1, 31) if lags is None else np.asarray(lags)
    q_list = np.asarray(q_list, dtype=float)
    zeta = np.zeros(len(q_list))
    for i, q in enumerate(q_list):
        m = np.array([np.mean(np.abs(x[d:] - x[:-d]) ** q) for d in lags])
        zeta[i] = np.polyfit(np.log(lags), np.log(m), 1)[0]
    H = np.polyfit(q_list, zeta, 1)[0]
    return dict(H=float(H), zeta_q=zeta, q=q_list)


def rbergomi_simulate(n_steps: int, n_paths: int, H: float, eta: float, rho: float, xi0: float = 0.04,
                      T: float = 1.0, S0: float = 1.0, seed=None) -> dict:
    """rough Bergomi 模拟（混合格式 κ=1）。返回 ``S (n_paths, n_steps+1)``、``V``、``Y``、``t``。"""
    rng = as_rng(seed)
    a = H - 0.5
    dt = T / n_steps
    t = np.arange(n_steps + 1) * dt
    # 第一段核的精确联合高斯：(ΔW, ∫_{t_{i-1}}^{t_i} (t_i − s)^a dW_s)
    cov = np.array([[dt, dt ** (a + 1) / (a + 1)],
                    [dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)]])
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((n_paths, n_steps, 2)) @ L.T
    dW = Z[:, :, 0]; W1 = Z[:, :, 1]
    # 其余段：g(b_k/n) 权重卷积，b_k = ((k^{a+1} − (k−1)^{a+1})/(a+1))^{1/a}
    k = np.arange(2, n_steps + 1)
    b = ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)
    g = (b * dt) ** a
    Y = np.zeros((n_paths, n_steps + 1))
    for p in range(n_paths):
        conv = np.convolve(dW[p], g)[:n_steps]       # Σ_{k≥2} g_k ΔW_{i−k+1}: 对齐见下
        # conv[i-1] = Σ_{j} dW[j] g[i-1-j]，其中 g[m] 对应 k = m+2 ⇒ j = i-1-(k-2) = i-k+1 ✓
        Y[p, 1:] = np.sqrt(2 * H) * (W1[p] + np.concatenate([[0.0], conv[:-1]]))
    V = xi0 * np.exp(eta * Y - 0.5 * eta ** 2 * t ** (2 * H))
    dB = rho * dW + np.sqrt(1 - rho ** 2) * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    logS = np.log(S0) + np.cumsum(-0.5 * V[:, :-1] * dt + np.sqrt(V[:, :-1]) * dB, axis=1)
    S = np.hstack([np.full((n_paths, 1), S0), np.exp(logS)])
    return dict(S=S, V=V, Y=Y, t=t)


def fbm_forecast(log_vol_path, H: float, horizon: int) -> float:
    """fBm 条件期望预测（GJR 2018 式 (5.1) 的离散近似）：
    E[log σ_{t+Δ} | F_t] ≈ (cos(Hπ)/π) Δ^{H+1/2} ∫_{-∞}^t log σ_s / ((t − s + Δ)(t − s)^{H+1/2}) ds。
    """
    x = as_1d(log_vol_path, "log_vol_path")
    n = len(x)
    s = np.arange(1, n + 1)[::-1]  # t − s，从 1 到 n
    weights = 1.0 / ((s + horizon) * s ** (H + 0.5))
    return float(np.cos(H * np.pi) / np.pi * horizon ** (H + 0.5) * np.sum(weights * x))
