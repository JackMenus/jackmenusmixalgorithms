"""
小波变换：多分辩率分析、去噪与小波方差
=======================================

学科来源
--------
地震信号处理（Morlet 1982）、调和分析（Daubechies 1988 的紧支撑正交小波、Mallat 1989 的多分辩率分析）。
小波把信号同时在时间与尺度上局部化，弥补 Fourier 对非平稳信号的不足。

量化映射
--------
- **多尺度分解**：价格 = 长期趋势（粗尺度）+ 各周期成分 + 噪声；不同持有期的策略看不同尺度；
- **去噪**：阈值收缩（Donoho–Johnstone）得到的平滑序列延迟小于移动平均；
- **小波方差 / 相关**：按尺度分解波动率与相关性（Gençay, Selçuk & Whitcher 2001），
  判断“哪个时间尺度上两个资产真正相关”；
- **多尺度因子**：对因子做 MODWT 后各尺度分量分别建模。

实现
----
正交 DWT（Haar、Daubechies D4）与最大重叠离散小波变换（MODWT，平移不变、任意长度）。

参考文献
--------
- Mallat (1989). A theory for multiresolution signal decomposition. IEEE PAMI.
- Daubechies (1992). Ten Lectures on Wavelets. SIAM.
- Donoho & Johnstone (1994). Ideal spatial adaptation by wavelet shrinkage. Biometrika.
- Gençay, Selçuk & Whitcher (2001). An Introduction to Wavelets and Other Filtering Methods in Finance and Economics. Academic Press.
- Percival & Walden (2000). Wavelet Methods for Time Series Analysis. CUP.
"""
from __future__ import annotations

import numpy as np

from .._utils import as_1d

_FILTERS = {
    "haar": np.array([1.0, 1.0]) / np.sqrt(2),
    "db2": np.array([(1 + np.sqrt(3)), (3 + np.sqrt(3)), (3 - np.sqrt(3)), (1 - np.sqrt(3))]) / (4 * np.sqrt(2)),
}


def _filters(name):
    h = _FILTERS[name]
    L = len(h)
    g = np.array([(-1) ** k * h[L - 1 - k] for k in range(L)])  # 小波（高通）滤波器
    return h, g


def dwt(x, wavelet: str = "haar", level: int | None = None) -> dict:
    """周期化正交 DWT。返回 ``coeffs``（[cA_J, cD_J, ..., cD_1]）与 ``level``。长度需为 2 的幂（否则截断）。"""
    x = as_1d(x, "x")
    n = 2 ** int(np.floor(np.log2(len(x))))
    x = x[-n:]
    h, g = _filters(wavelet)
    max_level = int(np.log2(n))
    level = max_level if level is None else min(level, max_level)
    a = x; details = []
    for _ in range(level):
        m = len(a)
        idx = (np.arange(m)[:, None] + np.arange(len(h))[None, :]) % m
        a_full = a[idx] @ h; d_full = a[idx] @ g
        a, d = a_full[::2], d_full[::2]
        details.append(d)
    return dict(coeffs=[a] + details[::-1], level=level, wavelet=wavelet)


def idwt(coeffs: list, wavelet: str = "haar") -> np.ndarray:
    """周期化正交 DWT 的逆变换。"""
    h, g = _filters(wavelet)
    a = coeffs[0]
    for d in coeffs[1:]:
        m = 2 * len(a)
        up_a = np.zeros(m); up_a[::2] = a
        up_d = np.zeros(m); up_d[::2] = d
        rec = np.zeros(m)
        for k in range(len(h)):
            rec += h[k] * np.roll(up_a, k) + g[k] * np.roll(up_d, k)   # 伴随算子：x[n] = Σ_k h[k] y[n−k]
        a = rec
    return a


def wavelet_denoise(x, wavelet: str = "haar", level: int | None = None, mode: str = "soft") -> np.ndarray:
    """Donoho–Johnstone 通用阈值 σ sqrt(2 log n) 去噪（σ 由最细尺度细节系数的 MAD 估计）。"""
    x = as_1d(x, "x")
    out = dwt(x, wavelet, level)
    coeffs = out["coeffs"]
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thr = sigma * np.sqrt(2 * np.log(len(x)))
    new = [coeffs[0]]
    for d in coeffs[1:]:
        if mode == "soft":
            new.append(np.sign(d) * np.maximum(np.abs(d) - thr, 0))
        else:
            new.append(np.where(np.abs(d) > thr, d, 0))
    rec = idwt(new, wavelet)
    return rec


def modwt(x, wavelet: str = "haar", level: int = 4) -> np.ndarray:
    """最大重叠 DWT（Percival–Walden 的金字塔算法，周期边界）。返回 (level+1, n)：W_1..W_J 与 V_J。"""
    x = as_1d(x, "x")
    h, g = _filters(wavelet)
    h_t, g_t = h / np.sqrt(2), g / np.sqrt(2)
    n = len(x)
    V = x.copy(); W = []
    for j in range(1, level + 1):
        step = 2 ** (j - 1)
        Wj = np.zeros(n); Vj = np.zeros(n)
        for k in range(len(h)):
            shifted = np.roll(V, k * step)
            Wj += g_t[k] * shifted; Vj += h_t[k] * shifted
        W.append(Wj); V = Vj
    return np.array(W + [V])


def wavelet_variance(x, wavelet: str = "haar", level: int = 4) -> dict:
    """各尺度的小波方差（MODWT 系数的方差），尺度 j 对应周期 ~ 2^j 到 2^{j+1}。"""
    W = modwt(x, wavelet, level)
    var = np.array([np.var(W[j]) for j in range(level)])
    return dict(scales=2 ** np.arange(1, level + 1), variance=var, total=float(np.sum(var) + np.var(W[-1])))


def wavelet_correlation(x, y, wavelet: str = "haar", level: int = 4) -> np.ndarray:
    """按尺度分解的相关系数（哪个尺度上两资产真正相关）。"""
    Wx = modwt(x, wavelet, level); Wy = modwt(y, wavelet, level)
    return np.array([np.corrcoef(Wx[j], Wy[j])[0, 1] for j in range(level)])


def multiresolution(x, wavelet: str = "haar", level: int = 4) -> np.ndarray:
    """MODWT 多分辩率分析：返回 (level+1, n) 的细节 D_1..D_J 与平滑 S_J，各行之和等于原序列。"""
    x = as_1d(x, "x")
    h, g = _filters(wavelet)
    h_t, g_t = h / np.sqrt(2), g / np.sqrt(2)
    W = modwt(x, wavelet, level)
    n = len(x)

    def inv_step(Wj, Vj, j):
        step = 2 ** (j - 1)
        out = np.zeros(n)
        for k in range(len(h)):
            out += g_t[k] * np.roll(Wj, -k * step) + h_t[k] * np.roll(Vj, -k * step)
        return out
    comps = []
    for j in range(1, level + 1):
        V = np.zeros(n); Wd = W[j - 1]
        rec = inv_step(Wd, V, j)
        for jj in range(j - 1, 0, -1):
            rec = inv_step(np.zeros(n), rec, jj)
        comps.append(rec)
    S = W[-1]
    for jj in range(level, 0, -1):
        S = inv_step(np.zeros(n), S, jj)
    comps.append(S)
    return np.array(comps)
