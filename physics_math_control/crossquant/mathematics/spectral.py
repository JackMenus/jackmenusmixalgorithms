"""
谱方法：周期图、Hilbert 变换与 Carr–Madan FFT 定价
==================================================

学科来源
--------
Fourier 分析（热传导 → 信号处理）。Carr & Madan (1999) 发现：期权价格对（对数）行权价的 Fourier 变换
可用标的对数价格的特征函数闭式表示，一次 FFT 得到整条行权价曲线——Heston、Lévy 模型定价由此可行。
Hilbert 变换给出解析信号的瞬时相位/幅度（Gabor 1946），Ehlers 把它引入技术分析做周期自适应指标。

量化映射
--------
- 期权快速定价与校准（特征函数已知的任何模型）；
- 周期检测：主频 → 自适应窗口长度（均线/振荡器参数随主周期变化）；
- 瞬时相位用于判断趟势/震荡状态、相位同步测两资产的领先滞后。

参考文献
--------
- Carr & Madan (1999). Option valuation using the fast Fourier transform. J. Comp. Finance.
- Albrecher, Mayer, Schoutens & Tistaert (2007). The Little Heston Trap. Wilmott.
- Ehlers (2001). Rocket Science for Traders. Wiley.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.signal import hilbert

from .._utils import as_1d


def periodogram(x) -> dict:
    x = as_1d(x, "x") - np.mean(x)
    n = len(x)
    power = np.abs(np.fft.rfft(x)) ** 2 / n
    freqs = np.fft.rfftfreq(n)
    return dict(freqs=freqs, power=power)


def dominant_cycles(x, n_peaks: int = 3, min_period: float = 3.0) -> dict:
    """周期图峰值对应的周期长度（样本数）。"""
    pg = periodogram(x)
    f, p = pg["freqs"][1:], pg["power"][1:]
    mask = (1 / f) >= min_period
    f, p = f[mask], p[mask]
    idx = np.argsort(p)[::-1][:n_peaks]
    return dict(periods=1 / f[idx], power=p[idx])


def hilbert_phase(x) -> dict:
    """解析信号的瞬时幅度、相位（unwrap）与瞬时频率（周期/样本）。"""
    x = as_1d(x, "x") - np.mean(x)
    a = hilbert(x)
    phase = np.unwrap(np.angle(a))
    freq = np.diff(phase) / (2 * np.pi)
    return dict(amplitude=np.abs(a), phase=phase, inst_frequency=freq, inst_period=1 / np.maximum(np.abs(freq), 1e-12))


# --------------------------------------------------------------------------- 特征函数
def bs_char_fn(S0: float, r: float, sigma: float, T: float) -> Callable:
    """Black–Scholes 对数价格的特征函数 φ(u) = E[e^{iu ln S_T}]。"""
    def phi(u):
        u = np.asarray(u, dtype=complex)
        return np.exp(1j * u * (np.log(S0) + (r - 0.5 * sigma ** 2) * T) - 0.5 * sigma ** 2 * u ** 2 * T)
    return phi


def heston_char_fn(S0: float, v0: float, r: float, kappa: float, theta: float, xi: float, rho: float, T: float) -> Callable:
    """Heston 特征函数（Albrecher 等 2007 的稳定形式）。"""
    def phi(u):
        u = np.asarray(u, dtype=complex)
        d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u ** 2))
        g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)
        C = r * 1j * u * T + kappa * theta / xi ** 2 * ((kappa - rho * xi * 1j * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = (kappa - rho * xi * 1j * u - d) / xi ** 2 * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        return np.exp(C + D * v0 + 1j * u * np.log(S0))
    return phi


def carr_madan_call(S0: float, r: float, T: float, char_fn: Callable, strikes=None, alpha: float = 1.5,
                    N: int = 4096, eta: float = 0.25) -> dict:
    """Carr–Madan FFT 欧式看涨定价。返回 ``strikes``、``prices``（若传入 strikes 则插值到这些行权价）。"""
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2
    v = np.arange(N) * eta
    k = -b + lam * np.arange(N)
    psi = np.exp(-r * T) * char_fn(v - (alpha + 1) * 1j) / (alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)
    simpson = (3 + (-1) ** (np.arange(N) + 1)) / 3; simpson[0] = 1 / 3
    x = np.exp(1j * b * v) * psi * eta * simpson
    fft = np.fft.fft(x)
    prices = np.exp(-alpha * k) / np.pi * fft.real
    K_grid = np.exp(k)
    if strikes is None:
        return dict(strikes=K_grid, prices=prices)
    strikes = np.asarray(strikes, dtype=float)
    return dict(strikes=strikes, prices=np.interp(np.log(strikes), k, prices))
