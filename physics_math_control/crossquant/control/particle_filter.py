"""
粒子滤波（Sequential Monte Carlo）
=================================

学科来源
--------
控制 / 信号处理 / 统计物理（Gordon, Salmond & Smith 1993 的 bootstrap filter；
Del Moral 1996 把它与 Feynman–Kac 公式联系起来）。用一组加权“粒子”近似非线性、非高斯
隐状态的后验分布，是 Kalman 滤波的非线性推广。

量化映射
--------
1. **随机波动率滤波**：r_t = exp(h_t/2) ε_t 的观测似然非高斯，粒子滤波给出 h_t 的后验与
   边际似然（可配合 PMMH 做贝叶斯参数估计）。
2. **状态切换 / 市场状态（regime）在线识别**：隐状态为离散状态 + 连续参数。
3. **非线性订单簿隐变量**（如真实流动性、库存压力）的在线估计。

算法（SIR / bootstrap）
----------------------
1. 传播：x_t^{(i)} ~ p(x_t | x_{t-1}^{(i)})
2. 加权：w_t^{(i)} ∝ w_{t-1}^{(i)} p(z_t | x_t^{(i)})
3. 边际似然增量：p̂(z_t | z_{1:t-1}) = Σ_i w̃_{t-1}^{(i)} p(z_t | x_t^{(i)})
4. 当有效样本量 ESS = 1/Σ w̃² 低于阈值时系统重采样（systematic resampling）

参考文献
--------
- Gordon, Salmond & Smith (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation.
- Doucet & Johansen (2009). A Tutorial on Particle Filtering and Smoothing.
- Andrieu, Doucet & Holenstein (2010). Particle Markov chain Monte Carlo methods. JRSS-B.
- Kim, Shephard & Chib (1998). Stochastic Volatility: Likelihood Inference and Comparison with ARCH.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .._utils import as_1d, as_rng


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """系统重采样：返回被选中粒子的索引数组（方差小于多项式重采样）。"""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    return np.searchsorted(cumsum, positions)


def effective_sample_size(weights: np.ndarray) -> float:
    """有效样本量 ESS = 1 / Σ w_i²。"""
    return 1.0 / np.sum(weights ** 2)


class ParticleFilter:
    """通用 bootstrap 粒子滤波器。

    参数
    ----
    transition : (particles (N,d), rng) -> new particles (N,d)，按状态方程采样
    loglik     : (particles (N,d), z) -> (N,) 观测对数似然
    init       : (N, rng) -> (N,d) 初始粒子
    n_particles: 粒子数
    resample_threshold : ESS/N 低于该比例时重采样
    """

    def __init__(self, transition: Callable, loglik: Callable, init: Callable,
                 n_particles: int = 1000, resample_threshold: float = 0.5, seed=None):
        self.transition, self.loglik, self.init = transition, loglik, init
        self.N = n_particles
        self.threshold = resample_threshold
        self.rng = as_rng(seed)

    def run(self, zs) -> dict:
        """返回 ``mean (T,d)``、``std (T,d)``、``loglik``（边际似然）、``ess (T,)``、``particles``（末期）。"""
        zs = np.asarray(zs, dtype=float)
        T = len(zs)
        x = self.init(self.N, self.rng)
        x = x if x.ndim == 2 else x[:, None]
        d = x.shape[1]
        logw = np.full(self.N, -np.log(self.N))
        mean = np.zeros((T, d)); std = np.zeros((T, d)); ess = np.zeros(T)
        loglik = 0.0
        for t in range(T):
            x = self.transition(x, self.rng)
            ll = self.loglik(x, zs[t])
            logw_new = logw + ll
            m = np.max(logw_new)
            # 边际似然增量 = log Σ w̃_{t-1} p(z_t|x_t)
            loglik += m + np.log(np.sum(np.exp(logw_new - m)))
            logw = logw_new - (m + np.log(np.sum(np.exp(logw_new - m))))
            w = np.exp(logw)
            mean[t] = w @ x
            std[t] = np.sqrt(np.maximum(w @ (x - mean[t]) ** 2, 0.0))
            ess[t] = effective_sample_size(w)
            if ess[t] < self.threshold * self.N:
                idx = systematic_resample(w, self.rng)
                x = x[idx]
                logw = np.full(self.N, -np.log(self.N))
        return dict(mean=mean, std=std, loglik=loglik, ess=ess, particles=x)


def sv_particle_filter(returns, mu: float, phi: float, sigma_eta: float,
                       n_particles: int = 2000, seed=None) -> dict:
    """随机波动率模型的 bootstrap 粒子滤波。

    模型：h_t = μ + φ(h_{t−1} − μ) + σ_η η_t,  r_t = exp(h_t/2) ε_t。
    返回 ``h_filt``、``vol``（exp(h/2)）、``loglik``（可用于参数估计/模型比较）。
    """
    r = as_1d(returns, "returns")
    stat_std = sigma_eta / np.sqrt(max(1 - phi ** 2, 1e-8))

    def transition(x, rng):
        return mu + phi * (x - mu) + sigma_eta * rng.standard_normal(x.shape)

    def loglik(x, z):
        h = x[:, 0]
        return -0.5 * (np.log(2 * np.pi) + h + z ** 2 * np.exp(-h))

    def init(N, rng):
        return mu + stat_std * rng.standard_normal((N, 1))

    pf = ParticleFilter(transition, loglik, init, n_particles=n_particles, seed=seed)
    out = pf.run(r)
    h = out["mean"][:, 0]
    return dict(h_filt=h, vol=np.exp(h / 2), h_std=out["std"][:, 0], loglik=out["loglik"], ess=out["ess"])


def pmmh_sv(returns, n_iter: int = 300, n_particles: int = 500, seed=None,
            init=(0.0, 0.95, 0.2), step=(0.1, 0.02, 0.05)) -> dict:
    """粒子边际 Metropolis–Hastings（PMMH）估计 SV 模型参数 (μ, φ, σ_η)。

    每一步用粒子滤波的边际似然估计替代精确似然，是 Andrieu–Doucet–Holenstein (2010) 的方法。
    迭代次数较少时仅作演示；返回参数链 ``chain (n_iter,3)`` 与接受率。
    """
    rng = as_rng(seed)
    r = as_1d(returns, "returns")
    theta = np.array(init, dtype=float)

    def logpost(th):
        mu, phi, sig = th
        if not (-0.999 < phi < 0.999) or sig <= 0:
            return -np.inf
        ll = sv_particle_filter(r, mu, phi, sig, n_particles, seed=rng.integers(1 << 31))["loglik"]
        prior = -0.5 * (mu / 5) ** 2 - 0.5 * ((phi - 0.9) / 0.2) ** 2 - np.log(sig) - 0.5 * (np.log(sig) / 1.0) ** 2
        return ll + prior

    lp = logpost(theta)
    chain = np.zeros((n_iter, 3)); acc = 0
    for i in range(n_iter):
        prop = theta + np.asarray(step) * rng.standard_normal(3)
        lp_prop = logpost(prop)
        if np.log(rng.random()) < lp_prop - lp:
            theta, lp = prop, lp_prop
            acc += 1
        chain[i] = theta
    return dict(chain=chain, accept_rate=acc / n_iter, mean=chain[n_iter // 2:].mean(axis=0))
