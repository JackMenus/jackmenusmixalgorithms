"""
MCMC：Metropolis–Hastings 与哈密顿蒙特卡洛（HMC）
=================================================

学科来源
--------
Metropolis 等 (1953) 在洛斯阿拉莫斯为统计力学模拟提出 MH 算法（Hastings 1970 推广）；
HMC（Duane 等 1987，原名 Hybrid Monte Carlo）为格点 QCD 发明：把参数看成粒子位置，引入辅助动量，
用哈密顿动力学（辛积分器 leapfrog）作长距离、高接受率的提议。

量化映射
--------
- **贝叶斯参数估计**：厚尾收益分布（Student-t）、随机波动率、分层因子模型的后验采样；
- **参数不确定性进组合**：对后验样本做组合优化 ⇒ 稳健权重（Bayesian portfolio）；
- **粒子 MCMC**（见 ``control.particle_filter.pmmh_sv``）：似然用粒子滤波估计。

参考文献
--------
- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953). Equation of State Calculations by Fast Computing Machines.
- Duane, Kennedy, Pendleton & Roweth (1987). Hybrid Monte Carlo. Phys. Lett. B.
- Neal (2011). MCMC using Hamiltonian dynamics. Handbook of MCMC.
- Betancourt (2017). A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv:1701.02434.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import digamma, gammaln

from .._utils import as_1d, as_rng


def metropolis_hastings(logp: Callable, x0, n_samples: int, proposal_std, seed=None, burn_in: int = 0) -> dict:
    """随机游走 MH。返回 ``samples (n, d)``、``accept_rate``。"""
    rng = as_rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    d = len(x)
    scale = np.broadcast_to(np.asarray(proposal_std, dtype=float), (d,))
    lp = logp(x)
    samples = np.zeros((n_samples, d)); acc = 0
    for i in range(n_samples + burn_in):
        prop = x + scale * rng.standard_normal(d)
        lp_prop = logp(prop)
        if np.log(rng.random()) < lp_prop - lp:
            x, lp = prop, lp_prop; acc += 1
        if i >= burn_in:
            samples[i - burn_in] = x
    return dict(samples=samples, accept_rate=acc / (n_samples + burn_in))


def hamiltonian_monte_carlo(logp: Callable, grad_logp: Callable, x0, n_samples: int, step_size: float = 0.1,
                            n_leapfrog: int = 20, seed=None, burn_in: int = 0, mass: float | np.ndarray = 1.0) -> dict:
    """HMC（leapfrog 辛积分 + Metropolis 校正）。返回 ``samples``、``accept_rate``。"""
    rng = as_rng(seed)
    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    d = len(x)
    M = np.broadcast_to(np.asarray(mass, dtype=float), (d,))
    lp, g = logp(x), grad_logp(x)
    samples = np.zeros((n_samples, d)); acc = 0
    for i in range(n_samples + burn_in):
        p = rng.standard_normal(d) * np.sqrt(M)
        H0 = -lp + 0.5 * np.sum(p ** 2 / M)
        xn, gn = x.copy(), g.copy()
        p = p + 0.5 * step_size * gn
        for _ in range(n_leapfrog):
            xn = xn + step_size * p / M
            gn = grad_logp(xn)
            p = p + step_size * gn
        p = p - 0.5 * step_size * gn
        lpn = logp(xn)
        H1 = -lpn + 0.5 * np.sum(p ** 2 / M)
        if np.isfinite(H1) and np.log(rng.random()) < H0 - H1:
            x, lp, g = xn, lpn, gn; acc += 1
        if i >= burn_in:
            samples[i - burn_in] = x
    return dict(samples=samples, accept_rate=acc / (n_samples + burn_in))


def effective_sample_size(chain) -> float:
    """一维链的有效样本量（初始正序列自相关估计）。"""
    x = as_1d(chain, "chain") - np.mean(chain)
    n = len(x)
    f = np.fft.rfft(np.concatenate([x, np.zeros(n)]))
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    rho = acov / acov[0]
    s = 0.0
    for k in range(1, n // 2):
        pair = rho[2 * k - 1] + rho[2 * k] if 2 * k < n else rho[2 * k - 1]
        if pair < 0:
            break
        s += pair
    return float(n / (1 + 2 * s))


def gelman_rubin(chains) -> float:
    """多链 R̂ 统计量（chains 形状 (m, n)）。"""
    c = np.asarray(chains, dtype=float)
    m, n = c.shape
    means = c.mean(axis=1); W = np.mean(c.var(axis=1, ddof=1)); B = n * np.var(means, ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


# --------------------------------------------------------------------------- 应用：Student-t 收益的贝叶斯估计
def student_t_logpost_factory(returns):
    """返回 (logp, grad) 闭包：参数 θ = (μ, log σ, log ν)，弱信息先验。"""
    x = as_1d(returns, "returns")
    n = len(x)

    def logp(theta):
        mu, ls, lnu = theta
        s, nu = np.exp(ls), np.exp(lnu)
        z = (x - mu) / s
        ll = n * (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - ls) \
            - (nu + 1) / 2 * np.sum(np.log1p(z ** 2 / nu))
        prior = -0.5 * (mu / 1.0) ** 2 - 0.5 * (ls / 3.0) ** 2 - 0.5 * ((lnu - np.log(5)) / 1.0) ** 2
        return ll + prior

    def grad(theta):
        mu, ls, lnu = theta
        s, nu = np.exp(ls), np.exp(lnu)
        z = (x - mu) / s
        w = (nu + 1) / (nu + z ** 2)
        g_mu = np.sum(w * z) / s - mu / 1.0
        g_ls = np.sum(w * z ** 2) - n - ls / 9.0
        dll_dnu = n * (0.5 * digamma((nu + 1) / 2) - 0.5 * digamma(nu / 2) - 0.5 / nu) \
            - 0.5 * np.sum(np.log1p(z ** 2 / nu)) + (nu + 1) / 2 * np.sum(z ** 2 / (nu * (nu + z ** 2)))
        g_lnu = dll_dnu * nu - (lnu - np.log(5)) / 1.0
        return np.array([g_mu, g_ls, g_lnu])

    return logp, grad


def bayesian_student_t(returns, n_samples: int = 2000, burn_in: int = 500, step_size: float | None = None,
                       n_leapfrog: int = 15, seed=None) -> dict:
    """HMC 采样 Student-t 收益模型的后验，返回 ``samples``（μ, σ, ν 原尺度）、``accept_rate``、``summary``。

    内部先把收益标准化（除以样本标准差）再采样，使三个参数的后验尺度相近；步长缺省取 0.7/√n
    （后验标准差 ∝ 1/√n，这样一步约走一个后验标准差）。
    """
    x = as_1d(returns, "returns")
    scale = np.std(x)
    z = x / scale
    n = len(z)
    step_size = 0.7 / np.sqrt(n) if step_size is None else step_size
    logp, grad = student_t_logpost_factory(z)
    theta0 = np.array([np.median(z), np.log(np.std(z) * 0.8), np.log(5.0)])
    out = hamiltonian_monte_carlo(logp, grad, theta0, n_samples, step_size, n_leapfrog, seed=seed, burn_in=burn_in)
    s = out["samples"].copy(); s[:, 0] *= scale; s[:, 1] = np.exp(s[:, 1]) * scale; s[:, 2] = np.exp(s[:, 2])
    summary = dict(mu=(s[:, 0].mean(), s[:, 0].std()), sigma=(s[:, 1].mean(), s[:, 1].std()), nu=(s[:, 2].mean(), s[:, 2].std()))
    return dict(samples=s, accept_rate=out["accept_rate"], summary=summary)
