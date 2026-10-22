"""
变点检测：CUSUM（统计过程控制）与贝叶斯在线变点检测（BOCPD）
=============================================================

学科来源
--------
- **CUSUM** 来自工业质量控制（Page 1954，Shewhart 控制图的序贯推广）：对偏离目标的累计和设阈值报警，
  在给定误报率下检测均值漂移的平均延迟最小（Lorden 1971, Moustakides 1986 的最优性）。
- **BOCPD**（Adams & MacKay 2007）来自贝叶斯信号处理：对“当前段已持续多长（run length）”做在线
  后验推断，带共轭先验时每步只需 O(当前 run length) 计算。

量化映射
--------
1. **事件驱动采样**（López de Prado 2018, AFML 第 2 章）：对价格用对称 CUSUM 过滤，只在累计偏离
   超过阈值时生成“事件”，作为因子/标签的采样点，比固定时间采样更贴近信息到达。
2. **策略失效 / 状态切换报警**：对策略日收益、因子 IC、延迟中位数做 CUSUM；超过 h 即报警。
3. **波动率/均值 regime 的在线概率**：BOCPD 给出“刚发生变点”的概率 P(r_t = 0 | x_{1:t})，
   可直接作为风险缩放或信号置信度的乘子。

核心公式
--------
Page CUSUM（均值从 μ0 漂到 μ1，已知 σ）：
    S_t = max(0, S_{t−1} + (μ1−μ0)/σ² · (x_t − (μ0+μ1)/2)),  S_t > h ⇒ 报警
对称 CUSUM（AFML）：
    S⁺_t = max(0, S⁺_{t−1} + Δx_t), S⁻_t = min(0, S⁻_{t−1} + Δx_t)；|S| > h ⇒ 事件并归零
BOCPD（Normal–Gamma 共轭）：
    P(r_t | x_{1:t}) ∝ Σ_{r_{t−1}} P(x_t | r_{t−1}, 段内数据) · P(r_t | r_{t−1}) · P(r_{t−1} | x_{1:t−1})
    预测分布为 Student-t；hazard H 为常数时 P(r_t=0|r_{t−1}) = H。

参考文献
--------
- Page (1954). Continuous Inspection Schemes. Biometrika.
- Adams & MacKay (2007). Bayesian Online Changepoint Detection. arXiv:0710.3742.
- López de Prado (2018). Advances in Financial Machine Learning, Ch. 2 (CUSUM filter).
- Lai (1995). Sequential changepoint detection in quality control and dynamical systems. JRSS-B.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .._utils import as_1d


def cusum_filter(series, threshold: float, use_diff: bool = True) -> dict:
    """对称 CUSUM 事件采样（AFML Snippet 2.4）。

    参数
    ----
    series : 价格或累积量序列；use_diff=True 时内部先做一阶差分
    threshold : 阈值 h（与 series 同单位；常取 Δx 日标准差的若干倍）

    返回 dict：``events``（触发事件的索引）、``s_pos``、``s_neg``（累计量路径）、``direction``（+1/−1）。
    """
    x = as_1d(series, "series")
    d = np.diff(x) if use_diff else x
    s_pos = np.zeros(len(d)); s_neg = np.zeros(len(d))
    events, direction = [], []
    sp = sn = 0.0
    for i, v in enumerate(d):
        sp = max(0.0, sp + v); sn = min(0.0, sn + v)
        if sp > threshold:
            events.append(i + (1 if use_diff else 0)); direction.append(1); sp = 0.0
        elif sn < -threshold:
            events.append(i + (1 if use_diff else 0)); direction.append(-1); sn = 0.0
        s_pos[i], s_neg[i] = sp, sn
    return dict(events=np.array(events, dtype=int), direction=np.array(direction, dtype=int), s_pos=s_pos, s_neg=s_neg)


def page_cusum(x, mu0: float, mu1: float, sigma: float, h: float, two_sided: bool = True, reset: bool = True) -> dict:
    """Page CUSUM 均值漂移检测（对数似然比形式）。

    返回 ``stat_up``、``stat_down``、``alarms``（报警索引）、``change_points``（估计的变点起点）。
    two_sided=True 时同时检测向上漂移（μ0→μ1）与向下漂移（μ0→2μ0−μ1）。
    """
    x = as_1d(x, "x")
    k_up = (mu1 - mu0) / sigma ** 2
    mid_up = 0.5 * (mu0 + mu1)
    mu1_dn = 2 * mu0 - mu1
    k_dn = (mu0 - mu1_dn) / sigma ** 2
    mid_dn = 0.5 * (mu0 + mu1_dn)
    su = np.zeros(len(x)); sd = np.zeros(len(x))
    alarms, cps = [], []
    s_u = s_d = 0.0; start_u = start_d = 0
    for t, v in enumerate(x):
        s_u_new = s_u + k_up * (v - mid_up)
        if s_u_new <= 0:
            s_u, start_u = 0.0, t + 1
        else:
            s_u = s_u_new
        if two_sided:
            s_d_new = s_d + k_dn * (mid_dn - v)
            if s_d_new <= 0:
                s_d, start_d = 0.0, t + 1
            else:
                s_d = s_d_new
        su[t], sd[t] = s_u, s_d
        if s_u > h or (two_sided and s_d > h):
            alarms.append(t); cps.append(start_u if s_u > h else start_d)
            if reset:
                s_u = s_d = 0.0; start_u = start_d = t + 1
    return dict(stat_up=su, stat_down=sd, alarms=np.array(alarms, dtype=int), change_points=np.array(cps, dtype=int))


def bocpd(x, hazard: float = 1 / 250, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
          beta0: float = 1.0, max_run: int = 500, window: int = 10) -> dict:
    """Adams–MacKay 贝叶斯在线变点检测（未知均值与方差的高斯段，Normal–Gamma 共轭先验）。

    参数
    ----
    hazard : 每步发生变点的先验概率 H（期望段长 1/H）
    mu0, kappa0, alpha0, beta0 : Normal–Gamma 先验超参数（beta0 建议取数据方差量级 × alpha0）
    max_run : 跟踪的最大 run length（截断，节省计算）

    返回 dict：
        prob_change     : (T,) P(r_t < window | x_{1:t})，“最近 window 步内发生过变点”的后验概率。
                          注意：常数 hazard 下 P(r_t = 0 | x_{1:t}) 恒等于 H（变点要在随后几步才被数据确认），
                          所以用短 run length 的累计概率作为报警指标。
        run_length_map  : (T,) 最大后验 run length（变点处会突然归零）
        run_length_mean : (T,) 后验期望 run length
        predictive_mean : (T,) 预测均值（按 run length 后验加权）
    """
    x = as_1d(x, "x")
    T = len(x)
    R = np.zeros(max_run + 1); R[0] = 1.0
    mu = np.array([mu0]); kappa = np.array([kappa0]); alpha = np.array([alpha0]); beta = np.array([beta0])
    prob_change = np.zeros(T); rl_map = np.zeros(T, dtype=int); rl_mean = np.zeros(T); pred_mean = np.zeros(T)
    for t in range(T):
        n_r = len(mu)
        pred_mean[t] = np.sum(R[:n_r] * mu) / max(np.sum(R[:n_r]), 1e-300)
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        pi = stats.t.pdf(x[t], df=2 * alpha, loc=mu, scale=scale)
        growth = R[:n_r] * pi * (1 - hazard)
        cp = np.sum(R[:n_r] * pi * hazard)
        newR = np.zeros(max_run + 1)
        n_keep = min(n_r, max_run)
        newR[1:n_keep + 1] = growth[:n_keep]
        newR[0] = cp
        s = newR.sum()
        R = newR / s if s > 0 else newR
        # 参数更新：每个 run length 的段内统计量加入 x_t；r=0 位置放回先验
        mu_new = (kappa * mu + x[t]) / (kappa + 1)
        beta_new = beta + kappa * (x[t] - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_new[:n_keep]])
        kappa = np.concatenate([[kappa0], (kappa + 1)[:n_keep]])
        alpha = np.concatenate([[alpha0], (alpha + 0.5)[:n_keep]])
        beta = np.concatenate([[beta0], beta_new[:n_keep]])
        prob_change[t] = R[:window].sum()
        rl_map[t] = int(np.argmax(R))
        rl_mean[t] = np.sum(np.arange(max_run + 1) * R)
    return dict(prob_change=prob_change, run_length_map=rl_map, run_length_mean=rl_mean, predictive_mean=pred_mean)
