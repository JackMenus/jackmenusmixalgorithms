"""
排队论与限价订单簿：成交概率、队列竞速与马尔可夫 LOB
=====================================================

学科来源
--------
Erlang (1909) 为电话交换机建立排队论；生灭过程、竞争指数时钟是其基本工具。
Cont, Stoikov & Talreja (2010) 把限价订单簿建成多维生灭过程（Markovian LOB），
用 Laplace 变换得到“中间价下一步上涨的概率”“订单在价格变动前成交的概率”等闭式/半闭式结果；
Huang, Lehalle & Rosenbaum (2015) 的 queue-reactive 模型让到达率依赖队列状态。

量化映射
--------
- **挂单 vs 吃单决策**：给定我前面还有 q 份、市价单到达率 λ_MO、每单撤单率 λ_C、价格变动率 ν，
  被动成交概率 P(fill) 与期望等待时间决定是否值得排队；
- **做市报价**：成交概率随队列位置的衰减替代 A e^{−kδ} 的粗糙强度；
- **订单簿不平衡信号**：买卖队列 (q_b, q_a) ⇒ P(下一次中间价上涨)，CST 的经典结果；
- **路由**：不同场所的队列长度与到达率 ⇒ 期望成交时间比较。

核心公式（纯死亡近似：我前面的队列只减不增）
---------------------------------------------
处于第 j 位时总消耗率 μ_j = λ_MO + j λ_C，价格变动率 ν 为竞争时钟：
    P(在价格变动前排到队首并成交) = Π_{j=1}^{q} μ_j / (μ_j + ν)
    E[排到队首时间] = Σ_{j=1}^{q} 1/μ_j

参考文献
--------
- Cont, Stoikov & Talreja (2010). A Stochastic Model for Order Book Dynamics. Operations Research.
- Huang, Lehalle & Rosenbaum (2015). Simulating and Analyzing Order Book Data: The Queue-Reactive Model. JASA.
- Abergel, Anane, Chakraborti, Jedidi & Muni Toke (2016). Limit Order Books. CUP.
- Moallemi & Yuan (2016). A Model for Queue Position Valuation in a Limit Order Book.
"""
from __future__ import annotations

import numpy as np

from .._utils import as_rng


def fill_probability_pure_death(q_ahead: int, lam_mo: float, lam_cancel: float, nu_move: float) -> float:
    """我前面有 q_ahead 份时，在价格变动（率 ν）前成交的概率（纯死亡近似）。"""
    j = np.arange(1, q_ahead + 1)
    mu = lam_mo + j * lam_cancel
    p_front = np.prod(mu / (mu + nu_move)) if q_ahead > 0 else 1.0
    # 排到队首后还需等一个市价单（率 λ_MO）先于价格变动到达
    return float(p_front * lam_mo / (lam_mo + nu_move))


def expected_time_to_front(q_ahead: int, lam_mo: float, lam_cancel: float) -> float:
    j = np.arange(1, q_ahead + 1)
    return float(np.sum(1.0 / (lam_mo + j * lam_cancel))) if q_ahead > 0 else 0.0


def queue_race_mc(q_ahead: int, lam_mo: float, lam_cancel: float, nu_move: float, n_sims: int = 20000, seed=None) -> dict:
    """蒙特卡洛验证纯死亡模型：返回 ``p_fill``、``mean_fill_time``。"""
    rng = as_rng(seed)
    fills = 0; times = []
    for _ in range(n_sims):
        q = q_ahead; t = 0.0; done = False
        while True:
            mu = lam_mo + q * lam_cancel
            total = mu + nu_move
            t += rng.exponential(1 / total)
            if rng.random() < nu_move / total:
                break
            if q == 0:
                fills += 1; times.append(t); done = True; break
            q -= 1
    return dict(p_fill=fills / n_sims, mean_fill_time=float(np.mean(times)) if times else np.nan)


def simulate_markov_lob(lam_lo: float, mu_mo: float, theta_cancel: float, q_bid0: int, q_ask0: int, T: float,
                        tick: float = 0.01, mid0: float = 100.0, seed=None, refill_mean: float | None = None) -> dict:
    """单档马尔可夫 LOB（CST 的最简版本）：买卖队列各为生灭过程，队列耗尽时中间价移动一个 tick 并重新补队。

    事件率：限价单 λ_LO（队列 +1）、市价单 μ_MO（−1）、撤单 θ·q（−1）。
    返回 ``times``、``mid``、``q_bid``、``q_ask``。
    """
    rng = as_rng(seed)
    refill_mean = (q_bid0 + q_ask0) / 2 if refill_mean is None else refill_mean
    t = 0.0; qb, qa, mid = q_bid0, q_ask0, mid0
    times, mids, qbs, qas = [0.0], [mid], [qb], [qa]
    while t < T:
        rates = np.array([lam_lo, mu_mo + theta_cancel * qb, lam_lo, mu_mo + theta_cancel * qa])
        total = rates.sum()
        t += rng.exponential(1 / total)
        ev = rng.choice(4, p=rates / total)
        if ev == 0:
            qb += 1
        elif ev == 1:
            qb -= 1
        elif ev == 2:
            qa += 1
        else:
            qa -= 1
        if qb <= 0:
            mid -= tick; qb = max(1, rng.poisson(refill_mean)); qa = max(1, rng.poisson(refill_mean))
        if qa <= 0:
            mid += tick; qa = max(1, rng.poisson(refill_mean)); qb = max(1, rng.poisson(refill_mean))
        times.append(t); mids.append(mid); qbs.append(qb); qas.append(qa)
    return dict(times=np.array(times), mid=np.array(mids), q_bid=np.array(qbs), q_ask=np.array(qas))


def prob_mid_up(q_bid: int, q_ask: int, lam_lo: float, mu_mo: float, theta_cancel: float, n_sims: int = 5000, seed=None) -> float:
    """给定买卖队列 (q_b, q_a)，下一次中间价变动为上涨（卖队列先耗尽）的概率（蒙特卡洛，CST 命题的数值版）。"""
    rng = as_rng(seed)
    ups = 0
    for _ in range(n_sims):
        qb, qa = q_bid, q_ask
        while qb > 0 and qa > 0:
            rates = np.array([lam_lo, mu_mo + theta_cancel * qb, lam_lo, mu_mo + theta_cancel * qa])
            ev = rng.choice(4, p=rates / rates.sum())
            qb += 1 if ev == 0 else (-1 if ev == 1 else 0)
            qa += 1 if ev == 2 else (-1 if ev == 3 else 0)
        ups += qa <= 0
    return ups / n_sims
