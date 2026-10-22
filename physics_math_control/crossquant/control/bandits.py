"""
多臂老虎机（序贯决策 / 对偶控制）
=================================

学科来源
--------
序贯试验设计与控制论中的“探索–利用”权衡（Thompson 1933；Robbins 1952；Lai & Robbins 1985 的对数
遗憾下界；Auer 等 2002 的 UCB1；Agrawal & Goyal 2012 证明 Thompson 采样的最优遗憾）。
它是“带学习的控制”（dual control, Feldbaum 1960）最简单、可证明最优的形式。

量化映射
--------
1. **交易所/场所路由**：每个场所是一个臂，奖励 = 成交率、实现价差或负延迟 ⇒ 在线选择最优路由，
   并随市况漂移自动再探索（配合折扣/滑窗）。
2. **策略 / 参数分配**：多个 alpha 或做市价差参数之间的资金分配（奖励 = 风险调整收益）。
3. **执行算法选择**：对同类母单在 TWAP / VWAP / 冰山 之间选择（LinUCB 把订单特征当作上下文）。

算法
----
- UCB1：选 argmax_a [ μ̂_a + sqrt(2 ln t / n_a) ]
- Thompson（Bernoulli/Gaussian）：从每个臂的后验抽样，选最大者
- EXP3（对抗环境）：指数权重 + 均匀探索
- LinUCB（上下文）：奖励 = θᵀx + 噪声，置信椭球上界 θ̂ᵀx + α sqrt(xᵀA⁻¹x)
- 折扣（discount<1）使估计跟随非平稳环境

参考文献
--------
- Lattimore & Szepesvári (2020). Bandit Algorithms. CUP.
- Auer, Cesa-Bianchi & Fischer (2002). Finite-time Analysis of the Multiarmed Bandit Problem.
- Agrawal & Goyal (2012). Analysis of Thompson Sampling for the Multi-armed Bandit Problem.
- Li, Chu, Langford & Schapire (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation (LinUCB).
"""
from __future__ import annotations

import numpy as np

from .._utils import as_rng


class UCB1:
    """UCB1；``discount<1`` 时为折扣 UCB（非平稳环境）。"""

    def __init__(self, n_arms: int, c: float = 2.0, discount: float = 1.0):
        self.n_arms, self.c, self.discount = n_arms, c, discount
        self.counts = np.zeros(n_arms); self.sums = np.zeros(n_arms); self.t = 0

    def select(self) -> int:
        self.t += 1
        if np.any(self.counts == 0):
            return int(np.argmin(self.counts))
        means = self.sums / self.counts
        bonus = np.sqrt(self.c * np.log(self.counts.sum()) / self.counts)
        return int(np.argmax(means + bonus))

    def update(self, arm: int, reward: float):
        self.counts *= self.discount; self.sums *= self.discount
        self.counts[arm] += 1; self.sums[arm] += reward


class ThompsonBernoulli:
    """Beta–Bernoulli Thompson 采样（奖励 ∈ {0,1}，如“是否成交”）。"""

    def __init__(self, n_arms: int, seed=None, discount: float = 1.0):
        self.a = np.ones(n_arms); self.b = np.ones(n_arms); self.rng = as_rng(seed); self.discount = discount

    def select(self) -> int:
        return int(np.argmax(self.rng.beta(self.a, self.b)))

    def update(self, arm: int, reward: float):
        self.a = 1 + (self.a - 1) * self.discount; self.b = 1 + (self.b - 1) * self.discount
        self.a[arm] += reward; self.b[arm] += 1 - reward


class ThompsonGaussian:
    """高斯 Thompson 采样（奖励连续，如 −延迟、实现价差），均值未知、观测方差已知。"""

    def __init__(self, n_arms: int, obs_var: float = 1.0, prior_var: float = 100.0, seed=None, discount: float = 1.0):
        self.n = np.zeros(n_arms); self.sum = np.zeros(n_arms)
        self.obs_var, self.prior_var, self.discount = obs_var, prior_var, discount
        self.rng = as_rng(seed)

    def posterior(self):
        prec = 1 / self.prior_var + self.n / self.obs_var
        mean = (self.sum / self.obs_var) / prec
        return mean, 1 / prec

    def select(self) -> int:
        m, v = self.posterior()
        return int(np.argmax(self.rng.normal(m, np.sqrt(v))))

    def update(self, arm: int, reward: float):
        self.n *= self.discount; self.sum *= self.discount
        self.n[arm] += 1; self.sum[arm] += reward


class EXP3:
    """EXP3（对抗性老虎机）：奖励需缩放到 [0,1]。"""

    def __init__(self, n_arms: int, gamma: float = 0.1, seed=None):
        self.w = np.ones(n_arms); self.gamma = gamma; self.rng = as_rng(seed); self.p = None

    def select(self) -> int:
        K = len(self.w)
        self.p = (1 - self.gamma) * self.w / self.w.sum() + self.gamma / K
        return int(self.rng.choice(K, p=self.p))

    def update(self, arm: int, reward: float):
        K = len(self.w)
        xhat = reward / self.p[arm]
        self.w[arm] *= np.exp(self.gamma * xhat / K)
        self.w /= self.w.max()


class LinUCB:
    """LinUCB（不相交模型）：每个臂有自己的线性模型 r = θ_aᵀx。"""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, ridge: float = 1.0):
        self.A = np.array([np.eye(dim) * ridge for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim)); self.alpha = alpha

    def select(self, x) -> int:
        x = np.asarray(x, dtype=float)
        scores = []
        for A, b in zip(self.A, self.b):
            Ainv = np.linalg.inv(A); theta = Ainv @ b
            scores.append(theta @ x + self.alpha * np.sqrt(x @ Ainv @ x))
        return int(np.argmax(scores))

    def update(self, arm: int, x, reward: float):
        x = np.asarray(x, dtype=float)
        self.A[arm] += np.outer(x, x); self.b[arm] += reward * x


def simulate_bandit(agent, reward_fn, T: int, seed=None, contexts=None) -> dict:
    """通用老虎机仿真。reward_fn(arm, rng[, context]) -> reward；返回 ``arms``、``rewards``、``regret``（需 reward_fn.means）。"""
    rng = as_rng(seed)
    arms = np.zeros(T, dtype=int); rewards = np.zeros(T)
    for t in range(T):
        if contexts is not None:
            arm = agent.select(contexts[t]); r = reward_fn(arm, rng, contexts[t]); agent.update(arm, contexts[t], r)
        else:
            arm = agent.select(); r = reward_fn(arm, rng); agent.update(arm, r)
        arms[t], rewards[t] = arm, r
    out = dict(arms=arms, rewards=rewards)
    means = getattr(reward_fn, "means", None)
    if means is not None:
        means = np.asarray(means)
        out["regret"] = np.cumsum(means.max() - means[arms])
    return out


def venue_fill_reward(fill_probs):
    """构造“场所成交概率”奖励函数（Bernoulli），附带 ``means`` 便于算遗憾。"""
    p = np.asarray(fill_probs, dtype=float)

    def fn(arm, rng):
        return float(rng.random() < p[arm])
    fn.means = p
    return fn
