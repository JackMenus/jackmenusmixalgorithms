"""强化学习最优执行：环境 + 表格 Q-learning 基线。

环境遵循 Gymnasium 风格接口（reset/step），不依赖 gym，便于替换为
DDQN（Ning, Lin & Jaimungal 2021）、PPO 等深度方法：``obs`` 同时提供离散状态和连续向量。

动作：本区间成交量 = TWAP 基准量 × ACTIONS[a]。
奖励：−（相对到达价的执行成本），含临时冲击 η·v² 与永久冲击；
到期未完成部分强制以惩罚性冲击清算。
基线：Nevmyvaka, Feng & Kearns (2006) 的表格 RL 执行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

ACTIONS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])


@dataclass
class ExecEnvConfig:
    X: float = 1000.0
    N: int = 10
    s0: float = 100.0
    sigma: float = 0.05     # 每区间价格标准差
    eta: float = 2e-4       # 临时冲击：价格让步 = η·v
    gamma: float = 2e-5     # 永久冲击
    drift_kappa: float = 0.3  # 可观测短期漂移（模拟 alpha）的均值回复
    drift_vol: float = 0.02
    risk_aversion: float = 0.0
    inv_bins: int = 10
    drift_bins: int = 3


class ExecutionEnv:
    """卖出 X 股。"""

    def __init__(self, cfg: ExecEnvConfig = ExecEnvConfig(), seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def reset(self) -> Tuple[Tuple[int, int, int], np.ndarray]:
        c = self.cfg
        self.t = 0
        self.x = c.X
        self.s = c.s0
        self.mu = 0.0
        return self._obs()

    def _obs(self):
        c = self.cfg
        ib = min(int(self.x / c.X * c.inv_bins), c.inv_bins - 1) if self.x > 0 else 0
        db = 0 if self.mu < -c.drift_vol * 0.5 else (2 if self.mu > c.drift_vol * 0.5 else 1)
        vec = np.array([self.t / c.N, self.x / c.X, self.mu / max(c.drift_vol, 1e-12)])
        return (self.t, ib, db), vec

    def step(self, a: int):
        c = self.cfg
        base = c.X / c.N
        last = self.t == c.N - 1
        v = self.x if last else min(self.x, base * ACTIONS[a])
        exec_px = self.s - c.eta * v
        # 逐步盯市奖励：本步成交相对当前价 + 剩余持仓的价格变动。
        # 各步求和恰为相对到达价的总收益（望远镜求和），但方差远低于直接用到达价。
        reward = v * (exec_px - self.s)
        self.x -= v
        s_new = self.s - c.gamma * v + self.mu + c.sigma * self.rng.standard_normal()
        reward += self.x * (s_new - self.s)
        self.s = s_new
        reward -= c.risk_aversion * (self.x * c.sigma) ** 2
        self.mu += -c.drift_kappa * self.mu + c.drift_vol * self.rng.standard_normal()
        self.t += 1
        done = self.t >= c.N or self.x <= 1e-9
        return self._obs(), float(reward), done, {"executed": v, "price": exec_px}


class QLearningExecutor:
    def __init__(self, cfg: ExecEnvConfig, lr: float = 0.1, eps: float = 0.2, seed: int = 0):
        self.cfg = cfg
        self.Q = np.zeros((cfg.N, cfg.inv_bins, cfg.drift_bins, len(ACTIONS)))
        self.lr, self.eps = lr, eps
        self.rng = np.random.default_rng(seed)

    def act(self, s, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(len(ACTIONS)))
        return int(np.argmax(self.Q[s]))

    def train(self, env: ExecutionEnv, episodes: int = 20000) -> "QLearningExecutor":
        scale = 1.0 / (self.cfg.X * self.cfg.s0 * 1e-4)  # 奖励归一到 bps 量级
        for ep in range(episodes):
            (s, _), done = env.reset(), False
            self.eps = max(0.02, 0.3 * (1 - ep / episodes))
            while not done:
                a = self.act(s)
                (s2, _), r, done, _ = env.step(a)
                target = r * scale + (0.0 if done else self.Q[s2].max())
                self.Q[s + (a,)] += self.lr * (target - self.Q[s + (a,)])
                s = s2
        return self

    def evaluate(self, env: ExecutionEnv, episodes: int = 2000, policy: str = "q") -> Dict[str, float]:
        costs = []
        for _ in range(episodes):
            (s, _), done, total = env.reset(), False, 0.0
            while not done:
                a = self.act(s, greedy=True) if policy == "q" else int(np.where(ACTIONS == 1.0)[0][0])
                (s, _), r, done, _ = env.step(a)
                total += r
            costs.append(-total / (self.cfg.X * self.cfg.s0) * 1e4)
        c = np.array(costs)
        return {"mean_cost_bps": float(c.mean()), "std_cost_bps": float(c.std())}
