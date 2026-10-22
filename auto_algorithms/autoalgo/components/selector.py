"""执行规划新算法：``BanditStrategySelector`` —— 用 crossquant 的 LinUCB 上下文老虎机在线学习"什么样的母单该用哪种执行策略"。

接口：quantstack ``TradingPipeline.strategy_selector``（``(order, snapshot, available) -> (strategy_name, params)``）。
奖励：母单完成后 TCA 的实现偏差（implementation shortfall，bps，正数 = 成本）取负并归一化。
上下文：母单相对 ADV 的规模、紧急度、alpha、价差、时长、波动率。

用法::

    sel = BanditStrategySelector(seed=0)
    sel.attach(pipeline)          # 设为流水线的选择器，并订阅 order_done 事件自动回填奖励
    pipeline.run_cycle(panel)     # 之后每一轮都会学习

冷启动：前 ``warmup`` 次（或探索概率 epsilon 命中时）沿用 quantstack 参考规则 ``choose_strategy``，
这样老虎机在没有数据时的行为不劣于基线。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._paths import require

US = 1_000_000
DEFAULT_ARMS = ("twap", "almgren_chriss", "pov", "adaptive_passive", "kalman_pid")


class BanditStrategySelector:
    """LinUCB 上下文老虎机执行策略选择器。

    参数
      arms              候选策略名（运行时与 ``available`` 取交集）
      alpha             LinUCB 探索系数（置信上界宽度）
      ridge             每臂线性模型的岭惩罚
      reward_scale_bps  奖励归一化尺度：reward = −clip(IS, ±3·scale) / scale
      warmup            前多少次选择沿用参考规则（并同时用其结果训练对应臂）
      epsilon           额外的 ε-贪心探索概率
      default_params    各策略的默认参数（如 pov 的 rate）
    """

    def __init__(self, arms: Sequence[str] = DEFAULT_ARMS, alpha: float = 0.6, ridge: float = 1.0, reward_scale_bps: float = 20.0, warmup: int = 2,
                 epsilon: float = 0.05, seed: int = 0, default_params: Optional[Dict[str, Dict[str, float]]] = None, fallback=None):
        require("crossquant")
        from crossquant.control.bandits import LinUCB

        self.arms: List[str] = list(arms)
        self.dim = 8
        self.bandit = LinUCB(n_arms=len(self.arms), dim=self.dim, alpha=float(alpha), ridge=float(ridge))
        self.alpha, self.ridge = float(alpha), float(ridge)
        self.reward_scale_bps, self.warmup, self.epsilon = float(reward_scale_bps), int(warmup), float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.default_params = {"pov": {"rate": 0.1}, **(default_params or {})}
        if fallback is None:
            from quantstack.reference.execution import choose_strategy

            fallback = choose_strategy
        self.fallback = fallback
        self.pending: Dict[str, Tuple[int, np.ndarray, str]] = {}
        self.history: List[Dict[str, float]] = []
        self.n_select = 0
        self.n_update = 0
        self.arm_rewards: Dict[str, List[float]] = {a: [] for a in self.arms}

    # ---------------------------------------------------------------- 上下文特征
    def features(self, order, snapshot) -> np.ndarray:
        adv = snapshot.adv if snapshot is not None and snapshot.adv else 1e6
        vol = snapshot.volatility if snapshot is not None and snapshot.volatility else 0.02
        spread = snapshot.spread_bps if snapshot is not None else 5.0
        pct_adv = order.quantity / max(adv, 1.0)
        span_h = max((order.end_us - order.start_us) / US, 1.0) / 3600.0
        return np.array([
            1.0,
            float(np.clip(math.log10(max(pct_adv, 1e-6)) + 3.0, -3.0, 3.0)) / 3.0,   # 规模（log10 %ADV）
            float(order.urgency),
            float(np.tanh(order.alpha_bps / 10.0)),
            float(np.clip(spread / 5.0, 0.0, 4.0)),
            float(np.clip(span_h, 0.0, 6.5)) / 6.5,
            float(np.clip(vol * 50.0, 0.0, 3.0)),
            float(np.clip(pct_adv * 10.0, 0.0, 3.0)),
        ])

    # ---------------------------------------------------------------- 选择
    def _ucb_scores(self, x: np.ndarray, allowed: List[int]) -> Dict[int, float]:
        out = {}
        for i in allowed:
            Ainv = np.linalg.inv(self.bandit.A[i])
            theta = Ainv @ self.bandit.b[i]
            out[i] = float(theta @ x + self.bandit.alpha * math.sqrt(max(x @ Ainv @ x, 0.0)))
        return out

    def __call__(self, order, snapshot, available: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
        x = self.features(order, snapshot)
        allowed = [i for i, a in enumerate(self.arms) if a in available]
        if not allowed:
            return self.fallback(order, snapshot, available)
        self.n_select += 1
        mode = "ucb"
        if self.n_select <= self.warmup or self.rng.random() < self.epsilon:
            name, params = self.fallback(order, snapshot, available)
            mode = "warmup" if self.n_select <= self.warmup else "explore"
            if name not in self.arms:
                self.pending[order.order_id] = (-1, x, name)
                return name, params
            arm = self.arms.index(name)
        else:
            scores = self._ucb_scores(x, allowed)
            arm = max(scores, key=scores.get)
            name = self.arms[arm]
            params = dict(self.default_params.get(name, {}))
        self.pending[order.order_id] = (arm, x, mode)
        return name, dict(params)

    # ---------------------------------------------------------------- 学习
    def update(self, order_id: str, shortfall_bps: float) -> None:
        item = self.pending.pop(order_id, None)
        if item is None or not np.isfinite(shortfall_bps):
            return
        arm, x, mode = item
        if arm < 0:
            return
        s = self.reward_scale_bps
        reward = -float(np.clip(shortfall_bps, -3 * s, 3 * s)) / s
        self.bandit.update(arm, x, reward)
        self.n_update += 1
        self.arm_rewards[self.arms[arm]].append(reward)
        self.history.append({"arm": float(arm), "reward": reward, "shortfall_bps": float(shortfall_bps)})

    def attach(self, pipeline) -> "BanditStrategySelector":
        """接入 quantstack 流水线：设为 strategy_selector 并订阅 order_done 以自动回填奖励。"""
        pipeline.strategy_selector = self

        def _on_done(event):
            payload = event.payload or {}
            oid = payload.get("order_id") if isinstance(payload, dict) else None
            if not oid:
                return
            rep = pipeline.tca.report(oid)
            if rep is not None and rep.filled_quantity > 0:
                self.update(oid, rep.implementation_shortfall_bps)
            else:
                self.update(oid, 3 * self.reward_scale_bps)  # 完全未成交按最差成本计

        pipeline.bus.subscribe("order_done", _on_done)
        return self

    # ---------------------------------------------------------------- 解释
    def theta(self) -> Dict[str, List[float]]:
        return {a: [float(v) for v in np.linalg.solve(self.bandit.A[i], self.bandit.b[i])] for i, a in enumerate(self.arms)}

    def describe(self) -> Dict[str, object]:
        return {
            "library": "autoalgo", "family": "selector.bandit", "arms": list(self.arms), "alpha": self.alpha, "ridge": self.ridge,
            "reward_scale_bps": self.reward_scale_bps, "warmup": self.warmup, "epsilon": self.epsilon, "n_select": self.n_select, "n_update": self.n_update,
            "mean_reward_per_arm": {a: float(np.mean(r)) if r else None for a, r in self.arm_rewards.items()},
            "n_per_arm": {a: len(r) for a, r in self.arm_rewards.items()},
            "feature_names": ["bias", "log_pct_adv", "urgency", "tanh_alpha", "spread", "duration", "vol", "pct_adv"],
            "theta": self.theta(),
        }
