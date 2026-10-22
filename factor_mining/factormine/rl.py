"""强化学习公式化因子挖掘（AlphaGen 范式的轻量 numpy 实现）。

参考：
  * AlphaGen（Yu et al., KDD 2023）：把公式写成逆波兰（RPN）记号序列，策略网络逐个生成记号，
    奖励 = 新因子加入因子池后“合成因子 IC”的提升，从而直接优化因子集合的协同性；
  * QuantFactor REINFORCE（Zhao et al., 2024）：公式生成是确定性 MDP，REINFORCE + 基线
    比 PPO 更合适，且方差可控；
  * Alpha²（Xu et al., 2024）：用语法约束屏蔽非法动作，保证公式合法。

为保持零依赖，这里的策略是“上下文线性 softmax”（上一记号 × 栈深 × 位置特征），
接口与 AlphaGen 一致，想换成 LSTM/Transformer 策略只需实现 ``Policy`` 的 act/update。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .expression import Expr
from .library import FactorPool
from .operators import OPERATORS


@dataclass
class Token:
    kind: str  # field / unary / binary / end
    name: str
    op: Optional[str] = None
    param: Optional[float] = None  # 窗口或常数

    def __str__(self) -> str:
        return self.name


def build_vocab(
    fields: Sequence[str] = ("open", "high", "low", "close", "volume", "vwap", "returns"),
    windows: Sequence[int] = (5, 10, 20, 40),
    unary_ops: Sequence[str] = (
        "neg", "abs", "log", "sign", "cs_rank", "ts_mean", "ts_std", "ts_rank", "delta",
        "ts_pct", "ts_zscore", "ts_max", "ts_min", "ts_decay_linear", "ts_skew",
    ),
    binary_ops: Sequence[str] = ("add", "sub", "mul", "div", "ts_corr", "ts_cov"),
) -> List[Token]:
    """记号表：字段 + 一元算子(带窗口展开) + 二元算子(带窗口展开) + 结束符。"""
    vocab = [Token("field", f) for f in fields]
    for op in unary_ops:
        ar = OPERATORS[op].arity
        if ar == "x":
            vocab.append(Token("unary", op, op))
        elif ar == "xw":
            vocab += [Token("unary", f"{op}_{w}", op, w) for w in windows]
    for op in binary_ops:
        ar = OPERATORS[op].arity
        if ar == "xx":
            vocab.append(Token("binary", op, op))
        elif ar == "xxw":
            vocab += [Token("binary", f"{op}_{w}", op, w) for w in windows]
    vocab.append(Token("end", "<END>"))
    return vocab


def rpn_to_expr(tokens: Sequence[Token]) -> Expr:
    stack: List[Expr] = []
    for t in tokens:
        if t.kind == "field":
            stack.append(Expr.feat(t.name))
        elif t.kind == "unary":
            a = stack.pop()
            args = [a] + ([Expr.const(t.param)] if t.param is not None else [])
            stack.append(Expr.call(t.op, *args))
        elif t.kind == "binary":
            b, a = stack.pop(), stack.pop()
            args = [a, b] + ([Expr.const(t.param)] if t.param is not None else [])
            stack.append(Expr.call(t.op, *args))
    if len(stack) != 1:
        raise ValueError("非法 RPN 序列")
    return stack[0]


class LinearSoftmaxPolicy:
    """π(a|s) = softmax(W·φ(s))，φ = [上一记号 one-hot, 栈深 one-hot, 位置 one-hot, 1]。Adam 更新。"""

    def __init__(self, vocab_size: int, max_len: int, lr: float = 0.05, seed: int = 0):
        self.V, self.L = vocab_size, max_len
        self.D = (vocab_size + 1) + (max_len + 2) + (max_len + 1) + 1
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((self.D, vocab_size))
        self.m = np.zeros_like(self.W)
        self.v = np.zeros_like(self.W)
        self.lr, self.t = lr, 0

    def features(self, prev: int, depth: int, pos: int) -> np.ndarray:
        x = np.zeros(self.D)
        x[prev] = 1  # prev = V 表示 BOS
        x[self.V + 1 + min(depth, self.L + 1)] = 1
        x[self.V + 1 + self.L + 2 + pos] = 1
        x[-1] = 1
        return x

    def probs(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        z = x @ self.W
        z = np.where(mask, z, -np.inf)
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    def act(self, x, mask) -> Tuple[int, np.ndarray]:
        p = self.probs(x, mask)
        return int(self.rng.choice(self.V, p=p)), p

    def update(self, trajectories: List[Tuple[List[Tuple[np.ndarray, int, np.ndarray]], float]], entropy: float = 0.01):
        g = np.zeros_like(self.W)
        for steps, adv in trajectories:
            for x, a, p in steps:
                d = -p.copy()
                d[a] += 1.0
                # 熵正则梯度（鼓励探索）：∂H/∂z = -p(log p + H)
                lp = np.log(np.where(p > 0, p, 1))
                H = -(p * lp).sum()
                d_ent = -p * (lp + H)
                g += np.outer(x, adv * d + entropy * d_ent)
        g /= max(len(trajectories), 1)
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        mh = self.m / (1 - 0.9**self.t)
        vh = self.v / (1 - 0.999**self.t)
        self.W += self.lr * mh / (np.sqrt(vh) + 1e-8)  # 梯度上升


@dataclass
class RLConfig:
    max_len: int = 12
    iterations: int = 30
    batch_size: int = 32
    lr: float = 0.05
    entropy: float = 0.02
    invalid_penalty: float = -0.02
    seed: int = 0


class RLMiner:
    def __init__(self, pool: FactorPool, vocab: Optional[List[Token]] = None, config: RLConfig = RLConfig(),
                 log: Callable[[str], None] = print):
        self.pool = pool
        self.vocab = vocab or build_vocab()
        self.cfg = config
        self.policy = LinearSoftmaxPolicy(len(self.vocab), config.max_len, config.lr, config.seed)
        self.baseline = 0.0
        self.log = log
        self.history: List[dict] = []
        self._base_key, self._base = None, 0.0
        self._kinds = np.array([t.kind for t in self.vocab])

    def _mask(self, depth: int, pos: int) -> np.ndarray:
        remaining = self.cfg.max_len - pos  # 含本步
        m = np.zeros(len(self.vocab), bool)
        m |= (self._kinds == "field") & (depth + 1 <= remaining)  # 压栈后仍能在剩余步数内归约
        m |= (self._kinds == "unary") & (depth >= 1) & (depth <= remaining)
        m |= (self._kinds == "binary") & (depth >= 2)
        m |= (self._kinds == "end") & (depth == 1)
        return m

    def sample(self) -> Tuple[List[Token], List[Tuple[np.ndarray, int, np.ndarray]]]:
        seq, steps = [], []
        prev, depth = len(self.vocab), 0
        for pos in range(self.cfg.max_len + 1):
            mask = self._mask(depth, pos)
            if pos == self.cfg.max_len:
                break
            x = self.policy.features(prev, depth, pos)
            a, p = self.policy.act(x, mask)
            steps.append((x, a, p))
            tok = self.vocab[a]
            if tok.kind == "end":
                break
            seq.append(tok)
            depth += {"field": 1, "unary": 0, "binary": -1}[tok.kind]
            prev = a
        return seq, steps

    def reward(self, e: Expr) -> float:
        ic, v = self.pool.score(e)
        if v is None:
            return self.cfg.invalid_penalty
        if len(self.pool) == 0:
            r = abs(ic)
        else:
            key = tuple(str(x) for x in self.pool.exprs)
            if key != self._base_key:
                self._base_key, self._base = key, self.pool.combined_ic()
            base = self._base
            r = abs(self.pool.combined_ic(extra=v)) - abs(base)
        self.pool.try_add(e, source="rl")
        return float(r)

    def run(self) -> FactorPool:
        for it in range(self.cfg.iterations):
            batch, rewards = [], []
            for _ in range(self.cfg.batch_size):
                seq, steps = self.sample()
                try:
                    e = rpn_to_expr(seq)
                    r = self.reward(e)
                except Exception:
                    r = self.cfg.invalid_penalty
                batch.append(steps)
                rewards.append(r)
            rw = np.array(rewards)
            self.baseline = 0.8 * self.baseline + 0.2 * rw.mean() if it else rw.mean()
            scale = rw.std() + 1e-6
            self.policy.update([(s, (r - self.baseline) / scale) for s, r in zip(batch, rw)], self.cfg.entropy)
            self.history.append({"iter": it, "mean_reward": float(rw.mean()), "pool": len(self.pool)})
            self.log(f"[RL] 第 {it} 轮  平均奖励={rw.mean():+.4f}  最大={rw.max():+.4f}  池大小={len(self.pool)}")
        return self.pool
