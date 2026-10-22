"""深度学习因子（非公式化）：用神经网络把一组输入特征映射为横截面得分。

对应前沿方向：
  * Gu, Kelly & Xiu（RFS 2020）“Empirical Asset Pricing via Machine Learning”：MLP 横截面预测；
  * FactorVAE（AAAI 2022）、HIST（2021）、MASTER（AAAI 2024）：概率/图/Transformer 的动态因子模型；
  * DoubleAdapt（KDD 2023）：面向分布漂移的增量学习。

这里提供一个零依赖 numpy MLP 作为基线与接口样板，损失为“横截面 IC 最大化”
（直接优化排序相关而非 MSE，与 AlphaForge 的生成-预测思路一致）。
需要更强模型时，只需实现同样的 fit / predict 接口（可换成 PyTorch 版 FactorVAE、MASTER 等）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .combine import standardize


@dataclass
class MLPConfig:
    hidden: int = 32
    epochs: int = 30
    lr: float = 3e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    seed: int = 0


class NeuralFactor:
    """输入：{特征名: date×asset DataFrame}（可以是已挖出的公式因子）；输出：date×asset 得分。"""

    def __init__(self, config: MLPConfig = MLPConfig()):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.names: List[str] = []
        self.params: Dict[str, np.ndarray] = {}
        self.history: List[float] = []

    def _tensor(self, feats: Dict[str, pd.DataFrame]) -> np.ndarray:
        return np.nan_to_num(np.stack([standardize(feats[n]).to_numpy(float) for n in self.names], -1))

    def _forward(self, X, train=False):
        p = self.params
        h = X @ p["W1"] + p["b1"]
        a = np.tanh(h)
        mask = None
        if train and self.cfg.dropout > 0:
            mask = (self.rng.random(a.shape) > self.cfg.dropout) / (1 - self.cfg.dropout)
            a = a * mask
        return a @ p["W2"] + p["b2"], (h, a, mask)

    def fit(self, feats: Dict[str, pd.DataFrame], fwd: pd.DataFrame, rows: Optional[slice] = None) -> "NeuralFactor":
        self.names = list(feats)
        X = self._tensor(feats)
        Y = standardize(fwd).to_numpy(float)
        if rows is not None:
            X, Y = X[rows], Y[rows]
        K, H = X.shape[-1], self.cfg.hidden
        self.params = {
            "W1": self.rng.normal(0, 1 / np.sqrt(K), (K, H)), "b1": np.zeros(H),
            "W2": self.rng.normal(0, 1 / np.sqrt(H), (H, 1)), "b2": np.zeros(1),
        }
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v2 = {k: np.zeros_like(v) for k, v in self.params.items()}
        step = 0
        for ep in range(self.cfg.epochs):
            ics = []
            for t in self.rng.permutation(len(X)):
                x, y = X[t], Y[t]
                ok = np.isfinite(y)
                if ok.sum() < 10:
                    continue
                x, y = x[ok], y[ok]
                out, (h, a, mask) = self._forward(x, train=True)
                s = out[:, 0]
                # 损失 = -corr(s, y)；对 s 求梯度
                sd, yd = s - s.mean(), y - y.mean()
                ns, ny = np.linalg.norm(sd) + 1e-12, np.linalg.norm(yd) + 1e-12
                c = sd @ yd / (ns * ny)
                ics.append(c)
                g_sd = -(yd / (ns * ny) - c * sd / ns**2)
                g_s = g_sd - g_sd.mean()
                g = {"W2": a.T @ g_s[:, None], "b2": np.array([g_s.sum()])}
                da = g_s[:, None] @ self.params["W2"].T
                if mask is not None:
                    da = da * mask
                dh = da * (1 - np.tanh(h) ** 2)
                g["W1"] = x.T @ dh
                g["b1"] = dh.sum(0)
                step += 1
                for k in self.params:
                    gk = g[k] + self.cfg.weight_decay * self.params[k]
                    m[k] = 0.9 * m[k] + 0.1 * gk
                    v2[k] = 0.999 * v2[k] + 0.001 * gk * gk
                    self.params[k] -= self.cfg.lr * (m[k] / (1 - 0.9**step)) / (np.sqrt(v2[k] / (1 - 0.999**step)) + 1e-8)
            self.history.append(float(np.mean(ics)) if ics else np.nan)
        return self

    def predict(self, feats: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        ref = feats[self.names[0]]
        out, _ = self._forward(self._tensor(feats))
        return pd.DataFrame(out[..., 0], index=ref.index, columns=ref.columns)
