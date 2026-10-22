"""
端到端流水线：市场状态（regime）仪表盘
=====================================

把多个来自不同学科的状态指标对齐到同一时间轴：
- Hurst（DFA，物理标度）：趣势 vs 均值回复
- 排列熵（信息论）：序列有序程度
- 滚动 W₁ 距离（最优传输）：分布漂移
- Fisher–Rao 距离（信息几何）：(μ, σ) 偏离基准的“真实距离”
- BOCPD 变点概率（贝叶斯信号处理）
"""
from __future__ import annotations

import numpy as np

from ..control.changepoint import bocpd
from ..mathematics.information_geometry import rolling_fisher_rao
from ..mathematics.optimal_transport import rolling_wasserstein
from ..physics.entropy import permutation_entropy
from ..physics.hurst import dfa


def regime_dashboard(returns, window: int = 120, step: int = 5, hazard: float = 1 / 250) -> dict:
    """返回按 ``index`` 对齐的指标数组 dict。"""
    r = np.asarray(returns, dtype=float)
    idx, H, PE = [], [], []
    for end in range(window, len(r) + 1, step):
        seg = r[end - window:end]
        idx.append(end - 1); H.append(dfa(seg, min_scale=6)["H"]); PE.append(permutation_entropy(seg, order=4))
    idx = np.array(idx)
    w1 = rolling_wasserstein(r, window, step=step)
    fr = rolling_fisher_rao(r, window)
    bo = bocpd(r / (r.std() + 1e-12), hazard=hazard, beta0=1.0)
    return dict(index=idx, hurst=np.array(H), permutation_entropy=np.array(PE), wasserstein=w1["distance"],
                fisher_rao=fr["distance"][idx - window], bocpd_change_prob=bo["prob_change"][idx],
                bocpd_run_length=bo["run_length_map"][idx])
