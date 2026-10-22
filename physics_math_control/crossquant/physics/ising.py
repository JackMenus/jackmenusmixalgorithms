"""
Ising 自旋模型与羊群效应（教学模块）
===================================

学科来源
--------
Ising (1925) 的铁磁模型：自旋 s_i = ±1 通过近邻耦合 J 趋同，温度 T 带来随机翻转；
在临界温度附近出现大尺度关联与幂律涨落。Bornholdt (2001) 加入全局反耦合项 −α s_i |M|
（“少数派”动机：当市场过度一致时，与多数反向更有利），模型自发产生波动率聚集与厚尾收益。

量化映射
--------
- 直观理解羊群与反转共存如何产生厚尾、聚集等“风格化事实”；
- 为基于 agent 的市场模拟器（压力测试、策略交互）提供最小模型；
- 临界慢化、磁化率上升等指标与“市场脆弱性”类比。

参考文献
--------
- Bornholdt (2001). Expectation bubbles in a spin model of markets. IJMPC.
- Sornette (2014). Physics and financial economics (1776–2014): puzzles, Ising and agent-based models. Rep. Prog. Phys.
"""
from __future__ import annotations

import numpy as np

from .._utils import as_rng


def bornholdt_ising(L: int = 32, n_sweeps: int = 2000, J: float = 1.0, alpha: float = 4.0, beta: float = 1.0 / 1.5,
                    seed=None, burn_in: int = 200) -> dict:
    """Bornholdt 自旋市场模型（二维周期格点，热浴更新）。

    返回 ``magnetization``（每次 sweep 后的 M = mean(s)，可视作市场净头寸）、``returns``（ΔM 作为收益代理）。
    """
    rng = as_rng(seed)
    s = rng.choice([-1, 1], size=(L, L))
    M_series = []
    for sweep in range(n_sweeps + burn_in):
        M = s.mean()
        for _ in range(L * L):
            i, j = rng.integers(L), rng.integers(L)
            nb = s[(i + 1) % L, j] + s[(i - 1) % L, j] + s[i, (j + 1) % L] + s[i, (j - 1) % L]
            h = J * nb - alpha * s[i, j] * abs(M)
            p_up = 1.0 / (1.0 + np.exp(-2 * beta * h))
            s[i, j] = 1 if rng.random() < p_up else -1
        if sweep >= burn_in:
            M_series.append(s.mean())
    M_series = np.array(M_series)
    return dict(magnetization=M_series, returns=np.diff(M_series), final_spins=s)
