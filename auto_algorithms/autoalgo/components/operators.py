"""跨学科因子算子：把 crossquant 中的物理 / 数学 / 控制工具改写成 factormine 的面板算子（``DataFrame(时间×资产)`` 进出），
并通过 ``factormine.register`` 注入其算子表，从而让 GP / RL / LLM 挖掘器能在表达式中直接使用它们。

| 算子 | arity | 学科来源（crossquant 模块） | 含义 |
|---|---|---|---|
| ``ts_fracdiff(x, d)``      | xc  | mathematics.fracdiff（分数阶差分 FFD）   | 保留长记忆的平稳化：0<d<1 介于"价格"与"收益"之间 |
| ``ts_vr(x, w)``            | xw  | physics.hurst（Lo–MacKinlay 方差比）     | 窗口内 log VR(5)：>0 趋势持续，<0 均值回复 |
| ``ts_hurst(x, w)``         | xw  | physics.hurst（DFA）                     | 滚动 Hurst 指数（窗口 ≥32，步长 w//4 前值填充） |
| ``ts_perm_entropy(x, w)``  | xw  | physics.entropy（Bandt–Pompe 排列熵）    | 窗口内三阶序数模式的归一化熵：越低越"可预测" |
| ``ts_kalman_resid(x, q)``  | xc  | control.kalman（局部水平模型稳态增益）   | 观测 − Kalman 滤波水平（q = 过程/观测噪声比） |
| ``ts_levy_area(x, y, w)``  | xxw | mathematics.signatures（二阶签名反对称部分） | 两条路径在窗口内的 Lévy 面积：谁领先谁 |

所有算子对每个资产列独立计算、只用历史窗口，无未来函数。
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .._paths import available

# 名称 -> (arity, 分类, 学科来源, 中文说明)
EXTENDED_OPERATORS: Dict[str, tuple] = {
    "ts_fracdiff": ("xc", "ts", "crossquant.mathematics.fracdiff", "分数阶差分（FFD），d∈(0,1) 保留长记忆的平稳化"),
    "ts_vr": ("xw", "ts", "crossquant.physics.hurst", "滚动 log 方差比 VR(5)：>0 趋势、<0 均值回复"),
    "ts_hurst": ("xw", "ts", "crossquant.physics.hurst", "滚动 DFA Hurst 指数（regime 强度）"),
    "ts_perm_entropy": ("xw", "ts", "crossquant.physics.entropy", "滚动排列熵（序数模式复杂度，越低越可预测）"),
    "ts_kalman_resid": ("xc", "ts", "crossquant.control.kalman", "观测 − 稳态 Kalman 局部水平（q=过程/观测噪声比）"),
    "ts_levy_area": ("xxw", "ts", "crossquant.mathematics.signatures", "两路径窗口内的 Lévy 面积（领先-滞后结构）"),
}
CONST_CHOICES = {"ts_fracdiff": (0.3, 0.5, 0.7), "ts_kalman_resid": (0.01, 0.1, 0.5)}


def _w(w) -> int:
    return int(max(2, round(float(w))))


# ---------------------------------------------------------------------------- 分数阶差分
def _frac_weights(d: float, threshold: float = 1e-3, max_len: int = 120) -> np.ndarray:
    if available("crossquant"):
        from crossquant.mathematics.fracdiff import frac_diff_weights

        w = frac_diff_weights(d, threshold=threshold)
        return np.asarray(w, float)[:max_len]
    w = [1.0]
    k = 1
    while k < max_len:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < threshold:
            break
        w.append(wk)
        k += 1
    return np.asarray(w, float)


def ts_fracdiff(a: pd.DataFrame, c) -> pd.DataFrame:
    d = float(np.clip(float(c), 0.05, 0.95))
    w = _frac_weights(d)[::-1]  # w_K ... w_0
    K = len(w) - 1
    x = a.ffill().to_numpy(float)
    out = np.full(x.shape, np.nan)
    if K < len(x):
        xx = np.nan_to_num(x)
        win = np.lib.stride_tricks.sliding_window_view(xx, K + 1, axis=0)  # (T-K, N, K+1)
        out[K:] = win @ w
    out[~np.isfinite(a.to_numpy(float))] = np.nan
    return pd.DataFrame(out, index=a.index, columns=a.columns)


# ---------------------------------------------------------------------------- 方差比 / Hurst
def ts_vr(a: pd.DataFrame, w, k: int = 5) -> pd.DataFrame:
    """窗口 w 内的 log 方差比 VR(k) = Var(k 期和)/(k·Var(1 期))。"""
    w = max(_w(w), 4 * k)
    rk = a.rolling(k).sum()
    var1 = a.rolling(w, min_periods=w // 2).var()
    vark = rk.rolling(w, min_periods=w // 2).var() / k
    vr = vark / var1.where(var1 > 0)
    return np.log(vr.where(vr > 0))


def ts_hurst(a: pd.DataFrame, w) -> pd.DataFrame:
    from .regime import _dfa_hurst

    w = max(_w(w), 32)
    step = max(w // 4, 5)
    x = a.to_numpy(float)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    for j in range(N):
        col = x[:, j]
        last = np.nan
        for end in range(w, T + 1):
            if (end - w) % step == 0:
                last = _dfa_hurst(col[end - w:end])
            out[end - 1, j] = last
    return pd.DataFrame(out, index=a.index, columns=a.columns)


# ---------------------------------------------------------------------------- 排列熵
def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.full(x.shape, np.nan)
    out[w - 1:] = c[w:] - c[:-w]
    return out


def ts_perm_entropy(a: pd.DataFrame, w) -> pd.DataFrame:
    """三阶 Bandt–Pompe 排列熵（6 种序数模式），窗口 w 内的模式分布熵 / log 6。"""
    w = max(_w(w), 8)
    x = a.to_numpy(float)
    T, N = x.shape
    out = np.full((T, N), np.nan)
    if T < 3:
        return pd.DataFrame(out, index=a.index, columns=a.columns)
    x0, x1, x2 = x[:-2], x[1:-1], x[2:]
    code = (x0 < x1).astype(int) + 2 * (x0 < x2).astype(int) + 4 * (x1 < x2).astype(int)  # (T-2, N) ∈ 0..7
    valid = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(x2)
    ent = np.zeros((T - 2, N))
    total = _rolling_sum(valid.astype(float), w)
    for cde in range(8):
        ind = ((code == cde) & valid).astype(float)
        cnt = _rolling_sum(ind, w)
        p = np.where(total > 0, cnt / np.where(total > 0, total, 1.0), 0.0)
        ent -= np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    ent = ent / math.log(6.0)
    ent[~(total >= w // 2)] = np.nan
    out[2:] = ent
    return pd.DataFrame(out, index=a.index, columns=a.columns)


# ---------------------------------------------------------------------------- Kalman 残差
def kalman_steady_gain(q_ratio: float) -> float:
    """局部水平模型稳态 Kalman 增益：P/R = (q + √(q²+4q))/2，K = (P/R)/(P/R + 1)。"""
    q = max(float(q_ratio), 1e-9)
    pr = 0.5 * (q + math.sqrt(q * q + 4.0 * q))
    return pr / (pr + 1.0)


def ts_kalman_resid(a: pd.DataFrame, c) -> pd.DataFrame:
    K = kalman_steady_gain(float(c))
    level = a.ffill().ewm(alpha=K, adjust=False).mean()
    return (a - level).where(a.notna())


# ---------------------------------------------------------------------------- Lévy 面积
def ts_levy_area(a: pd.DataFrame, b: pd.DataFrame, w, chunk: int = 64) -> pd.DataFrame:
    """窗口内两条路径的归一化 Lévy 面积 A = ½Σ_j (dB_j·ΣdA_{<j} − dA_j·ΣdB_{<j}) / (w·σ_A·σ_B)。

    正值：a 的变化领先于 b（a 先动、b 后动）。
    """
    w = max(_w(w), 4)
    a, b = a.align(b, join="inner")
    da = a.diff().to_numpy(float)[1:]
    db = b.diff().to_numpy(float)[1:]
    T1, N = da.shape
    out = np.full((T1 + 1, N), np.nan)
    if T1 >= w:
        for s in range(0, N, chunk):
            A = np.nan_to_num(da[:, s:s + chunk])
            B = np.nan_to_num(db[:, s:s + chunk])
            Aw = np.lib.stride_tricks.sliding_window_view(A, w, axis=0)  # (T1-w+1, n, w)
            Bw = np.lib.stride_tricks.sliding_window_view(B, w, axis=0)
            cumA = np.cumsum(Aw, axis=-1) - Aw
            cumB = np.cumsum(Bw, axis=-1) - Bw
            area = 0.5 * np.sum(Bw * cumA - Aw * cumB, axis=-1)
            norm = w * Aw.std(axis=-1) * Bw.std(axis=-1)
            res = np.where(norm > 0, area / np.where(norm > 0, norm, 1.0), np.nan)
            out[w:, s:s + chunk] = res
    return pd.DataFrame(out, index=a.index, columns=a.columns)


FUNCS = {
    "ts_fracdiff": ts_fracdiff,
    "ts_vr": ts_vr,
    "ts_hurst": ts_hurst,
    "ts_perm_entropy": ts_perm_entropy,
    "ts_kalman_resid": ts_kalman_resid,
    "ts_levy_area": ts_levy_area,
}

_REGISTERED = False


def register_all(force: bool = False) -> List[str]:
    """把扩展算子注册进 factormine 算子表（幂等），返回算子名列表。"""
    global _REGISTERED
    if not available("factormine"):
        raise ImportError("需要 factormine（factor_mining/）才能注册算子")
    import factormine
    from factormine import gp as _gp

    if _REGISTERED and not force:
        return list(FUNCS)
    for name, (arity, cat, src, doc) in EXTENDED_OPERATORS.items():
        factormine.register(name, FUNCS[name], arity, cat, f"{doc}（{src}）")
    _gp._CONST_CHOICES.update(CONST_CHOICES)
    _REGISTERED = True
    return list(FUNCS)


def describe_operators() -> str:
    return "\n".join(f"- {n}({a}) [{src}] {doc}" for n, (a, _, src, doc) in EXTENDED_OPERATORS.items())
