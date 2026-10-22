"""因子算子库（Alpha101 / 国泰君安 191 / Qlib 风格）。

所有算子作用在 ``date x asset`` 的 DataFrame 上：
  * ts_*  —— 时间序列算子，沿时间轴对每个资产独立滚动计算（只用历史数据，无未来函数）；
  * cs_*  —— 横截面算子，在每个时间点对所有资产计算；
  * 其余为逐元素算子。

算子通过 ``OPERATORS`` 注册表暴露给表达式引擎、遗传规划、强化学习与 LLM 生成器。
签名约定（``arity``）里的字符含义：``x`` = 因子输入，``w`` = 整数窗口，``c`` = 浮点常数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd

DF = pd.DataFrame
EPS = 1e-12


# ---------------------------------------------------------------- 逐元素
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b
def div(a, b): return a / _safe_den(b)
def neg(a): return -a
def abs_(a): return a.abs()
def log(a): return np.log(a.abs() + 1.0) * np.sign(a)
def sign(a): return np.sign(a)
def sqrt(a): return np.sqrt(a.abs()) * np.sign(a)
def square(a): return a * a
def inv(a): return 1.0 / _safe_den(a)
def max_(a, b): return np.maximum(a, b)
def min_(a, b): return np.minimum(a, b)
def greater(a, b): return (a > b).astype(float).where(a.notna() & b.notna())
def less(a, b): return (a < b).astype(float).where(a.notna() & b.notna())
def signed_power(a, c): return np.sign(a) * a.abs() ** float(c)
def add_c(a, c): return a + float(c)
def mul_c(a, c): return a * float(c)


def _safe_den(b):
    b = b if isinstance(b, (pd.DataFrame, pd.Series)) else pd.DataFrame(b)
    return b.where(b.abs() > EPS, np.nan)


# ---------------------------------------------------------------- 时间序列
def _w(w) -> int:
    return max(int(w), 1)


def delay(a, w): return a.shift(_w(w))
def delta(a, w): return a - a.shift(_w(w))
def ts_pct(a, w): return a / _safe_den(a.shift(_w(w))) - 1.0
def ts_mean(a, w): return a.rolling(_w(w), min_periods=max(2, _w(w) // 2)).mean()
def ts_sum(a, w): return a.rolling(_w(w), min_periods=max(2, _w(w) // 2)).sum()
def ts_std(a, w): return a.rolling(_w(w), min_periods=max(2, _w(w) // 2)).std()
def ts_max(a, w): return a.rolling(_w(w), min_periods=1).max()
def ts_min(a, w): return a.rolling(_w(w), min_periods=1).min()
def ts_median(a, w): return a.rolling(_w(w), min_periods=max(2, _w(w) // 2)).median()
def ts_skew(a, w): return a.rolling(max(_w(w), 3), min_periods=3).skew()
def ts_kurt(a, w): return a.rolling(max(_w(w), 4), min_periods=4).kurt()
def ts_zscore(a, w): return (a - ts_mean(a, w)) / _safe_den(ts_std(a, w))


def ts_argmax(a, w):
    return a.rolling(_w(w), min_periods=1).apply(np.argmax, raw=True)


def ts_argmin(a, w):
    return a.rolling(_w(w), min_periods=1).apply(np.argmin, raw=True)


def ts_rank(a, w):
    """当前值在过去 w 期中的分位（0~1），用滑动窗口向量化实现。"""
    w = _w(w)
    x = a.to_numpy(dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) >= w:
        win = np.lib.stride_tricks.sliding_window_view(x, w, axis=0)  # (T-w+1, N, w)
        last = win[..., -1:]
        valid = ~np.isnan(win)
        less_cnt = ((win < last) & valid).sum(-1)
        eq_cnt = ((win == last) & valid).sum(-1)
        n = valid.sum(-1)
        r = (less_cnt + 0.5 * (eq_cnt - 1)) / np.maximum(n - 1, 1)
        r[np.isnan(last[..., 0]) | (n < 2)] = np.nan
        out[w - 1 :] = r
    return pd.DataFrame(out, index=a.index, columns=a.columns)


def ts_decay_linear(a, w):
    """线性衰减加权均值（近期权重大），Alpha101 中的 decay_linear。"""
    w = _w(w)
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()
    return a.rolling(w, min_periods=w).apply(lambda v: np.dot(v, weights), raw=True)


def ts_ema(a, w): return a.ewm(span=_w(w), min_periods=max(2, _w(w) // 2)).mean()


def ts_corr(a, b, w):
    w = max(_w(w), 3)
    return a.rolling(w, min_periods=w).corr(b).replace([np.inf, -np.inf], np.nan)


def ts_cov(a, b, w):
    w = max(_w(w), 3)
    return a.rolling(w, min_periods=w).cov(b)


# ---------------------------------------------------------------- 横截面
def cs_rank(a): return a.rank(axis=1, pct=True)


def cs_zscore(a):
    mu = a.mean(axis=1)
    sd = a.std(axis=1).where(lambda s: s > EPS)
    return a.sub(mu, axis=0).div(sd, axis=0)


def cs_demean(a): return a.sub(a.mean(axis=1), axis=0)


def cs_scale(a):
    """缩放使每期绝对值之和为 1（Alpha101 中的 scale）。"""
    return a.div(a.abs().sum(axis=1).where(lambda s: s > EPS), axis=0)


def cs_winsorize(a, c=3.0):
    """按横截面 MAD 去极值。"""
    med = a.median(axis=1)
    mad = a.sub(med, axis=0).abs().median(axis=1) * 1.4826
    lo, hi = med - float(c) * mad, med + float(c) * mad
    return a.clip(lower=lo, upper=hi, axis=0)


def cs_neutralize(a, b):
    """对 b 做横截面一元回归，取残差（例如对市值/行业暴露中性化）。"""
    a_d, b_d = cs_demean(a), cs_demean(b)
    beta = (a_d * b_d).sum(axis=1) / (b_d * b_d).sum(axis=1).where(lambda s: s > EPS)
    return a_d - b_d.mul(beta, axis=0)


# ---------------------------------------------------------------- 注册表
@dataclass(frozen=True)
class OpSpec:
    name: str
    func: Callable
    arity: str  # 例如 "xw" 表示 (因子, 窗口)
    category: str  # elementwise / ts / cs
    doc: str = ""

    @property
    def n_inputs(self) -> int:
        return self.arity.count("x")


OPERATORS: Dict[str, OpSpec] = {}


def register(name: str, func: Callable, arity: str, category: str, doc: str = "") -> None:
    """注册新算子；外部扩展（如物理/控制方向的滤波器）也走这个入口。"""
    OPERATORS[name] = OpSpec(name, func, arity, category, doc)


for _n, _f, _a, _d in [
    ("add", add, "xx", "a+b"), ("sub", sub, "xx", "a-b"), ("mul", mul, "xx", "a*b"),
    ("div", div, "xx", "a/b（分母近 0 置空）"), ("max", max_, "xx", "逐元素最大"),
    ("min", min_, "xx", "逐元素最小"), ("greater", greater, "xx", "a>b 为 1"),
    ("less", less, "xx", "a<b 为 1"), ("neg", neg, "x", "-a"), ("abs", abs_, "x", "|a|"),
    ("log", log, "x", "sign(a)*log(1+|a|)"), ("sign", sign, "x", "符号"),
    ("sqrt", sqrt, "x", "保号平方根"), ("square", square, "x", "平方"),
    ("inv", inv, "x", "倒数"), ("signed_power", signed_power, "xc", "保号幂"),
    ("add_c", add_c, "xc", "加常数"), ("mul_c", mul_c, "xc", "乘常数"),
]:
    register(_n, _f, _a, "elementwise", _d)

for _n, _f, _a, _d in [
    ("delay", delay, "xw", "滞后 w 期"), ("delta", delta, "xw", "a - delay(a,w)"),
    ("ts_pct", ts_pct, "xw", "w 期变化率"), ("ts_mean", ts_mean, "xw", "滚动均值"),
    ("ts_sum", ts_sum, "xw", "滚动求和"), ("ts_std", ts_std, "xw", "滚动标准差"),
    ("ts_max", ts_max, "xw", "滚动最大"), ("ts_min", ts_min, "xw", "滚动最小"),
    ("ts_median", ts_median, "xw", "滚动中位数"), ("ts_skew", ts_skew, "xw", "滚动偏度"),
    ("ts_kurt", ts_kurt, "xw", "滚动峰度"), ("ts_zscore", ts_zscore, "xw", "时序标准化"),
    ("ts_rank", ts_rank, "xw", "时序分位"), ("ts_argmax", ts_argmax, "xw", "最大值位置"),
    ("ts_argmin", ts_argmin, "xw", "最小值位置"),
    ("ts_decay_linear", ts_decay_linear, "xw", "线性衰减加权均值"),
    ("ts_ema", ts_ema, "xw", "指数均线"), ("ts_corr", ts_corr, "xxw", "滚动相关"),
    ("ts_cov", ts_cov, "xxw", "滚动协方差"),
]:
    register(_n, _f, _a, "ts", _d)

for _n, _f, _a, _d in [
    ("cs_rank", cs_rank, "x", "横截面分位"), ("cs_zscore", cs_zscore, "x", "横截面标准化"),
    ("cs_demean", cs_demean, "x", "横截面去均值"), ("cs_scale", cs_scale, "x", "绝对值和归一"),
    ("cs_winsorize", cs_winsorize, "xc", "MAD 去极值"),
    ("cs_neutralize", cs_neutralize, "xx", "对 b 回归取残差"),
]:
    register(_n, _f, _a, "cs", _d)

# 默认的搜索窗口集合（日频）；分钟/tick 级可自行替换。
DEFAULT_WINDOWS = (3, 5, 10, 20, 40, 60)


def describe_operators() -> str:
    """生成算子说明文本，供 LLM 提示词与文档使用。"""
    lines = []
    for spec in OPERATORS.values():
        args = ", ".join({"x": "x", "w": "window:int", "c": "const:float"}[ch] for ch in spec.arity)
        lines.append(f"- {spec.name}({args})  [{spec.category}] {spec.doc}")
    return "\n".join(lines)
