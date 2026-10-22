"""公式化因子表达式：解析、求值、序列化、结构相似度。

表达式字符串采用函数调用语法，例如::

    cs_rank(neg(ts_corr(close, volume, 10)))
    div(sub(close, ts_mean(close, 20)), ts_std(close, 20))

叶子可以是行情字段（close、volume …）或数值常数。树结构 ``Expr`` 同时被
遗传规划（交叉/变异）、强化学习（RPN 序列）、LLM 生成器（文本解析+校验）复用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .data import Panel
from .operators import OPERATORS

Number = Union[int, float]


@dataclass
class Expr:
    """表达式树节点。op 为 None 时是叶子：field（行情字段）或 value（常数）。"""

    op: Optional[str] = None
    children: List["Expr"] = field(default_factory=list)
    field: Optional[str] = None
    value: Optional[Number] = None

    # ---------- 构造
    @staticmethod
    def feat(name: str) -> "Expr":
        return Expr(field=name)

    @staticmethod
    def const(v: Number) -> "Expr":
        return Expr(value=v)

    @staticmethod
    def call(op: str, *children: "Expr") -> "Expr":
        return Expr(op=op, children=list(children))

    # ---------- 属性
    @property
    def is_leaf(self) -> bool:
        return self.op is None

    def __str__(self) -> str:
        if self.field is not None:
            return self.field
        if self.value is not None:
            v = self.value
            return str(int(v)) if float(v).is_integer() else f"{v:.4g}"
        return f"{self.op}({', '.join(str(c) for c in self.children)})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other) -> bool:
        return isinstance(other, Expr) and str(self) == str(other)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def nodes(self) -> List[Tuple["Expr", Optional["Expr"], int]]:
        """前序遍历，返回 (节点, 父节点, 在父节点中的位置)。"""
        out = [(self, None, -1)]
        for i, c in enumerate(self.children):
            sub = c.nodes()
            sub[0] = (sub[0][0], self, i)
            out.extend(sub)
        return out

    def copy(self) -> "Expr":
        return Expr(self.op, [c.copy() for c in self.children], self.field, self.value)

    def fields(self) -> set:
        if self.field is not None:
            return {self.field}
        return set().union(*(c.fields() for c in self.children)) if self.children else set()

    def subtrees(self) -> set:
        """所有非叶子子树的字符串集合，用于结构相似度/原创性判断（AlphaAgent 风格）。"""
        if self.is_leaf:
            return set()
        s = {str(self)}
        for c in self.children:
            s |= c.subtrees()
        return s


# ------------------------------------------------------------------ 解析
_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|-?\d+\.?\d*(?:[eE][-+]?\d+)?|[(),])")


class ParseError(ValueError):
    pass


def parse(text: str, allowed_fields: Optional[set] = None) -> Expr:
    """把字符串解析为 Expr，并做算子/参数个数/字段合法性检查。"""
    tokens = _TOKEN.findall(text.strip())
    if "".join(tokens).replace(" ", "") != re.sub(r"\s+", "", text.strip()):
        raise ParseError(f"无法识别的字符: {text!r}")
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take(expected=None):
        nonlocal pos
        tok = peek()
        if tok is None or (expected and tok != expected):
            raise ParseError(f"期望 {expected or '记号'}，得到 {tok!r}")
        pos += 1
        return tok

    def node() -> Expr:
        tok = take()
        if re.fullmatch(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", tok):
            return Expr.const(float(tok))
        if peek() == "(":
            if tok not in OPERATORS:
                raise ParseError(f"未知算子 {tok}")
            take("(")
            args = [node()]
            while peek() == ",":
                take(",")
                args.append(node())
            take(")")
            spec = OPERATORS[tok]
            if len(args) != len(spec.arity):
                raise ParseError(f"{tok} 需要 {len(spec.arity)} 个参数，得到 {len(args)}")
            for ch, a in zip(spec.arity, args):
                if ch in "wc" and a.value is None:
                    raise ParseError(f"{tok} 的窗口/常数参数必须是数值")
                if ch == "w" and not (1 <= a.value <= 512):
                    raise ParseError(f"{tok} 窗口 {a.value} 超出 [1,512]")
            return Expr.call(tok, *args)
        if allowed_fields is not None and tok not in allowed_fields:
            raise ParseError(f"未知字段 {tok}")
        return Expr.feat(tok)

    e = node()
    if pos != len(tokens):
        raise ParseError(f"表达式末尾多余内容: {tokens[pos:]}")
    if e.value is not None:
        raise ParseError("表达式不能是纯常数")
    return e


# ------------------------------------------------------------------ 求值
class Evaluator:
    """带缓存的表达式求值器。同一面板上重复子树只算一次（GP/RL 中收益巨大）。"""

    def __init__(self, panel: Panel, max_cache: int = 5000):
        self.panel = panel
        self._ref = panel["close"]
        self.cache: Dict[str, pd.DataFrame] = {}
        self.max_cache = max_cache

    def __call__(self, e: Union[Expr, str]) -> pd.DataFrame:
        if isinstance(e, str):
            e = parse(e)
        return self._eval(e)

    def _eval(self, e: Expr) -> pd.DataFrame:
        key = str(e)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if e.field is not None:
            if e.field not in self.panel:
                raise KeyError(f"面板中没有字段 {e.field}")
            out = self.panel[e.field]
        elif e.value is not None:
            out = pd.DataFrame(float(e.value), index=self._ref.index, columns=self._ref.columns)
        else:
            spec = OPERATORS[e.op]
            args = []
            for ch, c in zip(spec.arity, e.children):
                args.append(c.value if ch in "wc" else self._eval(c))
            with np.errstate(all="ignore"):
                out = spec.func(*args)
            out = out.replace([np.inf, -np.inf], np.nan)
        if len(self.cache) >= self.max_cache:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = out
        return out


def structural_similarity(a: Expr, b: Expr) -> float:
    """基于公共子树的 Jaccard 相似度（0~1），用于惩罚同质化因子、对抗 alpha 衰减。"""
    sa, sb = a.subtrees(), b.subtrees()
    if not sa or not sb:
        return float(str(a) == str(b))
    return len(sa & sb) / len(sa | sb)
