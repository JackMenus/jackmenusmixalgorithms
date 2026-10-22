"""LLM 驱动的因子挖掘智能体。

融合的前沿思路：
  * Alpha-GPT（Wang et al., 2023）：人机交互式“想法 → 公式 → 回测 → 反馈”循环；
  * AlphaAgent（Tang et al., KDD 2025）：三角色（Idea / Factor / Eval）+ 原创性（与已有因子的
    AST 相似度）、假设-公式一致性、复杂度三项正则，主动对抗 alpha 衰减；
  * Alpha Jungle（Shi et al., AAAI 2026）：LLM + MCTS，把回测反馈作为树搜索的价值，
    在高潜力表达式附近做“精修”；
  * FactorMiner（2026, arXiv:2602.14670）：自我进化的智能体，带“技能 + 经验记忆”，
    把成功/失败模式沉淀下来指导下一轮生成；
  * R&D-Agent-Quant（Microsoft, 2025）：研究与开发两阶段，因子与模型联合迭代。

本模块不绑定任何 LLM 厂商：实现 ``LLMClient.complete(system, prompt) -> str`` 即可接入。
自带 ``AnthropicClient``（可选依赖 ``anthropic``）与离线测试用的 ``MockLLM``。
按项目约定：强推理阶段（提出假设、精修）建议用 ``claude-fable-5-1`` + effort="max"，
批量生成/改写可用 ``claude-opus-5-5`` + 自适应思考。
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .evaluate import evaluate_factor
from .expression import Expr, ParseError, parse, structural_similarity
from .library import FactorPool, FactorRecord
from .operators import describe_operators


# ------------------------------------------------------------------ LLM 客户端
class LLMClient(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class AnthropicClient:
    """Anthropic Claude 客户端（需要 ``pip install anthropic`` 与 API 凭据）。

    * Fable 5.1：思考始终开启，只用 ``output_config.effort`` 控制深度（low…max）；
    * Opus 5.5：自适应思考，effort 默认 medium，建议显式设置。
    默认开启服务端 refusal 回退（fallbacks="default"）。
    """

    def __init__(self, model: str = "claude-fable-5-1", effort: str = "max", max_tokens: int = 64000,
                 fallback: bool = True):
        import anthropic  # 延迟导入，保持核心库零依赖

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens, self.fallback = model, effort, max_tokens, fallback

    def complete(self, system: str, prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort},
        )
        if self.fallback:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"})
            stream_ctx = self.client.beta.messages.stream(**kwargs)
        else:
            stream_ctx = self.client.messages.stream(**kwargs)
        with stream_ctx as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"模型拒绝了请求: {getattr(msg, 'stop_details', None)}")
        return "".join(b.text for b in msg.content if b.type == "text")


class MockLLM:
    """离线模拟 LLM：从模板库随机组合出候选，用于测试与无网络环境。"""

    TEMPLATES = [
        ("短期反转：过去 {w} 日涨幅大的股票倾向回落", "neg(cs_rank(ts_pct(close, {w})))"),
        ("放量下跌：成交量相对均量放大往往伴随后续弱势", "neg(cs_rank(div(volume, ts_mean(volume, {w}))))"),
        ("量价相关：价量同向说明趋势由资金推动", "neg(ts_corr(cs_rank(close), cs_rank(volume), {w}))"),
        ("波动率异象：低波动股票长期收益更高", "neg(ts_std(returns, {w}))"),
        ("日内位置：收盘接近最高价表示买盘强", "div(sub(close, low), add(sub(high, low), 0.001))"),
        ("VWAP 偏离：收盘高于 VWAP 预示短期回吐", "neg(cs_rank(div(sub(close, vwap), vwap)))"),
        ("时序极值位置：近期创新高后的动量衰减", "neg(ts_rank(close, {w}))"),
    ]

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def complete(self, system: str, prompt: str) -> str:
        n = int(m.group(1)) if (m := re.search(r"生成\s*(\d+)\s*个", prompt)) else 3
        out = []
        for _ in range(n):
            h, t = self.rng.choice(self.TEMPLATES)
            out.append({"hypothesis": h, "expression": t.format(w=self.rng.choice([5, 10, 20]))})
        return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"


# ------------------------------------------------------------------ 提示词
SYSTEM_PROMPT = """你是顶级量化研究员，负责挖掘可交易的横截面 alpha 因子。
要求：
1. 每个因子先给出清晰的经济学/行为金融假设，再给出严格符合语法的公式；
2. 只能使用给定的字段与算子，窗口参数为 1~252 的整数；
3. 追求原创：避免与“已有因子”结构雷同；控制复杂度（节点数 ≤ 20）；
4. 只输出一个 JSON 数组（可放在 ```json 代码块中），元素为 {"hypothesis": str, "expression": str}。"""


def build_prompt(fields: Sequence[str], n: int, existing: Sequence[str], feedback: str, theme: str = "") -> str:
    return f"""可用字段：{', '.join(fields)}
可用算子：
{describe_operators()}

研究方向：{theme or '不限，鼓励多样化（量价、波动、流动性、日内结构等）'}

已有因子（请避免结构雷同）：
{chr(10).join('- ' + e for e in existing[-30:]) or '（暂无）'}

上一轮回测反馈：
{feedback or '（首轮）'}

请生成 {n} 个新因子。"""


def parse_candidates(text: str) -> List[Dict[str, str]]:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = m.group(1) if m else text
    m2 = re.search(r"\[.*\]", body, re.S)
    if not m2:
        return []
    try:
        data = json.loads(m2.group(0))
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict) and "expression" in d]


# ------------------------------------------------------------------ 智能体
@dataclass
class LLMConfig:
    rounds: int = 5
    per_round: int = 8
    max_size: int = 20
    max_similarity: float = 0.6  # 与已有因子的 AST 相似度上限（原创性约束）
    theme: str = ""
    fields: Sequence[str] = ("open", "high", "low", "close", "volume", "vwap", "returns")


@dataclass
class Candidate:
    hypothesis: str
    expression: str
    status: str = ""
    rank_ic: float = float("nan")
    accepted: bool = False


@dataclass
class LLMAlphaAgent:
    """想法 → 公式 → 校验 → 回测 → 反馈 的闭环智能体。"""

    pool: FactorPool
    llm: LLMClient
    config: LLMConfig = field(default_factory=LLMConfig)
    log: Callable[[str], None] = print
    memory: List[Candidate] = field(default_factory=list)  # 经验记忆（成功与失败都保留）

    def _validate(self, c: Candidate) -> Optional[Expr]:
        try:
            e = parse(c.expression, allowed_fields=set(self.config.fields))
        except ParseError as ex:
            c.status = f"语法错误: {ex}"
            return None
        if e.size() > self.config.max_size:
            c.status = f"过于复杂（{e.size()} 节点）"
            return None
        for old in self.pool.exprs:
            if structural_similarity(e, old) > self.config.max_similarity:
                c.status = f"与已有因子 {old} 结构雷同"
                return None
        return e

    def _feedback(self, batch: List[Candidate]) -> str:
        lines = []
        for c in batch:
            ic = "" if np.isnan(c.rank_ic) else f" RankIC={c.rank_ic:+.4f}"
            lines.append(f"- {c.expression}: {'入池' if c.accepted else '未入池'}{ic} {c.status}")
        good = sorted((c for c in self.memory if c.accepted), key=lambda c: -abs(c.rank_ic))[:5]
        if good:
            lines.append("历史上效果最好的思路：" + "；".join(c.hypothesis for c in good))
        return "\n".join(lines)

    def run(self) -> FactorPool:
        feedback = ""
        for r in range(self.config.rounds):
            prompt = build_prompt(self.config.fields, self.config.per_round,
                                  [str(e) for e in self.pool.exprs], feedback, self.config.theme)
            raw = self.llm.complete(SYSTEM_PROMPT, prompt)
            batch = [Candidate(d.get("hypothesis", ""), d["expression"]) for d in parse_candidates(raw)]
            for c in batch:
                e = self._validate(c)
                if e is None:
                    continue
                ic, v = self.pool.score(e)
                c.rank_ic = ic
                c.accepted = self.pool.try_add(e, source="llm")
                if not c.accepted and not c.status:
                    c.status = "IC 不足或与池内因子高度相关"
            self.memory.extend(batch)
            feedback = self._feedback(batch)
            n_ok = sum(c.accepted for c in batch)
            self.log(f"[LLM] 第 {r} 轮  候选 {len(batch)}  入池 {n_ok}  池大小={len(self.pool)}")
        return self.pool

    def records(self) -> List[FactorRecord]:
        hyp = {c.expression: c.hypothesis for c in self.memory}
        recs = []
        for i, (e, ic, s) in enumerate(zip(self.pool.exprs, self.pool.ics, self.pool.sources)):
            recs.append(FactorRecord(f"llm_{i:03d}", str(e), s, {"rank_ic": ic}, rationale=hyp.get(str(e), "")))
        return recs
