"""LLM 提议器：用 Claude（默认 ``claude-fable-5-1``、思考强度 ``max``）在模板族的搜索空间内提出候选参数与设计理由。

模型策略与 trading_system/docs/MODEL_POLICY.md 一致：算法设计/参数提议属于"强推理阶段"，固定用 Fable 5.1 最高思考强度；
评估、渲染、代码生成是确定性代码，不调用模型。

* ``ClaudeProposer``：需要 ``pip install anthropic`` 与 ``ANTHROPIC_API_KEY``；失败（无网络 / 无凭据 / 解析失败）时自动退回随机提议器，
  并把原因记录在 ``last_error``，血缘中标注 ``proposer = "claude->random"``。
* ``MockProposer``：离线确定性替身，产出与真实模型相同格式的 JSON，用于测试与无网络环境。

输出格式（模型必须返回 JSON 数组）::

    [{"params": {"hurst_window": 90, ...}, "rationale": "……为什么这样改……"}, ...]
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import numpy as np

from .catalog import describe_catalog
from .families import FamilySpec
from .search import History, RandomProposer, clip_to_space, params_key

DEFAULT_MODEL = "claude-fable-5-1"
DEFAULT_EFFORT = "max"

SYSTEM_PROMPT = (
    "你是一名量化交易算法研究员，正在为一个'自主生成算法库'提议新算法的参数配置。"
    "库里的算法由五个既有库的组件拼装而成（因子挖掘 factormine、挂单策略 order_placement、路由延迟 latency_routing、"
    "物理/数学/控制跨学科库 crossquant、全栈集成层 quantstack）。"
    "你的任务：阅读模板族的机制、搜索空间与已评估候选的结果，提出若干个新的、有理由的候选参数组合。"
    "要求：1) 只在给定搜索空间内取值；2) 候选之间要有差异（探索不同假设），不要重复历史；"
    "3) 每个候选给一句中文理由，说明预期为何更好以及可能的失效条件；4) 只输出 JSON 数组，不要多余文字。"
)


def build_prompt(spec: FamilySpec, n: int, history: Optional[History]) -> str:
    hist_lines = []
    for p, s in (history or [])[-20:]:
        hist_lines.append(f"- 得分 {s:+.4g}（越大越好）: {json.dumps(p, ensure_ascii=False, default=str)}")
    space = {k: list(v) for k, v in spec.search_space.items()}
    return "\n".join([
        f"## 模板族 {spec.family}（{spec.name_zh}，协议 {spec.protocol}）",
        f"假设：{spec.hypothesis}",
        "机制：", *[f"{i}. {m}" for i, m in enumerate(spec.mechanism, 1)],
        "参数含义：" + json.dumps(spec.param_doc, ensure_ascii=False),
        "搜索空间（int/float/logfloat 为闭区间，choice 为枚举）：" + json.dumps(space, ensure_ascii=False, default=str),
        f"默认参数：{json.dumps(spec.default_params, ensure_ascii=False, default=str)}",
        f"主目标：{spec.objective}（{'越大越好' if spec.higher_is_better else '越小越好'}）",
        "已评估候选（得分已统一为越大越好）：", *(hist_lines or ["（暂无）"]),
        "可用组件目录（供参考，参数提议不必全部用到）：", describe_catalog(),
        f"请提出 {n} 个新的候选参数组合，输出 JSON 数组：[{{\"params\": {{...}}, \"rationale\": \"...\"}}, ...]",
    ])


def parse_proposals(text: str) -> List[Dict[str, Any]]:
    """从模型输出中提取 JSON 数组（容忍 ```json 代码块与前后文字）。"""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    body = m.group(1) if m else None
    if body is None:
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("输出中没有 JSON 数组")
        body = text[start:end + 1]
    data = json.loads(body)
    if not isinstance(data, list):
        raise ValueError("JSON 顶层不是数组")
    out = []
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("params"), dict):
            out.append({"params": item["params"], "rationale": str(item.get("rationale", ""))})
        elif isinstance(item, dict):
            out.append({"params": item, "rationale": ""})
    return out


class _AnthropicClient:
    """最小 Anthropic 客户端（与 factormine.AnthropicClient 同一调用约定）。"""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, max_tokens: int = 16000, fallback: bool = True):
        import anthropic  # 延迟导入

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens, self.fallback = model, effort, max_tokens, fallback

    def complete(self, system: str, prompt: str) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=[{"role": "user", "content": prompt}], output_config={"effort": self.effort})
        if self.fallback:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"})
            ctx = self.client.beta.messages.stream(**kwargs)
        else:
            ctx = self.client.messages.stream(**kwargs)
        with ctx as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"模型拒绝了请求: {getattr(msg, 'stop_details', None)}")
        return "".join(b.text for b in msg.content if b.type == "text")


class ClaudeProposer:
    name = "claude"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, max_tokens: int = 16000, seed: int = 0, client=None, fallback: bool = True):
        self.model, self.effort, self.max_tokens, self.fallback = model, effort, int(max_tokens), fallback
        self.client = client
        self._random = RandomProposer(seed)
        self.last_raw: str = ""
        self.last_error: str = ""
        self.rationales: Dict[str, str] = {}
        self.used_model: bool = False

    def _client(self):
        if self.client is None:
            self.client = _AnthropicClient(self.model, self.effort, self.max_tokens, self.fallback)
        return self.client

    def propose(self, spec: FamilySpec, n: int, history: Optional[History] = None) -> List[Dict[str, Any]]:
        seen = {params_key(p) for p, _ in (history or [])}
        out: List[Dict[str, Any]] = []
        try:
            self.last_raw = self._client().complete(SYSTEM_PROMPT, build_prompt(spec, n, history))
            for item in parse_proposals(self.last_raw):
                cand = clip_to_space(item["params"], spec)
                key = params_key(cand)
                if key in seen:
                    continue
                seen.add(key)
                self.rationales[key] = item["rationale"]
                out.append(cand)
                if len(out) >= n:
                    break
            self.used_model = True
            self.last_error = ""
        except Exception as e:  # 无网络 / 无凭据 / 解析失败 → 随机兜底
            self.last_error = f"{type(e).__name__}: {e}"
            self.used_model = False
        if len(out) < n:
            out += self._random.propose(spec, n - len(out), (history or []) + [(p, 0.0) for p in out])
        return out

    @property
    def effective_name(self) -> str:
        return self.name if self.used_model else "claude->random"


class MockProposer(ClaudeProposer):
    """离线替身：在搜索空间内确定性地生成候选，并经过与真实模型相同的 JSON 解析路径。"""

    name = "mock"

    class _Client:
        def __init__(self, seed: int):
            self.rng = np.random.default_rng(seed)

        def complete(self, system: str, prompt: str) -> str:
            from .search import sample_dim

            m = re.search(r"请提出 (\d+) 个", prompt)
            n = int(m.group(1)) if m else 3
            space_m = re.search(r"搜索空间（[^）]*）：(\{.*?\})\n默认参数", prompt, re.S)
            space = json.loads(space_m.group(1)) if space_m else {}
            items = []
            for _ in range(n):
                params = {k: sample_dim(tuple(v), self.rng) for k, v in space.items()}
                items.append({"params": params, "rationale": "离线替身：在搜索空间内随机探索，不代表真实模型的判断。"})
            return "```json\n" + json.dumps(items, ensure_ascii=False, default=str) + "\n```"

    def __init__(self, seed: int = 0, **kw):
        super().__init__(seed=seed, client=self._Client(seed))

    @property
    def effective_name(self) -> str:
        return "mock"
