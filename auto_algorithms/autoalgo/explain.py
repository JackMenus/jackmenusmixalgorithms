"""可解释性文档生成：把蓝图 + 评估报告渲染为中文 Markdown。

固定章节（便于人工审阅与 agent 解析）：
  1 概述  2 所处协议层与接口  3 设计动机与假设  4 组件来源（来自哪些库）  5 算法流程与关键公式
  6 参数（取值 / 含义 / 搜索范围）  7 评估结果（本算法 vs 基线，是否通过）  8 已知风险与失效条件
  9 接入方式（quantstack 注册名 / 代码片段）  10 复现与血缘（种子 / 提议器 / 候选数 / 生成器版本）
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .blueprint import AlgorithmBlueprint, EvaluationReport

PROTOCOL_DOC = {
    "AlphaModel": ("因子层", "`fit(panel)` / `predict(panel, as_of) -> SignalBatch` / `describe()`", "输入 `Panel = {字段: DataFrame(时间×资产)}`，输出每个资产的分数、期望收益（bps）与置信度"),
    "FactorMiner": ("因子层", "`mine(panel, budget) -> [FactorSpec]` / `evaluate(spec, panel) -> FactorReport`", "候选因子表达式及其样本外评估"),
    "PortfolioConstructor": ("组合层", "`construct(signals, portfolio, market, limits) -> [TargetPosition]`", "把信号变成目标持仓（数量、权重、紧急度、参与率）"),
    "ExecutionStrategy": ("执行层", "`start / on_market / on_fill / on_event / cancels / is_complete / state`；经 `ExecutionStrategyFactory.create(name, order, **params)` 创建", "把母单切成子单切片（IOC / POST_ONLY …）"),
    "StrategySelector": ("执行规划", "`(order, snapshot, available) -> (strategy_name, params)`，注入 `TradingPipeline.strategy_selector`", "为每张母单选择执行策略与参数"),
    "Router": ("路由层", "`route(slice, snapshot) -> RoutingDecision` / `on_event(event)` / `venue_stats()`", "把子单切片拆到各交易场所"),
    "StateEstimator": ("跨学科钩子", "`update(obs, ts_us) -> float` / `predict(horizon_us) -> float`", "隐状态滤波与外推"),
    "Controller": ("跨学科钩子", "`step(error, dt_us) -> float`", "跟踪误差 → 控制量"),
    "ImpactModel": ("跨学科钩子", "`expected_cost_bps(quantity, snapshot, duration_us) -> float`", "期望交易成本"),
}

METRIC_DOC = {
    "rank_ic": "样本外 RankIC 均值（越大越好）", "rank_icir": "RankIC 均值/标准差（年化）", "ic_t": "RankIC 的 t 统计量", "turnover": "日均换手（分数变化幅度）",
    "coverage": "有效信号覆盖率", "n_periods": "样本外期数", "ls_sharpe": "五分组多空年化夏普",
    "sharpe": "组合年化夏普（扣成本）", "ann_return": "年化收益", "ann_vol": "年化波动", "max_drawdown": "最大回撤", "avg_turnover": "每次再平衡的单边换手",
    "avg_hhi": "权重集中度 HHI 均值", "max_weight": "最大单票权重", "n_rebalances": "每个面板的再平衡次数", "sharpe_per_panel": "逐面板夏普", "n_panels": "评估面板数",
    "shortfall_bps": "实现偏差均值（bps，越小越好）", "shortfall_median_bps": "实现偏差中位数", "shortfall_std_bps": "实现偏差标准差", "fill_ratio": "成交率",
    "maker_ratio": "被动成交比例", "n_orders": "母单数", "n_seeds": "随机种子数", "wins_vs_baseline": "逐种子 / 逐面板对比胜率",
    "learned_rounds_shortfall_bps": "学习期（后半程）实现偏差", "n_rebalanced": "被重新分配的切片数", "n_rewards": "收到的奖励样本数",
}


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        if v != v:
            return "NaN"
        return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.3e}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v)


def _space(dim) -> str:
    if not dim:
        return "—"
    k = dim[0]
    if k == "choice":
        return "{" + ", ".join(_fmt(x) for x in dim[1]) + "}"
    if k in ("int", "float", "logfloat"):
        return f"[{_fmt(dim[1])}, {_fmt(dim[2])}]" + ("（对数均匀）" if k == "logfloat" else "")
    return str(dim)


def render_markdown(bp: AlgorithmBlueprint, rep: Optional[EvaluationReport] = None, extra: Optional[Dict[str, Any]] = None) -> str:
    extra = extra or {}
    layer, iface, io = PROTOCOL_DOC.get(bp.protocol, ("", "", ""))
    L: List[str] = []
    L.append(f"# {bp.name_zh}（`{bp.algo_id}`）")
    L.append("")
    L.append(f"> 自主生成算法库（autoalgo）生成 · 协议 `{bp.protocol}`（{layer}） · 模板族 `{bp.family}` · 生成时间 {bp.created_at} · 生成器 v{bp.generator_version}")
    L.append("")
    # 1 概述
    L.append("## 1. 概述")
    L.append("")
    verdict = "**评估通过**" if (rep and rep.passed) else ("**评估未通过**（保留以供分析，默认不建议上线）" if rep else "未评估")
    L.append(f"{bp.hypothesis}")
    L.append("")
    L.append(f"结论：{verdict}。" + (f" {rep.verdict}" if rep and rep.verdict else ""))
    L.append("")
    # 2 协议层
    L.append("## 2. 所处协议层与接口")
    L.append("")
    L.append(f"- 协议：`quantstack.contracts.{bp.protocol}`" if bp.protocol != "StrategySelector" else "- 协议：`quantstack.core.pipeline.StrategySelector`（可调用对象）")
    L.append(f"- 接口：{iface}")
    L.append(f"- 输入 / 输出：{io}")
    L.append(f"- quantstack 注册名：`{bp.registry_kind}`")
    L.append("")
    # 3 假设
    L.append("## 3. 设计动机与假设")
    L.append("")
    for a in bp.assumptions:
        L.append(f"- {a}")
    L.append("")
    # 4 组件来源
    L.append("## 4. 组件来源（建立在哪些库之上）")
    L.append("")
    L.append("| 库 | 组件 | 在本算法中的作用 |")
    L.append("|---|---|---|")
    for c in bp.components:
        L.append(f"| `{c.library}` | `{c.path}` | {c.role} |")
    L.append("")
    L.append(f"涉及库：{', '.join('`' + x + '`' for x in bp.libraries)}。")
    L.append("")
    # 5 流程
    L.append("## 5. 算法流程与关键公式")
    L.append("")
    for i, m in enumerate(bp.mechanism, 1):
        L.append(f"{i}. {m}")
    formulas = extra.get("formula") or []
    if formulas:
        L.append("")
        L.append("关键公式：")
        L.append("")
        for f in formulas:
            L.append(f"- `{f}`")
    L.append("")
    # 6 参数
    L.append("## 6. 参数")
    L.append("")
    L.append("| 参数 | 取值 | 含义 | 搜索范围 |")
    L.append("|---|---|---|---|")
    for k, v in bp.params.items():
        L.append(f"| `{k}` | {_fmt(v)} | {bp.param_doc.get(k, '')} | {_space(bp.search_space.get(k))} |")
    L.append("")
    # 7 评估
    L.append("## 7. 评估结果")
    L.append("")
    if rep is None:
        L.append("（未评估）")
    else:
        L.append(f"- 主目标：`{rep.objective}`（{'越大越好' if rep.higher_is_better else '越小越好'}）；评估设置：{_fmt(rep.settings)}")
        imp = rep.improvement()
        if imp is not None:
            L.append(f"- 相对第一基线的改善：{imp:+.4g}（已按方向调整，正数 = 更好）")
        L.append("")
        cols = ["本算法"] + list(rep.baselines)
        keys = list(rep.metrics)
        for b in rep.baselines.values():
            for k in b:
                if k not in keys:
                    keys.append(k)
        L.append("| 指标 | 说明 | " + " | ".join(cols) + " |")
        L.append("|---|---|" + "---|" * len(cols))
        for k in keys:
            row = [_fmt(rep.metrics.get(k, "—"))] + [_fmt(b.get(k, "—")) for b in rep.baselines.values()]
            L.append(f"| `{k}` | {METRIC_DOC.get(k, '')} | " + " | ".join(row) + " |")
        L.append("")
        L.append(f"- 判定：{'通过' if rep.passed else '未通过'}。{rep.verdict}")
        for n in rep.notes:
            L.append(f"- {n}")
        L.append(f"- 评估耗时：{rep.wall_s:.1f} 秒")
    L.append("")
    # 8 风险
    L.append("## 8. 已知风险与失效条件")
    L.append("")
    for r in bp.risks:
        L.append(f"- {r}")
    L.append("")
    # 9 接入
    L.append("## 9. 接入方式")
    L.append("")
    L.append("```python")
    L.append(extra.get("usage_code", f"from autoalgo.registry import make\nobj = make({bp.algo_id!r})"))
    L.append("```")
    L.append("")
    integ = extra.get("integration_note")
    if integ:
        L.append(integ)
        L.append("")
    # 10 复现
    L.append("## 10. 复现与血缘")
    L.append("")
    lineage = dict(bp.lineage)
    cands = lineage.pop("candidates", None) or []
    label = {"generator": "生成器", "generator_version": "生成器版本", "proposer": "提议器", "model": "提议模型", "effort": "思考强度", "llm_error": "LLM 回退原因",
             "seed": "随机种子", "budget": "评估预算", "train_ratio": "训练集比例", "n_candidates": "评估候选数", "panel": "评估面板", "selected_score": "选中候选得分"}
    for k, v in lineage.items():
        if v in ("", None, [], {}):
            continue
        L.append(f"- {label.get(k, k)}：{_fmt(v)}")
    if cands:
        defaults = _family_defaults(bp.family)
        has_rationale = any(c.get("rationale") for c in cands)
        L.append("")
        L.append(f"候选评估记录（共 {len(cands)} 个，按评估顺序；★ 为最终选中的参数）：")
        L.append("")
        L.append("| # | 来源 | 与默认参数的差异 | 目标值 | 通过 | 耗时(s) |" + (" 提议理由 |" if has_rationale else ""))
        L.append("|---|---|---|---|---|---|" + ("---|" if has_rationale else ""))
        for i, c in enumerate(cands, 1):
            cp = c.get("params", {}) or {}
            diff = {k: v for k, v in cp.items() if defaults.get(k, object()) != v}
            diff_s = "（默认）" if not diff else ", ".join(f"`{k}`={_fmt(v)}" for k, v in diff.items())
            star = "★ " if cp == bp.params else ""
            row = f"| {i} | {c.get('source', '')} | {star}{diff_s} | {_fmt(c.get('value'))} | {'是' if c.get('passed') else '否'} | {_fmt(c.get('wall_s'))} |"
            if has_rationale:
                row += f" {str(c.get('rationale', '')).replace('|', '/')} |"
            L.append(row)
    L.append("")
    L.append(f"- 重新生成：`{extra.get('repro_cmd', 'python -m autoalgo generate --families ' + bp.family)}`")
    L.append(f"- 只重排文档（不重新评估）：`python -m autoalgo rerender`")
    L.append(f"- 产物：`generated/{bp.algo_id}/algorithm.py`、`{bp.algo_id}.md`、`blueprint.json`、`evaluation.json`")
    L.append("")
    return "\n".join(L)


def _family_defaults(family: str) -> Dict[str, Any]:
    try:
        from .families import FAMILIES

        return dict(FAMILIES[family].default_params)
    except Exception:  # 模板族已被移除时仍能渲染旧文档
        return {}


def render_index_markdown(entries: List[Dict[str, Any]]) -> str:
    L = ["# 自主生成算法索引", "", "由 `autoalgo` 生成管线自动维护。每个算法目录含 `algorithm.py`（可直接 import）、中文说明 `.md`、`blueprint.json` 与 `evaluation.json`。", "",
         "| 算法 | 协议 | 模板族 | 主目标 | 本算法 | 基线 | 通过 | 生成时间 |", "|---|---|---|---|---|---|---|---|"]
    for e in entries:
        L.append(f"| [`{e['algo_id']}`]({e['algo_id']}/{e['algo_id']}.md) | {e['protocol']} | `{e['family']}` | `{e.get('objective', '')}` | {_fmt(e.get('primary'))} | {_fmt(e.get('baseline'))} | {'✓' if e.get('passed') else '✗'} | {e.get('created_at', '')} |")
    L.append("")
    return "\n".join(L)
