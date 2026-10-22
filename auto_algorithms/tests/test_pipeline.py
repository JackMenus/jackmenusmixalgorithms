"""生成管线测试：小预算端到端生成到临时目录，检查代码 / 文档 / 索引 / 注册表 / 提议器。"""
import json
import os

import pytest

from conftest import TINY_BUDGET, need

ALL = need("quantstack", "factormine", "crossquant", "order_placement", "latency_routing")


def test_blueprint_roundtrip():
    from autoalgo.blueprint import AlgorithmBlueprint, ComponentRef, EvaluationReport

    bp = AlgorithmBlueprint(algo_id="x_demo", name_zh="演示", protocol="AlphaModel", family="alpha.signature", params={"window": 20},
                           components=[ComponentRef("signature", "crossquant", "crossquant.mathematics.signatures.signature", "特征")])
    bp2 = AlgorithmBlueprint.from_dict(json.loads(bp.to_json()))
    assert bp2.registry_kind == "autoalgo.x_demo" and bp2.components[0].library == "crossquant" and bp2.libraries == ["crossquant"]
    rep = EvaluationReport(algo_id="x_demo", protocol="AlphaModel", metrics={"rank_ic": 0.03}, baselines={"b": {"rank_ic": 0.02}}, objective="rank_ic")
    assert abs(rep.improvement() - 0.01) < 1e-12
    with pytest.raises(ValueError):
        AlgorithmBlueprint(algo_id="bad", name_zh="", protocol="Nope", family="f")


def test_search_space_and_proposers():
    from autoalgo.families import FAMILIES
    from autoalgo.search import EvolutionaryProposer, RandomProposer, clip_to_space, params_key

    spec = FAMILIES["execution.kalman_pid"]
    cands = RandomProposer(seed=1).propose(spec, 5)
    assert len(cands) == 5 and len({params_key(c) for c in cands}) == 5
    for c in cands:
        assert 0.1 <= c["kp"] <= 1.5 and c["controller"] in ("pid", "lqr") and c["mode"] in ("reversion", "momentum")
    hist = [(c, -float(i)) for i, c in enumerate(cands)]
    kids = EvolutionaryProposer(seed=2).propose(spec, 3, hist)
    assert len(kids) == 3 and all(params_key(k) not in {params_key(c) for c in cands} for k in kids)
    clipped = clip_to_space({"kp": 99, "controller": "mpc", "fair_horizon_s": 25, "unknown": 1}, spec)
    assert clipped["kp"] == 1.5 and clipped["controller"] == "pid" and clipped["fair_horizon_s"] == 30.0 and clipped["unknown"] == 1


def test_llm_parser_and_mock_proposer():
    from autoalgo.families import FAMILIES
    from autoalgo.llm import MockProposer, build_prompt, parse_proposals

    spec = FAMILIES["alpha.signature"]
    text = "思考……\n```json\n[{\"params\": {\"window\": 30, \"ridge\": 2.0}, \"rationale\": \"更长窗口\"}, {\"window\": 10}]\n```"
    items = parse_proposals(text)
    assert items[0]["params"]["window"] == 30 and items[1]["params"] == {"window": 10}
    assert "搜索空间" in build_prompt(spec, 2, [({"window": 20}, 0.01)])
    mp = MockProposer(seed=0)
    out = mp.propose(spec, 3)
    assert len(out) == 3 and mp.effective_name == "mock" and all(o["window"] in (10, 15, 20, 30, 40) for o in out)


@ALL
def test_generate_tiny_end_to_end(tmp_path, small_panel):
    import autoalgo
    from autoalgo.pipeline import GenerationConfig, generate
    from autoalgo import registry

    out = str(tmp_path / "generated")
    cfg = GenerationConfig(families=["alpha.signature", "execution.kalman_pid", "router.bayesian_venue"], n_candidates=1, budget=TINY_BUDGET, seed=0, proposer="mock", out_dir=out)
    results = generate(small_panel, cfg, log=lambda s: None)
    assert len(results) == 3
    for r in results:
        d = os.path.join(out, r.blueprint.algo_id)
        assert {"algorithm.py", f"{r.blueprint.algo_id}.md", "blueprint.json", "evaluation.json"} <= set(os.listdir(d))
        md = open(os.path.join(d, f"{r.blueprint.algo_id}.md"), encoding="utf-8").read()
        for sec in ("## 1. 概述", "## 4. 组件来源", "## 6. 参数", "## 7. 评估结果", "## 8. 已知风险", "## 9. 接入方式", "## 10. 复现与血缘"):
            assert sec in md
        assert len(r.candidates) == 2 and r.report.objective
    idx = json.load(open(os.path.join(out, "index.json"), encoding="utf-8"))
    assert {e["algo_id"] for e in idx} == {"signature_alpha", "kalman_pid_execution", "bayesian_venue_router"}
    assert os.path.exists(os.path.join(out, "README.md")) and os.path.exists(os.path.join(out, "last_run.json"))
    # 生成模块可直接 import 并实例化
    mod = registry.load_module("signature_alpha", out)
    m = mod.make()
    assert type(m).__name__ == "SignatureAlpha" and m.window == mod.PARAMS["window"]
    fac = registry.make_execution_factory("autoalgo.kalman_pid_execution", out_dir=out)
    assert "kalman_pid" in fac.available()
    rt = registry.make_router("autoalgo.bayesian_venue_router", venues=["NYSE", "NASDAQ"], out_dir=out)
    assert type(rt).__name__ == "BayesianVenueRouter"
    kinds = registry.available_kinds(out)
    assert "autoalgo.signature_alpha" in kinds["AlphaModel"]
    assert autoalgo.GENERATOR_VERSION == r.blueprint.generator_version


@ALL
def test_registry_family_defaults_without_generation(tmp_path):
    from autoalgo import registry

    empty = str(tmp_path / "none")
    m = registry.make_alpha_model("auto", out_dir=empty)
    assert type(m).__name__ == "RegimeGatedAlpha"
    sel = registry.make_strategy_selector("autoalgo.family.selector.bandit", out_dir=empty, alpha=0.9)
    assert sel.alpha == 0.9
    with pytest.raises(KeyError):
        registry.make("autoalgo.nothing_here", out_dir=empty)
