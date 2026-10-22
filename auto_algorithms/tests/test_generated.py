"""对仓库内 generated/ 目录（首批生成算法）的回归测试：索引完整、每个算法可实例化且满足协议。"""
import json
import os

import pytest

from conftest import ROOT, need

GEN = os.path.join(ROOT, "generated")
INDEX = os.path.join(GEN, "index.json")
ENTRIES = json.load(open(INDEX, encoding="utf-8")) if os.path.exists(INDEX) else []

ALL = need("quantstack", "factormine", "crossquant", "order_placement", "latency_routing")


@pytest.mark.skipif(not ENTRIES, reason="generated/ 为空（尚未运行生成管线）")
def test_index_consistent_with_files():
    for e in ENTRIES:
        d = os.path.join(GEN, e["algo_id"])
        assert os.path.exists(os.path.join(d, "algorithm.py")) and os.path.exists(os.path.join(d, f"{e['algo_id']}.md"))
        bp = json.load(open(os.path.join(d, "blueprint.json"), encoding="utf-8"))
        ev = json.load(open(os.path.join(d, "evaluation.json"), encoding="utf-8"))
        assert bp["algo_id"] == e["algo_id"] and ev["objective"] == e["objective"] and bp["lineage"]["candidates"]
    assert os.path.exists(os.path.join(GEN, "README.md"))


@ALL
@pytest.mark.skipif(not ENTRIES, reason="generated/ 为空（尚未运行生成管线）")
@pytest.mark.parametrize("entry", ENTRIES, ids=[e["algo_id"] for e in ENTRIES])
def test_generated_algorithm_instantiates(entry):
    from quantstack.contracts import AlphaModel, ExecutionStrategyFactory, PortfolioConstructor, Router

    from autoalgo import registry

    kind = f"autoalgo.{entry['algo_id']}"
    proto = entry["protocol"]
    if proto == "AlphaModel":
        assert isinstance(registry.make_alpha_model(kind), AlphaModel)
    elif proto == "PortfolioConstructor":
        assert isinstance(registry.make_portfolio(kind), PortfolioConstructor)
    elif proto == "ExecutionStrategy":
        f = registry.make_execution_factory(kind)
        assert isinstance(f, ExecutionStrategyFactory) and "kalman_pid" in f.available()
    elif proto == "StrategySelector":
        assert callable(registry.make_strategy_selector(kind))
    elif proto == "Router":
        assert isinstance(registry.make_router(kind, venues=["NYSE", "NASDAQ", "ARCA", "IEX"]), Router)
