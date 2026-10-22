"""quantstack 契约适配测试；若集成层代码不在共享目录则跳过。"""
import os
import sys

import numpy as np
import pytest

TS = os.environ.get("QUANT_PROJECT_ROOT",
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
TS = os.path.join(TS, "trading_system")
if not os.path.isdir(os.path.join(TS, "quantstack", "contracts")):
    pytest.skip("quantstack 集成层不可用", allow_module_level=True)
sys.path.insert(0, TS)

from quantstack.contracts import (ExecutionOrder, ExecutionStrategy, ExecutionStrategyFactory, Fill, ImpactModel,
                                  MarketSnapshot, OrderEvent, VenueQuote)
from order_placement import make_execution_factory, make_impact_model

US = 1_000_000


def snap(ts, mid, vol_so_far):
    q = [VenueQuote("A", "X", round(mid - 0.01, 2), round(mid + 0.01, 2), 500, 500, ts),
         VenueQuote("B", "X", round(mid - 0.02, 2), round(mid + 0.01, 2), 300, 300, ts)]
    return MarketSnapshot("X", ts, q, mid, adv=2_000_000, volatility=0.02, session_volume=vol_so_far)


def run(name, side=1, alpha=0.0, seed=0):
    rng = np.random.default_rng(seed)
    f = make_execution_factory()
    o = ExecutionOrder("P1", "X", side, 5000, 0, 600 * US, urgency=0.5, alpha_bps=alpha, alpha_horizon_us=60 * US)
    st = f.create(name, o)
    assert isinstance(st, ExecutionStrategy)
    mid, vol = 100.0, 0.0
    s = snap(0, mid, vol)
    pending = st.start(o, s)
    fid = 0
    for t in range(0, 700):
        ts = t * US
        s = snap(ts, mid, vol)
        for c in st.cancels(s):
            assert c.reason.startswith("slice:")
        for sl in pending:
            fill_q = 0
            if sl.order_type in ("IOC", "MARKET"):
                fill_q = sl.quantity
            elif rng.random() < 0.3:
                fill_q = min(sl.quantity, int(rng.integers(1, 200)))
            if fill_q:
                fid += 1
                px = sl.limit_price
                pending_new = st.on_fill(Fill(f"f{fid}", f"{sl.slice_id}-v", sl.slice_id, "P1", "X", "A", side,
                                              fill_q, px, ts), s)
            if sl.order_type == "IOC" and fill_q < sl.quantity:
                st.on_event(OrderEvent("cancel", sl.slice_id + "-v", ts), s)
        if st.is_complete():
            break
        vol += 3000
        mid += rng.normal(0, 0.01)
        pending = st.on_market(s)
    return st


@pytest.mark.parametrize("name", ["twap", "vwap", "pov", "almgren_chriss", "obizhaeva_wang", "adaptive_passive",
                                  "market_making", "is_rl"])
@pytest.mark.parametrize("side", [1, -1])
def test_strategy_completes(name, side):
    st = run(name, side)
    assert st.is_complete()
    s = st.state()
    assert s["filled"] == 5000 and s["remaining"] == 0


def test_factory_and_impact():
    f = make_execution_factory()
    assert isinstance(f, ExecutionStrategyFactory)
    assert set(f.available()) >= {"twap", "vwap", "pov", "almgren_chriss", "is_rl", "market_making", "adaptive_passive"}
    im = make_impact_model()
    assert isinstance(im, ImpactModel)
    s = snap(0, 100.0, 0)
    assert im.expected_cost_bps(100_000, s, 3600 * US) > im.expected_cost_bps(1000, s, 3600 * US) > 0


def test_ac_urgency_frontloads():
    f = make_execution_factory()
    s = snap(0, 100.0, 0)
    lo = f.create("almgren_chriss", ExecutionOrder("a", "X", 1, 10000, 0, 1800 * US, urgency=0.0))
    hi = f.create("almgren_chriss", ExecutionOrder("b", "X", 1, 10000, 0, 1800 * US, urgency=1.0))
    lo.start(lo.order, s); hi.start(hi.order, s)
    assert hi.target_at(300 * US) > lo.target_at(300 * US) * 1.5
    adv = f.create("almgren_chriss", ExecutionOrder("c", "X", 1, 10000, 0, 1800 * US, urgency=0.0,
                                                    alpha_bps=20, alpha_horizon_us=600 * US))
    adv.start(adv.order, s)
    assert adv.target_at(300 * US) > lo.target_at(300 * US)
