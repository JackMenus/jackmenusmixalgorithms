import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latency_routing import (  # noqa: E402
    ParentOrder, RoutingService, TOOL_SCHEMA, VenueLatencyModel, VenueQuote,
    P2Quantile, estimate_clock_offset, plan_synchronized_send, solve_cont_kukanov,
)
from latency_routing.infra.ring_buffer import SPSCRing  # noqa: E402
from latency_routing.sim.latency_arb_sim import SimVenue, simulate_sweep  # noqa: E402
from latency_routing.sor.cont_kukanov import ck_cost, default_outflow_samples  # noqa: E402
from latency_routing.sor.stale_quote import survival_prob  # noqa: E402


def test_p2_quantile_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.lognormal(5, 0.5, 20000)
    for q in (0.5, 0.9, 0.99):
        est = P2Quantile(q)
        for v in x:
            est.update(v)
        assert abs(est.value / np.quantile(x, q) - 1) < 0.03


def test_latency_model_detects_spikes_and_samples():
    rng = np.random.default_rng(1)
    m = VenueLatencyModel()
    body = rng.lognormal(np.log(200), 0.1, 5000)
    body[rng.random(5000) < 0.02] = 5000.0
    m.observe_many("A", body)
    assert 0.01 < m.spike_prob("A") < 0.04
    s = m.sample("A", 20000, rng)
    assert abs(np.median(s) / 200 - 1) < 0.1
    assert np.quantile(s, 0.995) > 1000


def test_clock_offset():
    rng = np.random.default_rng(2)
    true_off = 1234.0
    samples = []
    for _ in range(500):
        t1 = rng.uniform(0, 1e6)
        up = 50 + rng.exponential(30)
        down = 50 + rng.exponential(30)
        t2 = t1 + up + true_off
        t3 = t2 + 5
        t4 = t3 - true_off + down
        samples.append((t1, t2, t3, t4))
    off, unc = estimate_clock_offset(samples)
    assert abs(off - true_off) < 10
    assert unc < 60


def test_survival_prob_monotone():
    assert survival_prob(100, 100.0) > survival_prob(100, 1000.0)
    assert survival_prob(0, 1e6) == pytest.approx(1.0)


def test_cont_kukanov_beats_naive_and_uses_market_when_urgent():
    rng = np.random.default_rng(3)
    Q = np.array([500.0, 800.0, 300.0])
    xi = default_outflow_samples(Q, [1.2, 1.0, 1.5], n=3000, rng=rng)
    S, h, f = 1000.0, 0.005, 0.003
    r = np.array([0.002, 0.003, 0.001])
    lazy = solve_cont_kukanov(S, Q, xi, h, f, r, lam_u=0.004, lam_o=0.02)
    urgent = solve_cont_kukanov(S, Q, xi, h, f, r, lam_u=0.05, lam_o=0.02)
    naive = np.concatenate([[0.0], np.full(3, S / 3)])
    naive_cost = ck_cost(naive, S, Q, xi, h, f, r, 0.05, 0.02).mean() / S
    assert urgent.expected_cost <= naive_cost + 1e-9
    assert urgent.market_qty > lazy.market_qty
    assert lazy.market_qty < 1.0  # λ_u < h+f 时论文结论：不用市价单


def test_sync_send_improves_success():
    rng = np.random.default_rng(4)
    lat = {
        "NEAR": rng.lognormal(np.log(60), 0.15, 5000),
        "MID": rng.lognormal(np.log(300), 0.1, 5000),
        "FAR": rng.lognormal(np.log(900), 0.08, 5000),
    }
    plan = plan_synchronized_send(list(lat), lat, window_us=200)
    assert plan.naive_success_prob < 0.05
    assert plan.success_prob > 0.6
    assert plan.send_delay_us["FAR"] == pytest.approx(0.0)
    assert plan.send_delay_us["NEAR"] > plan.send_delay_us["MID"] > 0


def _service():
    rng = np.random.default_rng(5)
    svc = RoutingService()
    for v, mu in (("NEAR", 60), ("MID", 300), ("FAR", 900)):
        svc.latency.observe_many(v, rng.lognormal(np.log(mu), 0.1, 2000))
    quotes = [
        VenueQuote("NEAR", 99.99, 100.00, 300, 300, taker_fee=0.003, maker_fee=-0.002),
        VenueQuote("MID", 99.99, 100.00, 400, 400, taker_fee=0.002, maker_fee=-0.001),
        VenueQuote("FAR", 99.98, 100.00, 500, 500, taker_fee=0.001, maker_fee=0.0),
    ]
    return svc, quotes


def test_router_end_to_end_and_sync_beats_naive_in_sim():
    svc, quotes = _service()
    dec = svc.route(ParentOrder("XYZ", +1, 1000, urgency=1.0, order_id="p1"), quotes)
    ioc = [c for c in dec.children if c.order_type == "IOC"]
    assert sum(c.quantity for c in ioc) >= 900
    assert {c.venue for c in ioc} == {"NEAR", "MID", "FAR"}
    assert dec.diagnostics["sync_success_prob"] > dec.diagnostics["sync_naive_prob"]

    venues = {q.venue: SimVenue(q.venue, q.ask_size, svc.latency.sample(q.venue, 5000)) for q in quotes}
    synced = simulate_sweep(ioc, venues, reaction_window_us=200)
    for c in ioc:
        c.send_delay_us = 0.0
    naive = simulate_sweep(ioc, venues, reaction_window_us=200)
    assert synced["fill_ratio_mean"] > naive["fill_ratio_mean"] + 0.2


def test_router_passive_and_json_roundtrip():
    svc, quotes = _service()
    payload = {
        "order": {"symbol": "XYZ", "side": -1, "quantity": 800, "urgency": 0.0, "order_id": "p2"},
        "quotes": [q.to_dict() for q in quotes],
    }
    out = svc.route_json(json.loads(json.dumps(payload)))
    json.dumps(out)
    posts = [c for c in out["children"] if c["order_type"] == "POST_ONLY"]
    assert posts and all(c["price"] == 100.00 for c in posts)
    assert TOOL_SCHEMA["name"] == "route_order"


def test_alpha_raises_aggression():
    svc, quotes = _service()
    calm = svc.route(ParentOrder("XYZ", 1, 1000, urgency=0.2), quotes)
    hot = svc.route(ParentOrder("XYZ", 1, 1000, urgency=0.2, alpha_bps=5.0), quotes)
    assert hot.diagnostics["urgency_effective"] > calm.diagnostics["urgency_effective"]


def test_spsc_ring():
    q = SPSCRing[int](8)
    assert all(q.try_push(i) for i in range(8))
    assert not q.try_push(99)
    assert q.drain() == list(range(8))
    assert q.try_pop() is None
    for i in range(100):
        assert q.try_push(i)
        assert q.try_pop() == i
    with pytest.raises(ValueError):
        SPSCRing(6)


def test_feedback_loop_on_event_and_venue_stats():
    from types import SimpleNamespace as NS
    from latency_routing import RouterProtocol, make_router

    r = make_router(["NEAR", "FAR"], latency_prior_us={"NEAR": 60, "FAR": 900}, reaction_window_us=80)
    assert isinstance(r, RouterProtocol)
    quotes = [
        VenueQuote("NEAR", 99.99, 100.00, 300, 300, taker_fee=0.003),
        VenueQuote("FAR", 99.99, 100.00, 500, 500, taker_fee=0.001),
    ]
    dec = r.route(ParentOrder("XYZ", 1, 600, urgency=1.0), quotes)
    sent = {c.venue: c.quantity for c in dec.children}
    for _ in range(50):
        r.on_event(NS(kind="fill", venue="NEAR", fill=NS(quantity=2, price=100.01, side=1, liquidity="taker", latency_us=200.0, symbol="XYZ")))
    r.on_event(NS(kind="reject", venue="FAR", fill=None))
    r.on_event(NS(kind="ack", venue="FAR", fill=None))
    stats = {s.venue: s for s in r.venue_stats()}
    near = stats["NEAR"]
    assert near.fills == 50 and near.fill_ratio == pytest.approx(100 / sent["NEAR"])
    assert near.avg_slippage_bps == pytest.approx(0.015 / 99.995 * 1e4)   # mid=99.995 → 100.01
    assert 90 < near.latency_p50_us < 110                           # RTT 200 → 单程 100
    assert stats["FAR"].rejects == 1 and stats["FAR"].fills == 0
