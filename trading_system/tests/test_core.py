from quantstack.contracts import Fill, MarketSnapshot, OrderEvent, OrderSlice, RiskLimits, TargetPosition, VenueQuote
from quantstack.core import OrderManager, PreTradeRiskGate, TCAEngine


def _snap(sym="AAA", bid=99.9, ask=100.1):
    return MarketSnapshot(sym, 0, [VenueQuote("X", sym, bid, ask, 500, 500, ts_us=0)], last_price=100.0, adv=1e6, volatility=0.02)


def test_oms_targets_and_pnl():
    oms = OrderManager(initial_cash=1e6)
    snap = {"AAA": _snap()}
    orders = oms.targets_to_orders([TargetPosition("AAA", 1000, urgency=0.7)], snap, 0, 60_000_000)
    assert len(orders) == 1 and orders[0].side == 1 and orders[0].quantity == 1000
    o = orders[0]
    oms.register_order(o)
    oms.apply_fill(Fill("F1", "S1-1", "S1", o.order_id, "AAA", "X", 1, 600, 100.1, 1, fee=1.8))
    oms.apply_fill(Fill("F2", "S2-1", "S2", o.order_id, "AAA", "X", 1, 400, 99.95, 2, fee=-0.8))
    assert o.status == "filled" and oms.position("AAA").quantity == 1000
    assert abs(oms.position("AAA").avg_price - 100.04) < 1e-9
    oms.apply_fill(Fill("F3", "S3-1", "S3", "O2", "AAA", "X", -1, 1000, 101.0, 3))
    assert oms.position("AAA").quantity == 0
    assert abs(oms.position("AAA").realized_pnl - 1000 * (101.0 - 100.04)) < 1e-6
    assert abs(oms.cash - (1e6 - 600 * 100.1 - 400 * 99.95 + 1000 * 101.0 - 1.0)) < 1e-6


def test_risk_gate_resizes_rejects_and_kills():
    oms = OrderManager(initial_cash=1e6)
    snap = _snap()
    gate = PreTradeRiskGate(RiskLimits(max_order_notional=50_000, max_daily_loss=1000))
    o = oms.targets_to_orders([TargetPosition("AAA", 1000)], {"AAA": snap}, 0, 60_000_000)[0]
    v = gate.check_order(o, oms.snapshot(0), snap)
    assert v.approved and v.adjusted_quantity == 500
    bad = OrderSlice("S1", o.order_id, "AAA", 1, 100, "LIMIT", limit_price=104.0)
    assert not gate.check_slice(bad, snap, 0).approved
    ok = OrderSlice("S2", o.order_id, "AAA", 1, 100, "LIMIT", limit_price=100.0)
    assert gate.check_slice(ok, snap, 0).approved
    # 亏损触发 kill switch
    pf = oms.snapshot(0)
    assert gate.check_pnl(pf) is None
    oms.cash -= 5000
    inc = gate.check_pnl(oms.snapshot(1))
    assert inc is not None and gate.limits.kill_switch
    assert not gate.check_order(o, oms.snapshot(1), snap).approved
    # agent 调参必须在区间内
    assert gate.set_limit("max_participation", 0.05)[0]
    assert not gate.set_limit("max_participation", 0.9)[0]
    assert not gate.set_limit("kill_switch", 0)[0]


def test_tca_report():
    tca = TCAEngine()
    oms = OrderManager()
    o = oms.targets_to_orders([TargetPosition("AAA", 1000)], {"AAA": _snap()}, 0, 60_000_000)[0]
    tca.on_order_start(o, _snap(), "twap")
    f = Fill("F1", "S1-1", "S1", o.order_id, "AAA", "X", 1, 1000, 100.1, 5, fee=3.0, liquidity="taker")
    tca.on_event(OrderEvent("fill", "S1-1", 5, "X", fill=f))
    r = tca.report(o.order_id)
    assert abs(r.implementation_shortfall_bps - 10.0) < 1e-6 and r.fill_ratio == 1.0 and r.maker_ratio == 0.0
    assert tca.summary()["n_orders"] == 1
