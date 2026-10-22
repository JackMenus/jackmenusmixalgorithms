import pytest

from quantstack.contracts import (
    AlphaSignal, ExecutionOrder, ExecutionStrategy, Fill, FactorReport, FactorSpec, MarketSnapshot, OrderEvent, OrderSlice,
    RiskLimits, Router, SignalBatch, VenueQuote,
)


def _snap():
    return MarketSnapshot("AAPL", 1, [VenueQuote("NYSE", "AAPL", 100.0, 100.02, 500, 300, ts_us=1), VenueQuote("ARCA", "AAPL", 99.99, 100.01, 200, 800, ts_us=1)], last_price=100.0, adv=5e6, volatility=0.02)


def test_snapshot_helpers_and_roundtrip():
    s = _snap()
    assert s.nbbo() == (100.0, 100.01)
    assert s.displayed_size(1) == 1100 and s.displayed_size(-1) == 700
    assert MarketSnapshot.from_dict(s.to_dict()) == s
    assert MarketSnapshot.from_json(s.to_json()) == s


def test_nested_roundtrip():
    sig = AlphaSignal("AAPL", 1, 23_400_000_000, 35.0, confidence=0.7, score=1.2, source="ref", factor_exposures={"rev5": 0.4})
    b = SignalBatch(1, [sig], model_id="m")
    assert SignalBatch.from_dict(b.to_dict()) == b
    rep = FactorReport(FactorSpec("f1", "cs_rank(delta(close,5))", "gp"), 0.03, 0.5, coverage=0.9)
    assert FactorReport.from_dict(rep.to_dict()) == rep and rep.is_tradeable
    ev = OrderEvent("fill", "S1-1", 5, fill=Fill("f1", "S1-1", "S1", "O1", "AAPL", "NYSE", 1, 500, 100.02, 5, fee=1.5))
    assert OrderEvent.from_dict(ev.to_dict()) == ev
    assert RiskLimits.from_dict(RiskLimits().to_dict()) == RiskLimits()


def test_order_fill_bookkeeping():
    o = ExecutionOrder("o1", "AAPL", 1, 10000, 0, 10**6)
    o.apply_fill(4000, 100.1)
    o.apply_fill(6000, 100.2)
    assert o.remaining == 0 and o.status == "filled"
    assert abs(o.avg_fill_price - 100.16) < 1e-9


def test_slice_validation():
    with pytest.raises(ValueError):
        OrderSlice("s", "o", "A", 1, 10, "BAD")
    with pytest.raises(ValueError):
        OrderSlice("s", "o", "A", 2, 10, "IOC")
    with pytest.raises(ValueError):
        OrderSlice("s", "o", "A", 1, 0, "IOC")


def test_protocols_are_runtime_checkable():
    from quantstack.reference.execution import ReferenceExecutionFactory
    from quantstack.reference.routing import CostLatencyRouter

    o = ExecutionOrder("o1", "AAPL", 1, 1000, 0, 60 * 10**6)
    assert isinstance(ReferenceExecutionFactory().create("twap", o), ExecutionStrategy)
    assert isinstance(CostLatencyRouter(["NYSE"]), Router)
