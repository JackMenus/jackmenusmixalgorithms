"""单元测试。可用 pytest 运行，也可直接 ``python tests/test_factormine.py``。"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import factormine as fm  # noqa: E402
from factormine import operators as ops  # noqa: E402
from factormine.rl import build_vocab, rpn_to_expr  # noqa: E402

PANEL = fm.make_synthetic_panel(260, 60, seed=1)
FWD = fm.forward_returns(PANEL)
TRAIN = slice(0, 180)


def test_parse_roundtrip_and_errors():
    s = "cs_rank(neg(ts_corr(close, volume, 10)))"
    assert str(fm.parse(s)) == s
    for bad in ["foo(close)", "ts_mean(close)", "ts_mean(close, volume)", "ts_mean(close, 9999)", "3", "close)"]:
        try:
            fm.parse(bad)
        except fm.ParseError:
            continue
        raise AssertionError(f"应当报错: {bad}")


def test_no_lookahead():
    """截断未来数据后，历史时刻的因子值不应改变。"""
    ev_full = fm.Evaluator(PANEL)
    cut = {k: v.iloc[:200] for k, v in PANEL.items()}
    ev_cut = fm.Evaluator(cut)
    for s in ["ts_rank(close, 10)", "ts_corr(close, volume, 10)", "ts_decay_linear(returns, 5)",
              "cs_rank(delta(vwap, 3))", "ts_zscore(volume, 20)"]:
        a = ev_full(s).iloc[:200]
        b = ev_cut(s)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-8)


def test_ts_rank_matches_naive():
    x = PANEL["close"].iloc[:40, :5]
    fast = ops.ts_rank(x, 7)
    naive = x.rolling(7).apply(lambda v: (np.sum(v < v[-1]) + 0.5 * (np.sum(v == v[-1]) - 1)) / 6, raw=True)
    np.testing.assert_allclose(fast.iloc[6:].values, naive.iloc[6:].values, atol=1e-12)


def test_planted_signal_detected():
    f = fm.Evaluator(PANEL)("neg(cs_rank(ts_pct(close, 5)))")
    rep = fm.evaluate_factor(f, FWD)
    assert rep.rank_ic > 0.02, rep.summary()


def test_combine_and_orthogonalize():
    ev = fm.Evaluator(PANEL)
    facs = {s: ev(s) for s in ["neg(ts_pct(close, 5))", "neg(ts_pct(close, 10))", "neg(div(volume, ts_mean(volume, 20)))"]}
    for m in ["equal", "ic", "icir", "max_ir", "ridge"]:
        w = fm.fit_weights(facs, FWD, method=m, rows=TRAIN)
        assert np.isfinite(w).all()
        c = fm.combine(facs, w)
        assert abs(c.mean(axis=1).dropna()).max() < 1e-8
    orth = fm.orthogonalize(facs, "symmetric")
    a, b = orth[list(orth)[0]].iloc[100], orth[list(orth)[1]].iloc[100]
    assert abs(np.corrcoef(a, b)[0, 1]) < 1e-6
    dyn = fm.DynamicCombiner(lookback=40, top_k=2).fit_transform(facs, FWD)
    assert dyn.shape == FWD.shape


def test_rpn():
    vocab = {t.name: t for t in build_vocab()}
    e = rpn_to_expr([vocab["close"], vocab["volume"], vocab["ts_corr_10"], vocab["cs_rank"]])
    assert str(e) == "cs_rank(ts_corr(close, volume, 10))"


def test_miners_small():
    ev = fm.Evaluator(PANEL)
    pool = fm.FactorPool(ev, FWD, TRAIN, capacity=6, min_ic=0.01)
    fm.GPMiner(pool, fm.GPConfig(population=30, generations=2, seed=1), log=lambda s: None).run()
    fm.RLMiner(pool, config=fm.RLConfig(iterations=2, batch_size=8), log=lambda s: None).run()
    fm.LLMAlphaAgent(pool, fm.MockLLM(), fm.LLMConfig(rounds=2, per_round=4), log=lambda s: None).run()
    assert 1 <= len(pool) <= 6
    vals = list(pool.as_factors().values())
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            assert abs(fm.evaluate.factor_correlation(vals[i].iloc[TRAIN], vals[j].iloc[TRAIN])) <= 0.7 + 1e-9


def test_neural_factor():
    ev = fm.Evaluator(PANEL)
    feats = {s: ev(s) for s in ["ts_pct(close, 5)", "div(volume, ts_mean(volume, 20))", "ts_std(returns, 10)"]}
    nf = fm.NeuralFactor(fm.MLPConfig(epochs=3)).fit(feats, FWD, rows=TRAIN)
    out = nf.predict(feats)
    assert out.shape == FWD.shape and np.isfinite(out.values).all()


def test_library_model_signal_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        lib = fm.FactorLibrary(os.path.join(d, "lib.json"))
        lib.add(fm.FactorRecord("rev5", "neg(ts_pct(close, 5))", "manual"))
        lib.add(fm.FactorRecord("vol20", "neg(div(volume, ts_mean(volume, 20)))", "manual"))
        lib.save()
        lib2 = fm.FactorLibrary(os.path.join(d, "lib.json"))
        assert set(lib2.records) == {"rev5", "vol20"}
        model = fm.AlphaModel(lib2).fit(PANEL)
        model.save(os.path.join(d, "model.json"))
        model2 = fm.AlphaModel.load(os.path.join(d, "model.json"), lib2)
        sig = model2.signal(PANEL)
        d_ = json.loads(json.dumps(sig.to_dict()))
        sig2 = fm.AlphaSignal.from_dict(d_)
        assert d_["schema"] == "factormine.signal/1"
        assert 0 <= sig2.urgency <= 1
        w = sig.target_weights(gross=1.0, max_weight=0.05)
        assert abs(w.abs().sum() - 1) < 1e-6 and w.abs().max() <= 0.05 + 1e-9 and abs(w.sum()) < 1e-6
        wl = sig.target_weights(mode="long_only", max_weight=0.2)
        assert (wl >= 0).all()


def test_contract_adapter():
    """quantstack 契约适配：未安装集成层时返回 dict，安装时返回其 dataclass。"""
    miner = fm.make_miner(methods=("gp",), seed=2)
    specs = miner.mine(PANEL, budget=20)
    assert specs, "应至少挖出一个因子"
    get = lambda o, k: o[k] if isinstance(o, dict) else getattr(o, k)
    rep = miner.evaluate(specs[0], PANEL)
    assert get(rep, "sample") == "oos" and np.isfinite(get(rep, "ic_mean"))
    model = fm.make_alpha_model(library=miner.library)
    model.fit(PANEL)
    as_of = PANEL["close"].index[200]
    b1 = model.predict(PANEL, as_of=as_of)
    cut = {k: v.loc[:as_of] for k, v in PANEL.items()}
    b2 = model.predict(cut)
    assert get(b1, "ts_us") == get(b2, "ts_us") == int(as_of.value // 1000)
    s1 = {get(s, "symbol"): get(s, "score") for s in get(b1, "signals")}
    s2 = {get(s, "symbol"): get(s, "score") for s in get(b2, "signals")}
    assert all(abs(s1[k] - s2[k]) < 1e-9 for k in s1)  # 无未来函数
    assert all(0 <= get(s, "confidence") <= 1 for s in get(b1, "signals"))


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for f in fns:
        f()
        print("通过", f.__name__)
    print(f"全部 {len(fns)} 项测试通过")
