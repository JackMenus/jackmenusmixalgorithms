"""评估台：对每个协议层给出"本算法 vs 基线"的量化结论（``EvaluationReport``）。

| 协议 | 评估方式 | 主目标 | 基线 |
|---|---|---|---|
| AlphaModel | 训练段拟合，样本外逐日 ``predict(as_of)`` 滚动出分数面板，factormine 评估 RankIC/ICIR/换手 | rank_ic | quantstack 参考 ReversalVolumeAlpha |
| PortfolioConstructor | 固定同一信号（默认受控 IC 的合成信号，也可用参考 alpha），多面板样本外每 k 日再平衡的权重回测（扣线性成本） | sharpe | quantstack 参考 LongShortConstructor |
| ExecutionStrategy | ``build_sim_pipeline`` 合成多场所市场 + 撮合模拟器，多种子成对比较 TCA 实现偏差 | shortfall_bps | order_placement 的 almgren_chriss / twap |
| StrategySelector | 同一模拟器连续多轮母单，选择器在线学习，比较后半程实现偏差 | shortfall_bps | quantstack 参考 choose_strategy |
| Router | 同一模拟器，叠加层 vs 底层路由器，多种子成对比较 | shortfall_bps | latency_routing 适配器 |

所有随机性由 ``seed`` 控制；``budget`` 预设（quick / standard / full）决定样本量与耗时。
通过（passed）判定写在各函数的 docstring 里，并在报告 ``verdict`` 中用中文复述。
"""
from __future__ import annotations

import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._paths import require
from .blueprint import EvaluationReport
from .families import FamilySpec

US = 1_000_000

BUDGETS: Dict[str, Dict[str, Any]] = {
    "quick": dict(alpha_stride=3, portfolio_every=5, portfolio_extra_panels=0, portfolio_controlled_ic=0.08, exec_orders=3, exec_seeds=2, exec_duration_s=480, selector_rounds=4, selector_orders=4, selector_seeds=1,
                  router_orders=4, router_seeds=2, router_duration_s=480, step_s=2.0, mc_samples=400),
    "standard": dict(alpha_stride=1, portfolio_every=2, portfolio_extra_panels=2, portfolio_controlled_ic=0.08, exec_orders=4, exec_seeds=4, exec_duration_s=900, selector_rounds=6, selector_orders=5, selector_seeds=2,
                     router_orders=5, router_seeds=3, router_duration_s=600, step_s=1.0, mc_samples=400),
    "full": dict(alpha_stride=1, portfolio_every=1, portfolio_extra_panels=4, portfolio_controlled_ic=0.08, exec_orders=6, exec_seeds=8, exec_duration_s=1200, selector_rounds=10, selector_orders=6, selector_seeds=3,
                 router_orders=6, router_seeds=8, router_duration_s=1200, step_s=1.0, mc_samples=4000),
}

# 执行 / 选择器评估用 quantstack 参考路由器（便宜且与被测对象无关）；路由评估用 latency_routing 适配器作底层与基线
SIM_KINDS = {"alpha": "reference", "execution": "order_placement", "router": "reference"}
ROUTER_SIM_KINDS = {"alpha": "reference", "execution": "order_placement", "router": "latency_routing"}


def _budget(b) -> Dict[str, Any]:
    if isinstance(b, dict):
        return {**BUDGETS["quick"], **b}
    return dict(BUDGETS[b])


def _finite(x, default=float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ============================================================================ 因子层
def _score_panel(model, panel, dates, use_fast: bool = True) -> pd.DataFrame:
    """样本外分数面板。模型若提供 ``score_panel``（逐行只用历史数据的一次性出分）则直接切片，否则逐日 ``predict(as_of)``。"""
    if use_fast and hasattr(model, "score_panel"):
        sc = model.score_panel(panel)
        return sc.loc[list(dates)].reindex(columns=panel["close"].columns)
    rows = {}
    for d in dates:
        batch = model.predict(panel, as_of=d)
        rows[d] = {s.symbol: s.score for s in batch.signals}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=panel["close"].columns).sort_index()


def alpha_metrics(scores: pd.DataFrame, fwd: pd.DataFrame, horizon: int) -> Dict[str, float]:
    require("factormine")
    from factormine import evaluate_factor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = evaluate_factor(scores, fwd.loc[scores.index], horizon=horizon)
    return {"rank_ic": _finite(rep.rank_ic, 0.0), "rank_icir": _finite(rep.rank_icir, 0.0), "ic_t": _finite(rep.ic_t, 0.0), "ls_sharpe": _finite(rep.ls_sharpe, 0.0),
            "turnover": _finite(rep.turnover, 0.0), "coverage": _finite(rep.coverage, 0.0), "n_periods": float(rep.n_periods)}


def evaluate_alpha(model, panel, train_ratio: float = 0.7, stride: int = 1, baselines: Optional[Dict[str, Any]] = None, horizon: Optional[int] = None,
                   min_t: float = 2.0) -> EvaluationReport:
    """通过条件：样本外 RankIC > 0 且 t 统计量 ≥ min_t（统计显著）。是否优于基线单独在 notes 中说明。"""
    require("factormine", "quantstack")
    from factormine import forward_returns

    t0 = time.time()
    T = len(panel["close"])
    cut = int(T * train_ratio)
    h = int(horizon or getattr(model, "horizon", 1))
    train = {k: v.iloc[:cut] for k, v in panel.items()}
    model.fit(train)
    dates = list(panel["close"].index[cut: T - h: max(int(stride), 1)])
    fwd = forward_returns(panel, h)
    metrics = alpha_metrics(_score_panel(model, panel, dates), fwd, h)
    if baselines is None:
        from quantstack.reference.alpha import ReversalVolumeAlpha

        baselines = {"reference.ReversalVolumeAlpha": ReversalVolumeAlpha(horizon=h)}
    base_metrics: Dict[str, Dict[str, float]] = {}
    for name, b in baselines.items():
        b.fit(train)
        base_metrics[name] = alpha_metrics(_score_panel(b, panel, dates), fwd, h)
    rep = EvaluationReport(algo_id=getattr(model, "model_id", "alpha"), protocol="AlphaModel", metrics=metrics, baselines=base_metrics, objective="rank_ic", higher_is_better=True,
                           settings={"train_ratio": train_ratio, "n_train": cut, "n_oos": len(dates), "stride": stride, "horizon": h, "n_assets": int(panel["close"].shape[1])})
    rep.passed = bool(metrics["rank_ic"] > 0 and metrics["ic_t"] >= min_t)
    imp = rep.improvement()
    beats = imp is not None and imp > 0
    rep.verdict = (f"样本外 RankIC={metrics['rank_ic']:.4f}（t={metrics['ic_t']:.2f}，{'显著' if rep.passed else '不显著'}），"
                   f"{'优于' if beats else '不及'}基线 {rep.baseline_primary():.4f}；换手 {metrics['turnover']:.3f}。")
    rep.notes.append("通过条件：样本外 RankIC > 0 且 t ≥ %.1f；'优于基线' 不是通过的必要条件（算法假设不同，覆盖的信息也不同）。" % min_t)
    if not beats:
        rep.notes.append("提示：合成面板只植入了反转与量价背离两类信号，参考基线恰好直接使用它们，因此在合成数据上难以被超越；请在真实面板上复核。")
    rep.wall_s = time.time() - t0
    return rep


# ============================================================================ 组合层
def _snapshots(panel, t, adv_window: int = 20, spread_bps: float = 2.0) -> Dict[str, Any]:
    from quantstack.contracts import MarketSnapshot, VenueQuote

    close = panel["close"].loc[:t]
    px = close.iloc[-1]
    adv = panel["volume"].loc[:t].iloc[-adv_window:].mean()
    vol = close.pct_change(fill_method=None).iloc[-adv_window:].std()
    out = {}
    for s, p in px.items():
        if not np.isfinite(p) or p <= 0:
            continue
        half = p * spread_bps * 1e-4 / 2
        out[s] = MarketSnapshot(symbol=s, ts_us=int(pd.Timestamp(t).value // 1000), quotes=[VenueQuote("SIM", s, p - half, p + half, 5000, 5000, taker_fee=0.003, maker_fee=-0.002)],
                                last_price=float(p), adv=float(adv.get(s, 1e6)) if np.isfinite(adv.get(s, np.nan)) else 1e6, volatility=float(vol.get(s, 0.02)) if np.isfinite(vol.get(s, np.nan)) else 0.02)
    return out


def _portfolio_snapshot(weights: Dict[str, float], nav: float, market, ts_us: int):
    from quantstack.contracts import PortfolioSnapshot, Position

    positions = {}
    invested = 0.0
    for s, w in weights.items():
        if s in market and abs(w) > 0:
            px = market[s].mid
            q = int(round(w * nav / px))
            positions[s] = Position(symbol=s, quantity=q, avg_price=px, last_price=px)
            invested += q * px
    return PortfolioSnapshot(ts_us=ts_us, cash=nav - invested, positions=positions)


def portfolio_backtest(constructor, panel, alpha, dates, rebalance_every: int, cost_bps: float, limits, nav: float = 1e7) -> Dict[str, float]:
    close = panel["close"]
    idx = list(close.index)
    pos = {d: i for i, d in enumerate(idx)}
    w_prev: Dict[str, float] = {}
    rets, turns, hhis, maxw = [], [], [], []
    for t in dates:
        i = pos[t]
        j = min(i + rebalance_every, len(idx) - 1)
        if j <= i:
            break
        market = _snapshots(panel, t)
        hist = {k: v.loc[:t] for k, v in panel.items()}
        if hasattr(constructor, "fit"):
            constructor.fit(hist)
        batch = alpha.predict(panel, as_of=t)
        ts_us = int(pd.Timestamp(t).value // 1000)
        targets = constructor.construct(batch, _portfolio_snapshot(w_prev, nav, market, ts_us), market, limits)
        w_new = dict(w_prev)
        for tg in targets:
            w_new[tg.symbol] = float(tg.target_weight)
        w_new = {s: w for s, w in w_new.items() if abs(w) > 1e-9}
        period = (close.iloc[j] / close.iloc[i] - 1.0)
        gross = sum(abs(w) for w in w_new.values())
        turnover = sum(abs(w_new.get(s, 0.0) - w_prev.get(s, 0.0)) for s in set(w_new) | set(w_prev))
        r = sum(w * _finite(period.get(s, 0.0), 0.0) for s, w in w_new.items()) - cost_bps * 1e-4 * turnover
        rets.append(r)
        turns.append(turnover)
        hhis.append(sum((abs(w) / gross) ** 2 for w in w_new.values()) if gross > 0 else 0.0)
        maxw.append(max((abs(w) for w in w_new.values()), default=0.0))
        w_prev = w_new
    r = np.asarray(rets)
    ann = 252.0 / rebalance_every
    sharpe = float(r.mean() / r.std() * math.sqrt(ann)) if len(r) > 2 and r.std() > 0 else 0.0
    eq = np.cumprod(1 + r)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0
    return {"sharpe": sharpe, "ann_return": float(r.mean() * ann), "ann_vol": float(r.std() * math.sqrt(ann)), "max_drawdown": dd,
            "avg_turnover": float(np.mean(turns)) if turns else 0.0, "avg_hhi": float(np.mean(hhis)) if hhis else 0.0, "max_weight": float(np.max(maxw)) if maxw else 0.0,
            "n_rebalances": float(len(r))}


class ControlledICAlpha:
    """受控信息系数的合成信号，仅供组合层评估台使用（故意使用未来收益，不是可交易的 alpha）。

    score_t = z( IC·z(未来 horizon 日收益) + √(1−IC²)·ε )，期望收益 = IC·σ_i·z_i（Grinold 法则），σ_i 为截至 t 的 60 日波动率。
    用它把"组合构造质量"与"alpha 质量"分离：IC 固定，不同构造器在同一信号下比较实现夏普。噪声由 (seed, t) 决定，完全可复现。
    """

    model_id = "bench_controlled_ic"

    def __init__(self, ic: float = 0.08, horizon: int = 5, seed: int = 0, vol_window: int = 60):
        self.ic, self.horizon, self.seed, self.vol_window = float(ic), int(horizon), int(seed), int(vol_window)
        self.horizon_us = int(self.horizon * 86_400_000_000)

    def fit(self, panel, **kwargs) -> None:  # 无需拟合
        return None

    def predict(self, panel, as_of=None, **kwargs):
        from quantstack.contracts import AlphaSignal, SignalBatch

        close = panel["close"]
        i = len(close) - 1 if as_of is None else int(close.index.get_indexer([pd.Timestamp(as_of)])[0])
        j = min(i + self.horizon, len(close) - 1)
        fwd = (close.iloc[j] / close.iloc[i] - 1.0).to_numpy(float)
        rets = close.iloc[max(0, i - self.vol_window): i + 1].pct_change(fill_method=None)
        sigma = rets.std().fillna(0.02).clip(lower=1e-4).to_numpy(float)
        rng = np.random.default_rng([self.seed, i])
        zf = _zs(fwd)
        raw = self.ic * zf + math.sqrt(max(1.0 - self.ic ** 2, 0.0)) * rng.standard_normal(len(fwd))
        z = _zs(raw)
        ts_us = int(pd.Timestamp(close.index[i]).value // 1000)
        sigs = [AlphaSignal(symbol=str(sym), ts_us=ts_us, horizon_us=self.horizon_us, expected_return_bps=float(self.ic * sigma[k] * z[k] * 1e4),
                            confidence=0.5, score=float(z[k]), source=self.model_id, factor_exposures={"controlled": float(z[k])},
                            metadata={"idio_vol": float(sigma[k]), "ic": self.ic, "half_life_bars": float(self.horizon), "urgency": 0.33})
                for k, sym in enumerate(close.columns) if np.isfinite(z[k])]
        return SignalBatch(ts_us=ts_us, signals=sigs, model_id=self.model_id, universe_size=close.shape[1])

    def describe(self) -> Dict[str, Any]:
        return {"model_id": self.model_id, "ic": self.ic, "horizon": self.horizon, "seed": self.seed}


def _zs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    return (x - m) / s if s > 0 else np.zeros_like(x)


def _mean_metrics(ms: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ms[0].keys()
    return {k: float(np.mean([m[k] for m in ms])) for k in keys}


def evaluate_portfolio(constructor, panel, train_ratio: float = 0.7, rebalance_every: int = 5, cost_bps: float = 5.0, alpha=None, baselines: Optional[Dict[str, Any]] = None,
                       tolerance: float = 0.1, extra_panels: int = 0, seed: int = 0, controlled_ic: float = 0.0) -> EvaluationReport:
    """通过条件：样本外夏普 > 0 且不低于基线夏普 − tolerance·|基线夏普|（同一 alpha、同一再平衡频率、同一成本）。

    单个面板上几十次再平衡的夏普标准误约 √(年化因子/期数)，噪声很大；``extra_panels`` > 0 时再生成同形状、不同种子的合成面板，
    对所有面板的指标取平均（逐面板夏普写入 metrics["sharpe_per_panel"]）。
    """
    require("quantstack")
    from quantstack.contracts import RiskLimits

    t0 = time.time()
    if alpha is None and controlled_ic > 0:
        alpha = ControlledICAlpha(ic=controlled_ic, seed=seed)
    elif alpha is None:
        from quantstack.reference.alpha import ReversalVolumeAlpha

        alpha = ReversalVolumeAlpha()
    if baselines is None:
        from quantstack.reference.portfolio import LongShortConstructor

        kw = {k: getattr(constructor, k) for k in ("top_k", "gross_target", "max_weight", "cost_threshold_bps") if hasattr(constructor, k)}
        baselines = {"reference.LongShortConstructor": LongShortConstructor(**kw)}
    panels = [panel]
    if extra_panels > 0:
        from quantstack.sim.market import make_synthetic_panel

        T0, N0 = panel["close"].shape
        panels += [make_synthetic_panel(n_days=T0, n_assets=N0, seed=seed * 7919 + 1000 + k) for k in range(int(extra_panels))]
    limits = RiskLimits()
    mine_ms: List[Dict[str, float]] = []
    base_ms: Dict[str, List[Dict[str, float]]] = {name: [] for name in baselines}
    n_dates = 0
    for pn in panels:
        T = len(pn["close"])
        cut = int(T * train_ratio)
        alpha.fit({k: v.iloc[:cut] for k, v in pn.items()})
        dates = list(pn["close"].index[cut: T - rebalance_every: rebalance_every])
        n_dates = len(dates)
        mine_ms.append(portfolio_backtest(constructor, pn, alpha, dates, rebalance_every, cost_bps, limits))
        for name, b in baselines.items():
            base_ms[name].append(portfolio_backtest(b, pn, alpha, dates, rebalance_every, cost_bps, limits))
    metrics = _mean_metrics(mine_ms)
    metrics["sharpe_per_panel"] = [round(m["sharpe"], 4) for m in mine_ms]  # type: ignore[assignment]
    metrics["n_panels"] = float(len(panels))
    base_metrics = {name: _mean_metrics(ms) for name, ms in base_ms.items()}
    for name, ms in base_ms.items():
        base_metrics[name]["sharpe_per_panel"] = [round(m["sharpe"], 4) for m in ms]  # type: ignore[assignment]
    wins = sum(1 for a, b_ in zip(mine_ms, list(base_ms.values())[0]) if a["sharpe"] >= b_["sharpe"])
    metrics["wins_vs_baseline"] = wins / len(panels)
    rep = EvaluationReport(algo_id="portfolio", protocol="PortfolioConstructor", metrics=metrics, baselines=base_metrics, objective="sharpe", higher_is_better=True,
                           settings={"train_ratio": train_ratio, "n_rebalances_per_panel": n_dates, "n_panels": len(panels), "rebalance_every": rebalance_every, "cost_bps": cost_bps,
                                     "alpha": type(alpha).__name__, "alpha_params": alpha.describe() if hasattr(alpha, "describe") else {}})
    b = rep.baseline_primary() or 0.0
    b0 = list(base_metrics.values())[0]
    rep.passed = bool(metrics["sharpe"] > 0 and metrics["sharpe"] >= b - tolerance * abs(b))
    rep.verdict = (f"样本外夏普（{len(panels)} 个面板平均）{metrics['sharpe']:.2f} vs 基线 {b:.2f}（逐面板胜率 {metrics['wins_vs_baseline']:.0%}）；"
                   f"年化波动 {metrics['ann_vol']:.2%} vs {b0['ann_vol']:.2%}；集中度 HHI {metrics['avg_hhi']:.3f} vs {b0['avg_hhi']:.3f}；换手 {metrics['avg_turnover']:.2f} vs {b0['avg_turnover']:.2f}。")
    rep.notes.append(f"通过条件：夏普 > 0 且 ≥ 基线 − {tolerance:.0%}·|基线|；两者使用同一 alpha（{type(alpha).__name__}）与成本假设（{cost_bps} bps 单边），每 {rebalance_every} 日再平衡。")
    if isinstance(alpha, ControlledICAlpha):
        rep.notes.append(f"信号为受控 IC={alpha.ic} 的合成信号（含未来信息，只用于把组合构造质量与 alpha 质量分离），不代表可交易收益。")
    rep.wall_s = time.time() - t0
    return rep


# ============================================================================ 模拟器公共部分
def make_orders(pipe, n: int, seed: int, start_us: Optional[int] = None, duration_s: float = 900.0, adv_frac: Tuple[float, float] = (0.002, 0.012)) -> List[Any]:
    """在模拟器上生成 n 张随机母单（规模 0.2%–1.2% ADV，随机方向 / 紧急度 / alpha）。"""
    from quantstack.contracts import ExecutionOrder

    rng = np.random.default_rng(seed)
    start_us = pipe.now() if start_us is None else int(start_us)
    syms = pipe.feed.symbols()
    out = []
    for i in range(n):
        s = syms[int(rng.integers(len(syms)))]
        snap = pipe.feed.snapshot(s, start_us)
        qty = int(max(500, rng.uniform(*adv_frac) * snap.adv))
        out.append(ExecutionOrder(order_id=f"E{seed}-{start_us // US}-{i}", symbol=s, side=int(rng.choice([-1, 1])), quantity=qty, start_us=start_us,
                                  end_us=start_us + int(duration_s * US), urgency=float(rng.uniform(0.2, 0.9)), alpha_bps=float(rng.normal(0.0, 3.0)),
                                  alpha_horizon_us=int(2 * duration_s * US), max_participation=0.1, arrival_price=snap.mid))
    return out


def _sim_pipeline(panel, seed: int, kinds: Optional[Dict[str, str]] = None, step_s: float = 1.0, router_cfg: Optional[Dict[str, Any]] = None):
    require("quantstack", "order_placement")
    from quantstack.core.pipeline import PipelineConfig
    from quantstack.core.registry import build_sim_pipeline

    cfg = PipelineConfig(step_us=int(max(step_s, 0.5) * US))
    return build_sim_pipeline(panel, kinds={**SIM_KINDS, **(kinds or {})}, seed=seed, config=cfg, router_cfg=router_cfg)


def _tca_rows(pipe, orders) -> List[Dict[str, float]]:
    rows = []
    for o in orders:
        r = pipe.tca.report(o.order_id)
        if r is None:
            continue
        rows.append({"shortfall_bps": _finite(r.implementation_shortfall_bps, 0.0), "fill_ratio": _finite(r.fill_ratio, 0.0), "maker_ratio": _finite(r.maker_ratio, 0.0),
                     "fees_bps": _finite(r.fees / max(r.filled_quantity * r.avg_fill_price, 1e-9) * 1e4, 0.0)})
    return rows


def _agg(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"shortfall_bps": float("nan"), "fill_ratio": 0.0, "maker_ratio": 0.0, "n_orders": 0.0}
    sf = np.array([r["shortfall_bps"] for r in rows])
    return {"shortfall_bps": float(sf.mean()), "shortfall_median_bps": float(np.median(sf)), "shortfall_std_bps": float(sf.std()),
            "fill_ratio": float(np.mean([r["fill_ratio"] for r in rows])), "maker_ratio": float(np.mean([r["maker_ratio"] for r in rows])), "n_orders": float(len(rows))}


def _run_orders(pipe, orders, plans=None) -> List[Dict[str, float]]:
    for o in orders:
        pipe.oms.register_order(o)
    pipe.start_execution(orders, plans)
    pipe.run_until_done()
    return _tca_rows(pipe, orders)


# 基线与候选参数无关：同一面板 / 种子 / 预算下只模拟一次，供同一进程内的多个候选复用（搜索时省掉约一半模拟耗时）
_BASELINE_CACHE: Dict[Tuple, Any] = {}


def _panel_key(panel) -> Tuple:
    c = panel["close"]
    return (tuple(c.shape), str(c.index[0]), str(c.index[-1]), float(np.nansum(c.values[0])), float(np.nansum(c.values[-1])))


def _cached(key: Tuple, fn: Callable[[], Any]) -> Any:
    if key not in _BASELINE_CACHE:
        _BASELINE_CACHE[key] = fn()
    return _BASELINE_CACHE[key]


def clear_baseline_cache() -> None:
    _BASELINE_CACHE.clear()


# ============================================================================ 执行层
def evaluate_execution(factory_builder: Callable[[Any], Any], panel, strategy: str = "kalman_pid", params: Optional[Dict[str, Any]] = None,
                       baselines: Sequence[str] = ("almgren_chriss", "twap"), n_orders: int = 4, seeds: Sequence[int] = (1, 2, 3), duration_s: float = 900.0,
                       tolerance_bps: float = 1.0, step_s: float = 1.0) -> EvaluationReport:
    """同一组母单与同一市场种子下成对比较。通过条件：平均实现偏差 ≤ 第一基线 + tolerance_bps 且成交率 ≥ 0.95。

    factory_builder(base_factory) -> 实现 ExecutionStrategyFactory 的对象（通常是 AutoExecutionFactory）。
    """
    t0 = time.time()
    rows: List[Dict[str, float]] = []
    base_rows: Dict[str, List[Dict[str, float]]] = {b: [] for b in baselines}
    wins = 0
    for sd in seeds:
        pipe = _sim_pipeline(panel, sd, step_s=step_s)
        pipe.execution_factory = factory_builder(pipe.execution_factory)
        orders = make_orders(pipe, n_orders, sd, duration_s=duration_s)
        mine = _run_orders(pipe, orders, {o.order_id: {"strategy": strategy, "params": dict(params or {})} for o in orders})
        rows += mine
        per_seed_base = None
        for b in baselines:
            def _base(b=b):
                pb = _sim_pipeline(panel, sd, step_s=step_s)
                ob = make_orders(pb, n_orders, sd, duration_s=duration_s)
                return _run_orders(pb, ob, {o.order_id: {"strategy": b} for o in ob})
            br = list(_cached(("exec", _panel_key(panel), sd, b, n_orders, duration_s, step_s), _base))
            base_rows[b] += br
            if per_seed_base is None:
                per_seed_base = br
        if per_seed_base and mine and _agg(mine)["shortfall_bps"] <= _agg(per_seed_base)["shortfall_bps"]:
            wins += 1
    metrics = _agg(rows)
    metrics["n_seeds"] = float(len(seeds))
    metrics["wins_vs_baseline"] = wins / max(len(seeds), 1)
    base_metrics = {f"order_placement.{b}": _agg(r) for b, r in base_rows.items()}
    rep = EvaluationReport(algo_id=strategy, protocol="ExecutionStrategy", metrics=metrics, baselines=base_metrics, objective="shortfall_bps", higher_is_better=False,
                           settings={"n_orders_per_seed": n_orders, "seeds": list(seeds), "duration_s": duration_s, "step_s": step_s, "params": dict(params or {}), "market": "quantstack 合成多场所模拟器（参考路由器）"})
    b0 = rep.baseline_primary()
    rep.passed = bool(np.isfinite(metrics["shortfall_bps"]) and b0 is not None and metrics["shortfall_bps"] <= b0 + tolerance_bps and metrics["fill_ratio"] >= 0.95)
    rep.verdict = (f"实现偏差 {metrics['shortfall_bps']:.2f} bps vs {list(base_metrics)[0]} {b0:.2f} bps（逐种子胜率 {metrics['wins_vs_baseline']:.0%}）；"
                   f"成交率 {metrics['fill_ratio']:.3f}，被动成交占比 {metrics['maker_ratio']:.2f}。")
    rep.notes.append(f"通过条件：平均实现偏差 ≤ 第一基线 + {tolerance_bps} bps 且成交率 ≥ 0.95。单路径噪声可达 20 bps，种子数越多结论越可靠。")
    rep.wall_s = time.time() - t0
    return rep


# ============================================================================ 执行规划（选择器）
def evaluate_selector(selector_builder: Callable[[], Any], panel, rounds: int = 6, orders_per_round: int = 5, seeds: Sequence[int] = (1, 2), duration_s: float = 600.0,
                      tolerance_bps: float = 1.0, step_s: float = 1.0) -> EvaluationReport:
    """连续 rounds 轮母单，选择器在线学习；比较后半程（学习期）平均实现偏差。通过条件：后半程 ≤ 基线后半程 + tolerance_bps。"""
    from .components.execution import AutoExecutionFactory

    t0 = time.time()
    mine_all, mine_late, base_all, base_late = [], [], [], []
    arm_counts: Dict[str, int] = {}
    def _episode(sd: int, sel):
        """连续 rounds 轮母单；sel 为 None 时走 quantstack 规则基线。返回 (全程行, 后半程行)。"""
        pipe = _sim_pipeline(panel, sd, step_s=step_s)
        pipe.execution_factory = AutoExecutionFactory(base=pipe.execution_factory)
        if sel is not None:
            sel.attach(pipe)
        all_rows, late_rows = [], []
        for r in range(rounds):
            orders = make_orders(pipe, orders_per_round, sd * 1000 + r, duration_s=duration_s)
            rows = _run_orders(pipe, orders)
            all_rows += rows
            if r >= rounds // 2:
                late_rows += rows
        return all_rows, late_rows

    for sd in seeds:
        sel = selector_builder()
        a, l = _episode(sd, sel)
        mine_all += a
        mine_late += l
        for arm, n in sel.describe().get("n_per_arm", {}).items():
            arm_counts[arm] = arm_counts.get(arm, 0) + int(n)
        a, l = _cached(("selector", _panel_key(panel), sd, rounds, orders_per_round, duration_s, step_s), lambda: _episode(sd, None))
        base_all += list(a)
        base_late += list(l)
    metrics = _agg(mine_all)
    metrics["learned_rounds_shortfall_bps"] = _agg(mine_late)["shortfall_bps"]
    metrics["n_seeds"] = float(len(seeds))
    bm = _agg(base_all)
    bm["learned_rounds_shortfall_bps"] = _agg(base_late)["shortfall_bps"]
    bm["n_seeds"] = float(len(seeds))
    rep = EvaluationReport(algo_id="selector", protocol="StrategySelector", metrics=metrics, baselines={"reference.choose_strategy": bm}, objective="learned_rounds_shortfall_bps",
                           higher_is_better=False, settings={"rounds": rounds, "orders_per_round": orders_per_round, "seeds": list(seeds), "duration_s": duration_s, "step_s": step_s, "arm_counts": arm_counts})
    rep.passed = bool(np.isfinite(metrics["learned_rounds_shortfall_bps"]) and metrics["learned_rounds_shortfall_bps"] <= bm["learned_rounds_shortfall_bps"] + tolerance_bps)
    rep.verdict = (f"后半程实现偏差 {metrics['learned_rounds_shortfall_bps']:.2f} bps vs 规则基线 {bm['learned_rounds_shortfall_bps']:.2f} bps；"
                   f"全程 {metrics['shortfall_bps']:.2f} vs {bm['shortfall_bps']:.2f}；各臂被选次数 {arm_counts}。")
    rep.notes.append(f"通过条件：后半程平均实现偏差 ≤ 基线 + {tolerance_bps} bps。前 warmup 轮沿用基线规则，因此全程差异主要来自后半程。")
    rep.wall_s = time.time() - t0
    return rep


# ============================================================================ 路由层
def evaluate_router(router_builder: Callable[[Any, Sequence[str]], Any], panel, n_orders: int = 5, seeds: Sequence[int] = (1, 2, 3), duration_s: float = 900.0,
                    tolerance_bps: float = 1.0, step_s: float = 1.0, mc_samples: int = 1000) -> EvaluationReport:
    """叠加层 vs 底层路由器（latency_routing），同一母单与种子成对比较。通过条件：实现偏差 ≤ 基线 + tolerance_bps 且成交率不低于基线 − 0.02。"""
    t0 = time.time()
    rows, base_rows = [], []
    wins = 0
    n_rebalanced = n_rewards = 0
    rcfg = {"mc_samples": int(mc_samples)}
    for sd in seeds:
        pipe = _sim_pipeline(panel, sd, kinds=ROUTER_SIM_KINDS, step_s=step_s, router_cfg=rcfg)
        venues = [q.name for q in pipe.feed.venues]
        pipe.router = router_builder(pipe.router, venues)
        orders = make_orders(pipe, n_orders, sd, duration_s=duration_s, adv_frac=(0.004, 0.015))
        mine = _run_orders(pipe, orders, {o.order_id: {"strategy": "almgren_chriss"} for o in orders})
        rows += mine
        d = pipe.router.describe() if hasattr(pipe.router, "describe") else {}
        n_rebalanced += int(d.get("n_rebalanced", 0))
        n_rewards += int(d.get("n_rewards", 0))
        def _base():
            pb = _sim_pipeline(panel, sd, kinds=ROUTER_SIM_KINDS, step_s=step_s, router_cfg=rcfg)
            ob = make_orders(pb, n_orders, sd, duration_s=duration_s, adv_frac=(0.004, 0.015))
            return _run_orders(pb, ob, {o.order_id: {"strategy": "almgren_chriss"} for o in ob})
        base = list(_cached(("router", _panel_key(panel), sd, n_orders, duration_s, step_s, int(mc_samples)), _base))
        base_rows += base
        if mine and base and _agg(mine)["shortfall_bps"] <= _agg(base)["shortfall_bps"]:
            wins += 1
    metrics = _agg(rows)
    metrics.update({"n_seeds": float(len(seeds)), "wins_vs_baseline": wins / max(len(seeds), 1), "n_rebalanced": float(n_rebalanced), "n_rewards": float(n_rewards)})
    bm = _agg(base_rows)
    bm["n_seeds"] = float(len(seeds))
    rep = EvaluationReport(algo_id="router", protocol="Router", metrics=metrics, baselines={"latency_routing.make_router": bm}, objective="shortfall_bps", higher_is_better=False,
                           settings={"n_orders_per_seed": n_orders, "seeds": list(seeds), "duration_s": duration_s, "step_s": step_s, "mc_samples": mc_samples, "execution": "almgren_chriss"})
    rep.passed = bool(np.isfinite(metrics["shortfall_bps"]) and metrics["shortfall_bps"] <= bm["shortfall_bps"] + tolerance_bps and metrics["fill_ratio"] >= bm["fill_ratio"] - 0.02)
    rep.verdict = (f"实现偏差 {metrics['shortfall_bps']:.2f} bps vs 底层路由 {bm['shortfall_bps']:.2f} bps（逐种子胜率 {metrics['wins_vs_baseline']:.0%}）；"
                   f"成交率 {metrics['fill_ratio']:.3f} vs {bm['fill_ratio']:.3f}；重分配切片 {n_rebalanced} 个，奖励样本 {n_rewards} 个。")
    rep.notes.append(f"通过条件：实现偏差 ≤ 基线 + {tolerance_bps} bps 且成交率 ≥ 基线 − 0.02。")
    rep.wall_s = time.time() - t0
    return rep


# ============================================================================ 统一入口
def evaluate(spec: FamilySpec, params: Dict[str, Any], panel, budget: str | Dict[str, Any] = "quick", seed: int = 0, train_ratio: float = 0.7) -> EvaluationReport:
    """按模板族协议分派到具体评估函数。"""
    b = _budget(budget)
    p = {**spec.default_params, **(params or {})}
    if spec.protocol == "AlphaModel":
        rep = evaluate_alpha(spec.build(p), panel, train_ratio=train_ratio, stride=b["alpha_stride"])
    elif spec.protocol == "PortfolioConstructor":
        rep = evaluate_portfolio(spec.build(p), panel, train_ratio=train_ratio, rebalance_every=b["portfolio_every"], extra_panels=int(b.get("portfolio_extra_panels", 0)), seed=seed,
                                 controlled_ic=float(b.get("portfolio_controlled_ic", 0.0)))
    elif spec.protocol == "ExecutionStrategy":
        seeds = [seed * 100 + k + 1 for k in range(b["exec_seeds"])]
        rep = evaluate_execution(lambda base: spec.build(p, base=base), panel, strategy="kalman_pid", params=p, n_orders=b["exec_orders"], seeds=seeds, duration_s=b["exec_duration_s"], step_s=b["step_s"])
    elif spec.protocol == "StrategySelector":
        seeds = [seed * 100 + k + 1 for k in range(b["selector_seeds"])]
        rep = evaluate_selector(lambda: spec.build({**p, "seed": seed}), panel, rounds=b["selector_rounds"], orders_per_round=b["selector_orders"], seeds=seeds, duration_s=b["exec_duration_s"], step_s=b["step_s"])
    elif spec.protocol == "Router":
        seeds = [seed * 100 + k + 1 for k in range(b["router_seeds"])]
        rep = evaluate_router(lambda base, venues: spec.build({**p, "seed": seed}, base=base, venues=venues), panel, n_orders=b["router_orders"], seeds=seeds, duration_s=b["router_duration_s"], step_s=b["step_s"], mc_samples=b["mc_samples"])
    else:
        raise NotImplementedError(f"协议 {spec.protocol} 暂无评估台")
    rep.algo_id = spec.default_algo_id
    rep.settings["budget"] = budget if isinstance(budget, str) else "custom"
    rep.settings["seed"] = seed
    return rep


def signed_score(rep: EvaluationReport) -> float:
    """把主目标转成'越大越好'的分数（供搜索排序）。"""
    v = rep.primary()
    if v is None or not np.isfinite(v):
        return float("-inf")
    return float(v) if rep.higher_is_better else -float(v)
