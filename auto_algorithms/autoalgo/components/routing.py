"""路由层新算法：``BayesianVenueRouter`` —— 在任意底层 Router（默认 latency_routing 适配器）之上叠加
场所级的高斯 Thompson 采样与 BOCPD 延迟退化监测。

组合关系
  * 底层 ``Router``（latency_routing 的 Cont-Kukanov 拆分 + 同步到达，或 quantstack 参考路由）给出基准分配；
  * crossquant ``ThompsonGaussian``：每个场所一条臂，奖励 = −(实现滑点 bps + 费用 bps)，
    对吃单切片按后验采样把基准分配向"近期更便宜"的场所倾斜（mix 控制倾斜力度）；
  * crossquant ``bocpd``：对每个场所最近的成交回报延迟（log）做在线变点检测，
    变点概率高且延迟上行 → 判定退化，吃单份额按 degrade_penalty 压缩，被动单整单改投别处。

实现 quantstack ``Router`` 协议：``route(slice, snapshot)``、``on_event(event)``、``venue_stats()``。
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._paths import available, require


class BayesianVenueRouter:
    """Thompson 采样 + BOCPD 退化监测 的场所路由叠加层。

    参数
      base              底层路由器（实现 Router 协议）
      venues            场所列表
      mix               吃单分配向 Thompson 采样权重倾斜的比例 ∈ [0,1]
      obs_var_bps2      奖励观测方差（bps²）；prior_var：先验方差
      min_share / max_share  单场所份额上下限（相对切片总量）
      latency_window    每场所保留的延迟样本数（用于 BOCPD）
      degrade_hazard    BOCPD 先验变点概率；degrade_threshold：判定退化的变点概率阈值
      degrade_penalty   退化场所吃单份额乘 (1 − penalty)
      reject_penalty_bps 拒单计入的等价成本
    """

    def __init__(self, base, venues: Sequence[str], mix: float = 0.5, obs_var_bps2: float = 4.0, prior_var: float = 25.0, min_share: float = 0.05,
                 max_share: float = 0.8, latency_window: int = 40, degrade_hazard: float = 0.05, degrade_threshold: float = 0.5, degrade_penalty: float = 0.6,
                 reject_penalty_bps: float = 3.0, seed: int = 0):
        require("crossquant", "quantstack")
        from crossquant.control.bandits import ThompsonGaussian

        self.base = base
        self.venues = list(venues)
        self.index = {v: i for i, v in enumerate(self.venues)}
        self.mix = float(np.clip(mix, 0.0, 1.0))
        self.obs_var_bps2, self.prior_var = float(obs_var_bps2), float(prior_var)
        self.ts = ThompsonGaussian(n_arms=len(self.venues), obs_var=self.obs_var_bps2, prior_var=self.prior_var, seed=seed)
        self.min_share, self.max_share = float(min_share), float(max_share)
        self.latency_window, self.degrade_hazard, self.degrade_threshold, self.degrade_penalty = int(latency_window), float(degrade_hazard), float(degrade_threshold), float(degrade_penalty)
        self.reject_penalty_bps = float(reject_penalty_bps)
        self.seed = int(seed)
        self._ref: Dict[Tuple[str, str], float] = {}
        self._ref_order: Deque[Tuple[str, str]] = deque()
        self._lat: Dict[str, Deque[float]] = {v: deque(maxlen=self.latency_window) for v in self.venues}
        self.degraded: Dict[str, float] = {v: 0.0 for v in self.venues}
        self.n_routes = 0
        self.n_rebalanced = 0
        self.n_rewards = 0

    # ---------------------------------------------------------------- 路由
    def _remember(self, slice_id: str, venue: str, mid: float) -> None:
        key = (slice_id, venue)
        if key not in self._ref:
            self._ref_order.append(key)
        self._ref[key] = mid
        while len(self._ref_order) > 20000:
            self._ref.pop(self._ref_order.popleft(), None)

    def _thompson_weights(self, venues: List[str]) -> np.ndarray:
        m, v = self.ts.posterior()
        samp = np.array([self.ts.rng.normal(m[self.index[x]], math.sqrt(v[self.index[x]])) for x in venues])
        samp = samp - samp.min() + 1.0                     # 平移到正数，保持序关系
        pen = np.array([1.0 - self.degrade_penalty * (self.degraded.get(x, 0.0) > self.degrade_threshold) for x in venues])
        w = samp * pen
        return w / w.sum() if w.sum() > 0 else np.full(len(venues), 1.0 / len(venues))

    def route(self, order_slice, snapshot):
        from quantstack.contracts import RoutingDecision, VenueOrder

        dec = self.base.route(order_slice, snapshot)
        self.n_routes += 1
        mid = snapshot.mid
        for vo in dec.venue_orders:
            self._remember(order_slice.slice_id, vo.venue, mid)
        if not dec.venue_orders or mid <= 0:
            return dec
        aggressive = order_slice.order_type in ("IOC", "MARKET")
        if aggressive and len(dec.venue_orders) >= 2 and self.mix > 0:
            venues = [vo.venue for vo in dec.venue_orders]
            base_q = np.array([vo.quantity for vo in dec.venue_orders], float)
            total = base_q.sum()
            if total <= 0:
                return dec
            tw = self._thompson_weights(venues)
            blend = (1.0 - self.mix) * base_q / total + self.mix * tw
            blend = np.clip(blend, self.min_share, self.max_share)
            blend = blend / blend.sum()
            new_q = np.floor(blend * total).astype(int)
            # 不超过该场所可见量的 1.3 倍（与底层 oversize 口径一致）
            for i, vo in enumerate(dec.venue_orders):
                q = snapshot.quote(vo.venue)
                cap = int(max(vo.quantity, 1.3 * q.far_side(order_slice.side)[1])) if q is not None else vo.quantity
                new_q[i] = min(new_q[i], cap)
            diff = int(total - new_q.sum())
            if diff > 0:  # 余量补给采样权重最高者
                new_q[int(np.argmax(tw))] += diff
            orders = []
            for vo, q in zip(dec.venue_orders, new_q):
                if q <= 0:
                    continue
                d = vo.to_dict()
                d.pop("schema_version", None)
                d["quantity"] = int(q)
                d["route_reason"] = (d.get("route_reason") or "") + "|thompson"
                orders.append(VenueOrder(**d))
            self.n_rebalanced += 1
            diag = dict(dec.diagnostics)
            diag.update({"bandit_mix": self.mix, "n_degraded": float(sum(p > self.degrade_threshold for p in self.degraded.values()))})
            return RoutingDecision(slice_id=dec.slice_id, venue_orders=orders, expected_cost_bps=dec.expected_cost_bps,
                                   expected_fill_ratio=dec.expected_fill_ratio, diagnostics=diag, rationale=(dec.rationale or "") + "+bayesian_venue")
        if aggressive and len(dec.venue_orders) == 1 and self.mix > 0:
            # 单场所吃单：在"含费价格不劣于所选场所、可见量足够"的场所里按 Thompson 采样换场所（以 mix 为概率）
            vo = dec.venue_orders[0]
            q0 = snapshot.quote(vo.venue)
            if q0 is not None and self.ts.rng.random() < self.mix:
                side = order_slice.side
                px0, _ = q0.far_side(side)
                eff0 = px0 + side * q0.taker_fee
                cands = [vo.venue]
                sizes = {vo.venue: q0.far_side(side)[1]}
                for q in snapshot.quotes:
                    if q.venue == vo.venue or q.venue not in self.index:
                        continue
                    px, sz = q.far_side(side)
                    if (px + side * q.taker_fee - eff0) * side <= 1e-9 and sz >= 0.5 * vo.quantity:
                        cands.append(q.venue)
                        sizes[q.venue] = sz
                if len(cands) >= 2:
                    tw = self._thompson_weights(cands)
                    best = cands[int(np.argmax(tw))]
                    if best != vo.venue:
                        q_new = int(min(vo.quantity, max(1.3 * sizes[best], 1)))
                        orders = []
                        d = vo.to_dict()
                        d.pop("schema_version", None)
                        d.update(venue=best, quantity=q_new, venue_order_id=f"{order_slice.slice_id}-ts1", route_reason=(d.get("route_reason") or "") + "|thompson_switch")
                        orders.append(VenueOrder(**d))
                        if vo.quantity - q_new > 0:
                            d2 = vo.to_dict()
                            d2.pop("schema_version", None)
                            d2.update(quantity=int(vo.quantity - q_new), venue_order_id=f"{order_slice.slice_id}-ts2")
                            orders.append(VenueOrder(**d2))
                        self._remember(order_slice.slice_id, best, mid)
                        self.n_rebalanced += 1
                        diag = dict(dec.diagnostics)
                        diag.update({"bandit_mix": self.mix, "thompson_switch": 1.0})
                        return RoutingDecision(slice_id=dec.slice_id, venue_orders=orders, expected_cost_bps=dec.expected_cost_bps, expected_fill_ratio=dec.expected_fill_ratio,
                                               diagnostics=diag, rationale=(dec.rationale or "") + "+thompson_switch")
        if not aggressive and len(dec.venue_orders) == 1:
            vo = dec.venue_orders[0]
            if self.degraded.get(vo.venue, 0.0) > self.degrade_threshold:
                m, _ = self.ts.posterior()
                cands = [q.venue for q in snapshot.quotes if q.venue in self.index and self.degraded.get(q.venue, 0.0) <= self.degrade_threshold]
                if cands:
                    best = max(cands, key=lambda x: m[self.index[x]])
                    d = vo.to_dict()
                    d.pop("schema_version", None)
                    d["venue"] = best
                    d["route_reason"] = (d.get("route_reason") or "") + "|degraded_reroute"
                    self._remember(order_slice.slice_id, best, mid)
                    self.n_rebalanced += 1
                    return RoutingDecision(slice_id=dec.slice_id, venue_orders=[VenueOrder(**d)], expected_cost_bps=dec.expected_cost_bps,
                                           expected_fill_ratio=dec.expected_fill_ratio, diagnostics=dict(dec.diagnostics), rationale=(dec.rationale or "") + "+degraded_reroute")
        return dec

    # ---------------------------------------------------------------- 学习
    def _update_degraded(self, venue: str) -> None:
        lat = self._lat[venue]
        if len(lat) < 10 or not available("crossquant"):
            return
        from crossquant.control.changepoint import bocpd

        x = np.log(np.asarray(lat, float) + 1.0)
        x = (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
        res = bocpd(x, hazard=self.degrade_hazard, mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=2.0, max_run=len(x), window=5)
        p = float(res["prob_change"][-1])
        rising = float(np.mean(x[-3:])) > float(np.mean(x[:-3])) if len(x) > 6 else False
        self.degraded[venue] = p if rising else 0.0

    def on_event(self, event) -> None:
        self.base.on_event(event)
        venue = event.venue or (event.fill.venue if event.fill is not None else "")
        if venue not in self.index:
            return
        arm = self.index[venue]
        if event.kind in ("fill", "partial") and event.fill is not None:
            f = event.fill
            ref = self._ref.get((f.slice_id, venue))
            if ref and ref > 0:
                slip = f.side * (f.price - ref) / ref * 1e4 + (f.fee / max(f.quantity, 1)) / ref * 1e4
                self.ts.update(arm, -float(slip))
                self.n_rewards += 1
            if f.latency_us and f.latency_us > 0:
                self._lat[venue].append(float(f.latency_us))
                self._update_degraded(venue)
        elif event.kind == "reject":
            self.ts.update(arm, -self.reject_penalty_bps)
            self.n_rewards += 1

    def venue_stats(self):
        return self.base.venue_stats()

    def describe(self) -> Dict[str, object]:
        m, v = self.ts.posterior()
        return {
            "library": "autoalgo", "family": "router.bayesian_venue", "venues": list(self.venues), "mix": self.mix, "obs_var_bps2": self.obs_var_bps2,
            "prior_var": self.prior_var, "degrade_threshold": self.degrade_threshold, "degrade_penalty": self.degrade_penalty,
            "posterior_mean_reward_bps": {x: float(m[i]) for i, x in enumerate(self.venues)},
            "posterior_var": {x: float(v[i]) for i, x in enumerate(self.venues)},
            "n_obs": {x: float(self.ts.n[i]) for i, x in enumerate(self.venues)},
            "degraded_prob": dict(self.degraded), "n_routes": self.n_routes, "n_rebalanced": self.n_rebalanced, "n_rewards": self.n_rewards,
            "base": type(self.base).__name__,
        }
