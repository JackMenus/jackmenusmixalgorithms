"""执行层新算法：``KalmanPIDExecution``（实现 quantstack ``ExecutionStrategy``）与把它并入挂单库工厂的 ``AutoExecutionFactory``。

组合关系
  * 继承 order_placement ``QuantstackExecution``：复用其"计划曲线 → 落后追单（IOC）/ 近端被动挂单（POST_ONLY）→ 扫尾"
    的全部机制与在途簿记（切片撤单、部分释放等 §8 约定）；
  * 计划曲线取 order_placement 的 Almgren–Chriss 轨迹（``AlmgrenChrissExec.plan``，urgency → 风险厌恶 λ，alpha → 漂移）；
  * crossquant ``KalmanTrendEstimator`` 在线滤波中间价，得到平滑的公允价（level）与斜率，两种用法由 ``mode`` 选择：
      - reversion（默认）：中间价相对公允价的偏离视为暂时性（自身冲击、报价噪声），偏贵 → 放慢并后撤一跳，便宜 → 加速并改善一跳；
      - momentum：按斜率外推 fair_horizon 秒后的水平，预测不利 → 提前并改善一跳，预测有利 → 后撤；
    统一用 ``pred_move_bps``（按 side 方向为正 = 不利 / 应加速）驱动节奏与挂价；
  * crossquant ``PIDParticipationController`` / ``LQRTrackingController`` 把"计划完成比例 − 实际完成比例"的误差
    转成计划量的修正（反馈控制），避免 AC 开环轨迹在被动成交偏慢时持续落后。

所有参数都可通过 ``params`` 覆盖（见 ``KP_DEFAULTS``）。
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from .._paths import require

US = 1_000_000

KP_DEFAULTS: Dict[str, object] = {
    "schedule": "ac",              # ac（Almgren–Chriss）/ twap
    "mode": "reversion",           # Kalman 信号用法：reversion（偏离公允价 → 均值回复，贵了放慢、便宜加速）/ momentum（斜率外推 → 追趋势）
    "controller": "pid",           # pid / lqr
    "kp": 0.5,                     # PID 比例增益（误差 = 计划比例 − 实际比例）
    "ki": 0.05,                    # PID 积分增益（每秒）
    "kd": 0.0,
    "ctrl_out_max": 0.2,           # 控制量上限（占母单的比例）
    "lqr_q": 1.0,
    "lqr_r": 0.1,
    "fair_horizon_s": 30.0,        # Kalman 公允价外推时长
    "kf_level_mult": 1.0,          # 水平过程噪声倍数（基准 = σ_price²/秒）
    "kf_obs_mult": 10.0,           # 观测噪声倍数（基准 = (价差/4)²）；越大水平越平滑，reversion 模式下偏离信号越"慢"
    "kf_slope_ratio": 1e-4,        # 斜率过程噪声 / 水平过程噪声
    "drift_threshold_bps": 1.0,    # 预测偏移超过该值才调节节奏
    "drift_gain": 1.0,             # 每单位（倍阈值）预测偏移提前/推后的切片数
    "drift_max_slices": 3.0,
    "improve_threshold_bps": 1.5,  # 预测不利偏移超过该值 → 挂价改善一跳
    "backoff": 1.0,                # 预测有利时是否后撤一跳（1 = 是）
}


def _f(p: Dict, key: str) -> float:
    return float(p.get(key, KP_DEFAULTS[key]))


def _base_classes():
    require("order_placement", "quantstack")
    from order_placement.adapters.quantstack_exec import AlmgrenChrissExec, QuantstackExecution

    return QuantstackExecution, AlmgrenChrissExec


_Base, _AC = _base_classes()


class KalmanPIDExecution(_Base):
    """Almgren–Chriss 计划 × Kalman 公允价外推 × PID/LQR 进度反馈 的执行策略。"""

    name = "kalman_pid"

    def __init__(self, order, params: Optional[Dict] = None):
        merged = {**KP_DEFAULTS, **(params or {})}
        super().__init__(order, merged)
        require("crossquant")
        from crossquant import LQRTrackingController, PIDParticipationController

        out_max = _f(self.p, "ctrl_out_max")
        if str(self.p.get("controller", "pid")).lower() == "lqr":
            self.controller = LQRTrackingController(q=_f(self.p, "lqr_q"), r=_f(self.p, "lqr_r"), dt_ref_sec=1.0, out_max=out_max)
        else:
            self.controller = PIDParticipationController(kp=_f(self.p, "kp"), ki=_f(self.p, "ki"), kd=_f(self.p, "kd"), out_min=-out_max, out_max=out_max)
        self.estimator = None
        self._ctrl_ts: Optional[int] = None
        self.ctrl_adj = 0.0          # 控制器给出的计划修正（占母单比例）
        self.pred_move_bps = 0.0     # Kalman 预测的公允价偏移（按 side 方向为正 = 不利）
        self.kf_level = float("nan")
        self.n_obs = 0

    # ---------------------------------------------------------------- 计划曲线
    def plan(self, snap):
        if str(self.p.get("schedule", "ac")).lower() == "twap":
            return super().plan(snap)
        return _AC.plan(self, snap)

    # ---------------------------------------------------------------- Kalman 公允价
    def _observe(self, snap) -> None:
        mid = snap.mid
        if mid <= 0:
            return
        if self.estimator is None:
            from crossquant import KalmanTrendEstimator

            sigma_s = self.sigma_price_per_sqrt_s(snap)              # 价格单位 / √秒
            level_var = max(sigma_s ** 2, 1e-12) * _f(self.p, "kf_level_mult")
            bid, ask = snap.nbbo()
            obs_var = max(((ask - bid) / 4.0) ** 2, 1e-12) * _f(self.p, "kf_obs_mult")
            self.estimator = KalmanTrendEstimator(level_var=level_var, slope_var=level_var * _f(self.p, "kf_slope_ratio"), obs_var=obs_var)
        self.kf_level = self.estimator.update(mid, snap.ts_us)
        self.n_obs += 1
        if self.n_obs < 3:
            self.pred_move_bps = 0.0
        elif str(self.p.get("mode", "reversion")).lower() == "momentum":
            # 斜率外推：fair_horizon 秒后的水平相对当前水平的变化，按 side 方向为正 = 不利（价格将朝我方向走，等待更贵）
            pred = self.estimator.predict(int(_f(self.p, "fair_horizon_s") * US))
            self.pred_move_bps = float(self.order.side * (pred - self.kf_level) / mid * 1e4)
        else:
            # 均值回复：当前中间价相对平滑公允价的偏离；买单时 mid 高于公允 = 暂时偏贵 → 预期回落（有利）→ 放慢；低于公允 = 便宜 → 加速
            self.pred_move_bps = float(-self.order.side * (mid - self.kf_level) / mid * 1e4)

    # ---------------------------------------------------------------- 反馈控制 + 漂移调节
    def urgency_target(self, snap) -> float:
        base = self.target(snap)
        if str(self.p.get("schedule", "ac")).lower() == "twap" and self.order.alpha_bps > float(self.p.get("alpha_urgency_bps", 2.0)):
            base += self.slice_size()
        q = max(self.order.quantity, 1)
        dt_us = int(snap.ts_us - self._ctrl_ts) if self._ctrl_ts is not None else int(float(self.p.get("refresh_s", 5.0)) * US)
        self._ctrl_ts = snap.ts_us
        error = (base - self.filled) / q
        self.ctrl_adj = float(self.controller.step(error, max(dt_us, 1)))
        tgt = base + self.ctrl_adj * q
        thr = _f(self.p, "drift_threshold_bps")
        if abs(self.pred_move_bps) > thr:
            k = float(np.clip(abs(self.pred_move_bps) / thr, 0.0, _f(self.p, "drift_max_slices")))
            shift = _f(self.p, "drift_gain") * k * self.slice_size()
            tgt += shift if self.pred_move_bps > 0 else -shift
        return float(np.clip(tgt, 0.0, self.order.quantity))

    def passive_price(self, snap, near, sl) -> float:
        px = super().passive_price(snap, near, sl)
        o = self.order
        tick = snap.extra.get("tick_size", 0.01)
        bid, ask = snap.nbbo()
        far = ask if o.side > 0 else bid
        thr = _f(self.p, "improve_threshold_bps")
        if self.pred_move_bps > thr:
            cand = near + o.side * tick
            if (far - cand) * o.side > 1e-9:          # 不穿越对手价
                px = max(px, cand) if o.side > 0 else min(px, cand)
        elif self.pred_move_bps < -thr and _f(self.p, "backoff") > 0:
            px = near - o.side * tick
        return float(px)

    def on_market(self, snapshot):
        self._observe(snapshot)
        return super().on_market(snapshot)

    def state(self) -> Dict[str, float]:
        s = super().state()
        s.update({"pred_move_bps": float(self.pred_move_bps), "ctrl_adj": float(self.ctrl_adj), "kf_level": float(self.kf_level) if np.isfinite(self.kf_level) else 0.0})
        return s


STRATEGY_DOCS: Dict[str, str] = {
    "kalman_pid": "Almgren–Chriss 轨迹 + Kalman 公允价偏离/外推调节节奏与挂价（mode=reversion|momentum）+ PID(LQR) 进度反馈；params: mode, controller, kp, ki, kf_obs_mult, drift_gain, improve_threshold_bps。",
}
AUTO_STRATEGIES = {"kalman_pid": KalmanPIDExecution}


class AutoExecutionFactory:
    """组合式执行策略工厂：自研策略 + 挂单库（order_placement）全部策略，实现 ``ExecutionStrategyFactory``。

    参数
      base     底层工厂（默认 ``order_placement.make_execution_factory(**cfg)``）
      params   自研策略的默认参数（``create`` 时可再覆盖）
      cfg      透传给底层工厂（trading_seconds_per_day / bucket_s / refresh_s …）
    """

    def __init__(self, base=None, params: Optional[Dict] = None, **cfg):
        if base is None:
            require("order_placement")
            from order_placement import make_execution_factory

            base = make_execution_factory(**cfg)
        self.base = base
        self.cfg = dict(cfg)
        self.params = dict(params or {})

    def create(self, name: str, order, **params):
        key = (name or order.strategy_hint or "twap").lower()
        if key in AUTO_STRATEGIES:
            return AUTO_STRATEGIES[key](order, {**self.cfg, **self.params, **params})
        return self.base.create(name, order, **params)

    def available(self) -> Dict[str, str]:
        return {**self.base.available(), **STRATEGY_DOCS}

    def describe(self) -> Dict[str, object]:
        return {"library": "autoalgo", "family": "execution.kalman_pid", "params": dict(self.params), "base": type(self.base).__name__, "strategies": sorted(self.available())}
