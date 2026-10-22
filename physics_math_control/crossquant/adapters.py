"""
与全栈交易系统（quantstack）契约的适配器
========================================

集成层（/mnt/project-files/trading_system/quantstack/contracts/protocols.py）预留了三个薄协议，
本模块给出可直接登记到 ``quantstack/core/registry.py`` 的实现（结构化子类型，无需导入 quantstack）：

- ``StateEstimator``：``update(observation: float, ts_us: int) -> float``、``predict(horizon_us: int) -> float``
    * :class:`KalmanTrendEstimator`  —— 局部线性趋势 Kalman（价格/公允价/延迟的隐状态与外推）
    * :class:`OUSpreadEstimator`     —— 在线 OU（价差的均值回复预测）
- ``Controller``：``step(error: float, dt_us: int) -> float``
    * :class:`PIDParticipationController` —— PID 调节执行参与率
    * :class:`LQRTrackingController`      —— LQ 最优反馈（带控制代价的进度跟踪）
- ``ImpactModel``：``expected_cost_bps(quantity: int, snapshot, duration_us: int) -> float``
    * :class:`SquareRootImpactModel`   —— 平方根冲击律 + 半价差
    * :class:`AlmgrenChrissImpactModel`—— Almgren–Chriss 期望成本（临时 + 永久冲击 + 价差）

约定：时间为微秒整数 ``ts_us``；成本用基点 (bps)；随机算法接受 ``seed``。
``snapshot`` 只需提供 ``mid``、``spread_bps``、``adv``、``volatility`` 属性（与 ``quantstack.contracts.MarketSnapshot`` 一致）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .control.execution import almgren_chriss, square_root_impact
from .control.kalman import KalmanFilter
from .control.lqr import lqr_discrete
from .control.pid import PID
from .control.rls import RecursiveLeastSquares

US_PER_SEC = 1_000_000
TRADING_DAY_US = int(6.5 * 3600 * US_PER_SEC)


# --------------------------------------------------------------------------- StateEstimator
class KalmanTrendEstimator:
    """局部线性趋势模型：state = (level, slope/秒)。

    参数
    ----
    level_var : 水平过程噪声方差（每秒）
    slope_var : 斜率过程噪声方差（每秒）
    obs_var   : 观测噪声方差
    """

    def __init__(self, level_var: float = 1e-4, slope_var: float = 1e-8, obs_var: float = 1e-2):
        self.level_var, self.slope_var, self.obs_var = level_var, slope_var, obs_var
        self._kf: Optional[KalmanFilter] = None
        self._last_ts: Optional[int] = None

    def update(self, observation: float, ts_us: int) -> float:
        if self._kf is None:
            self._kf = KalmanFilter(F=np.eye(2), Q=np.zeros((2, 2)), R=[[self.obs_var]], H=[[1.0, 0.0]],
                                    x0=[observation, 0.0], P0=np.diag([self.obs_var, 1e-6]))
            self._last_ts = ts_us
            return float(observation)
        dt = max((ts_us - self._last_ts) / US_PER_SEC, 1e-9)
        self._kf.F = np.array([[1.0, dt], [0.0, 1.0]])
        self._kf.Q = np.array([[self.level_var * dt + self.slope_var * dt ** 3 / 3, self.slope_var * dt ** 2 / 2],
                               [self.slope_var * dt ** 2 / 2, self.slope_var * dt]])
        self._kf.predict()
        self._kf.update(observation)
        self._last_ts = ts_us
        return float(self._kf.x[0])

    def predict(self, horizon_us: int) -> float:
        if self._kf is None:
            return float("nan")
        h = horizon_us / US_PER_SEC
        return float(self._kf.x[0] + self._kf.x[1] * h)

    @property
    def state(self) -> dict:
        return dict(level=float(self._kf.x[0]), slope_per_sec=float(self._kf.x[1])) if self._kf is not None else {}


class OUSpreadEstimator:
    """在线 OU 价差估计：用带遗忘因子的 RLS 拟合 AR(1)，predict 给出均值回复后的期望水平。"""

    def __init__(self, forgetting: float = 0.995, dt_sec: float | None = None):
        self._rls = RecursiveLeastSquares(2, forgetting)
        self._last_x: Optional[float] = None
        self._last_ts: Optional[int] = None
        self._dt = dt_sec
        self.n = 0

    def update(self, observation: float, ts_us: int) -> float:
        if self._last_x is not None:
            self._rls.update(np.array([self._last_x, 1.0]), observation)
            if self._dt is None and self._last_ts is not None and ts_us > self._last_ts:
                self._dt = (ts_us - self._last_ts) / US_PER_SEC
            self.n += 1
        self._last_x, self._last_ts = observation, ts_us
        return float(observation)

    @property
    def params(self) -> dict:
        b, a = self._rls.theta
        dt = self._dt or 1.0
        if not (0 < b < 1):
            return dict(theta=float(self._last_x or 0.0), kappa=0.0, b=float(b))
        return dict(theta=float(a / (1 - b)), kappa=float(-np.log(b) / dt), b=float(b))

    def predict(self, horizon_us: int) -> float:
        if self._last_x is None or self.n < 5:
            return float(self._last_x) if self._last_x is not None else float("nan")
        p = self.params
        h = horizon_us / US_PER_SEC
        return float(p["theta"] + (self._last_x - p["theta"]) * np.exp(-p["kappa"] * h))


# --------------------------------------------------------------------------- Controller
class PIDParticipationController:
    """error = 目标完成比例 − 实际完成比例，输出参与率调整量（限幅在 [out_min, out_max]）。"""

    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.0, out_min: float = -0.5, out_max: float = 0.5):
        self._pid = PID(kp, ki, kd, out_min=out_min, out_max=out_max)

    def step(self, error: float, dt_us: int) -> float:
        return self._pid.step(error, dt_us / US_PER_SEC)

    def reset(self):
        self._pid.reset()


class LQRTrackingController:
    """标量 LQ 反馈：状态 e_{t+1} = e_t − u_t·dt，代价 q e² + r u²，u = K e（K 为 Riccati 解）。"""

    def __init__(self, q: float = 1.0, r: float = 0.1, dt_ref_sec: float = 1.0, out_max: float = 1.0):
        self.K = float(lqr_discrete([[1.0]], [[-dt_ref_sec]], [[q]], [[r]])["K"][0, 0])
        self.out_max = out_max

    def step(self, error: float, dt_us: int) -> float:
        return float(np.clip(-self.K * error, -self.out_max, self.out_max))   # u = −K e（K<0 ⇒ 正误差推高参与率）


# --------------------------------------------------------------------------- ImpactModel
class SquareRootImpactModel:
    """期望成本(bps) = 半价差 + Y·σ_daily·sqrt(Q/ADV)·1e4。``horizon_volume=True`` 时用执行时段内的预期成交量代替 ADV。"""

    def __init__(self, Y: float = 0.5, horizon_volume: bool = False, default_sigma: float = 0.02, default_adv: float = 1e6):
        self.Y, self.horizon_volume = Y, horizon_volume
        self.default_sigma, self.default_adv = default_sigma, default_adv

    def expected_cost_bps(self, quantity: int, snapshot, duration_us: int) -> float:
        adv = getattr(snapshot, "adv", None) or self.default_adv
        sigma = getattr(snapshot, "volatility", None) or self.default_sigma
        half_spread = 0.5 * (getattr(snapshot, "spread_bps", 0.0) or 0.0)
        volume = adv * min(max(duration_us / TRADING_DAY_US, 1e-3), 1.0) if self.horizon_volume else adv
        return float(half_spread + 1e4 * square_root_impact(abs(quantity), volume, sigma, self.Y))


class AlmgrenChrissImpactModel:
    """Almgren–Chriss 期望实施缺口（bps）。

    默认参数映射：σ_price = volatility·mid（每天）；η = 半价差/ADV（以 ADV 速率交易每股付半个价差）；
    γ = perm_frac·η；ε = 半价差；λ 为风险厌恶；T = duration（天），N = n_slices。
    """

    def __init__(self, lam: float = 1e-6, n_slices: int = 10, perm_frac: float = 0.1, default_sigma: float = 0.02, default_adv: float = 1e6):
        self.lam, self.n_slices, self.perm_frac = lam, n_slices, perm_frac
        self.default_sigma, self.default_adv = default_sigma, default_adv

    def expected_cost_bps(self, quantity: int, snapshot, duration_us: int) -> float:
        mid = getattr(snapshot, "mid", None) or getattr(snapshot, "last_price", None) or 1.0
        adv = getattr(snapshot, "adv", None) or self.default_adv
        sigma = (getattr(snapshot, "volatility", None) or self.default_sigma) * mid
        half_spread_px = 0.5 * (getattr(snapshot, "spread_bps", 0.0) or 0.0) / 1e4 * mid
        T = max(duration_us / TRADING_DAY_US, 1e-3)
        eta = half_spread_px / adv if half_spread_px > 0 else 1e-6 * mid / adv
        gamma = self.perm_frac * eta
        X = abs(quantity)
        res = almgren_chriss(X, T, self.n_slices, sigma, eta, self.lam, gamma=gamma, epsilon=half_spread_px)
        return float(res["expected_cost"] / (X * mid) * 1e4) if X > 0 else 0.0


def registry_entries() -> dict:
    """供 quantstack 注册表使用：协议名 → {实现名: 类}。"""
    return {
        "StateEstimator": {"crossquant.KalmanTrendEstimator": KalmanTrendEstimator, "crossquant.OUSpreadEstimator": OUSpreadEstimator},
        "Controller": {"crossquant.PIDParticipationController": PIDParticipationController, "crossquant.LQRTrackingController": LQRTrackingController},
        "ImpactModel": {"crossquant.SquareRootImpactModel": SquareRootImpactModel, "crossquant.AlmgrenChrissImpactModel": AlmgrenChrissImpactModel},
    }
