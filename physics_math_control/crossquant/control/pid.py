"""
PID 控制器：执行进度与库存的反馈调节
=====================================

学科来源
--------
经典控制（Minorsky 1922 用于船舶自动驾驶；Ziegler–Nichols 1942 整定）。u = K_p e + K_i ∫e + K_d de/dt，
工业界 90% 以上的回路仍是 PID。

量化映射
--------
- **执行进度跟踪**：误差 e = 目标已完成比例（如 VWAP/TWAP 曲线）− 实际完成比例，
  输出参与率的增减，I 项消除持续落后，D 项抑制追价过猛。
- **做市库存控制**：e = 目标库存 − 当前库存，输出报价偏移（skew）。
- **延迟预算控制**：e = 目标延迟 − 实测延迟，调节重传/路由保守度。
抗积分饱和（anti-windup）与输出限幅是实盘必需。
"""
from __future__ import annotations

import numpy as np


class PID:
    """离散 PID（时间单位自定，``step(error, dt)``）。

    参数
    ----
    kp, ki, kd : 比例/积分/微分增益
    out_min, out_max : 输出限幅（也用于抗积分饱和）
    derivative_filter : 微分项一阶低通系数 (0,1]，1 表示不过滑
    """

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, out_min: float = -np.inf,
                 out_max: float = np.inf, derivative_filter: float = 1.0, setpoint: float = 0.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.alpha = derivative_filter
        self.setpoint = setpoint
        self.reset()

    def reset(self):
        self.integral = 0.0; self.prev_error = None; self.d_filt = 0.0; self.last_output = 0.0

    def step(self, error: float, dt: float) -> float:
        """输入误差与时间步长，返回控制量。"""
        dt = max(float(dt), 1e-12)
        p = self.kp * error
        d_raw = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.d_filt = self.alpha * d_raw + (1 - self.alpha) * self.d_filt
        d = self.kd * self.d_filt
        i_candidate = self.integral + error * dt
        u_unsat = p + self.ki * i_candidate + d
        u = float(np.clip(u_unsat, self.out_min, self.out_max))
        # 抗积分饱和：仅当输出未饱和、或积分方向有助于脱离饱和时才累积
        if u == u_unsat or (u_unsat > self.out_max and error < 0) or (u_unsat < self.out_min and error > 0):
            self.integral = i_candidate
        self.prev_error = error
        self.last_output = u
        return u

    def update(self, measurement: float, dt: float) -> float:
        """以 setpoint − measurement 为误差的便捷接口。"""
        return self.step(self.setpoint - measurement, dt)


def simulate_execution_progress(pid: PID, target_curve, total_qty: float, dt: float = 1.0,
                                base_rate: float = 0.0, max_rate: float | None = None, noise_std: float = 0.0,
                                seed=None) -> dict:
    """用 PID 跟踪执行进度曲线的仿真。

    target_curve : (T,) 目标累计完成比例（0→1）；每步实际成交 = (base_rate + 控制量) · total_qty · dt + 噪声。
    返回 ``executed``（累计比例）、``rate``（每步参与率）、``error``。
    """
    rng = np.random.default_rng(seed)
    target = np.asarray(target_curve, dtype=float)
    T = len(target)
    done = 0.0
    executed = np.zeros(T); rates = np.zeros(T); errors = np.zeros(T)
    for t in range(T):
        e = target[t] - done
        u = pid.step(e, dt)
        rate = base_rate + u
        if max_rate is not None:
            rate = min(rate, max_rate)
        rate = max(rate, 0.0)
        fill = rate * dt + noise_std * rng.standard_normal() * dt
        done = min(1.0, max(done, done + fill))
        executed[t], rates[t], errors[t] = done, rate, e
    return dict(executed=executed, rate=rates, error=errors)
