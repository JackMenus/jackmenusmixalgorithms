"""跨学科库 crossquant 钩子演示：Kalman 公允价外推、OU 价差预测、PID / LQR 参与率跟踪、两种冲击模型成本对比。

    python examples/run_crossquant_hooks.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantstack.core.registry import build_sim_pipeline, crossquant_entries, discover, make_controller, make_impact_model, make_state_estimator
from quantstack.sim.market import make_synthetic_panel

if not discover(["crossquant"]).get("crossquant"):
    sys.exit("crossquant 不可用：pip install numpy scipy，并确认 /mnt/project-files/physics_math_control 存在")

US = 1_000_000
print("crossquant 登记:", {k: sorted(v) for k, v in crossquant_entries().items()})

panel = make_synthetic_panel(200, 10, seed=1)
pipe = build_sim_pipeline(panel, kinds={"alpha": "reference", "execution": "reference", "router": "reference"}, seed=1)
sym = list(panel["close"].columns)[0]

# 1) 状态估计：Kalman 对中间价做 10 秒外推，与「最近中间价」的朴素预测比较；OU 拟合价差
kalman = make_state_estimator("crossquant.KalmanTrendEstimator", level_var=1e-4, slope_var=1e-8, obs_var=1e-3)
ou = make_state_estimator("crossquant.OUSpreadEstimator")
mids, kalman_pred, naive_pred = [], [], []
H = 10
for k in range(1, 301):
    ts = k * US
    pipe.feed.advance_to(ts)
    snap = pipe.feed.snapshot(sym, ts)
    mids.append(snap.mid)
    kalman.update(snap.mid, ts)
    ou.update(snap.spread_bps, ts)
    kalman_pred.append(kalman.predict(H * US))
    naive_pred.append(snap.mid)
mids_a = np.array(mids)
err_k = np.abs(np.array(kalman_pred[:-H]) - mids_a[H:]).mean()
err_n = np.abs(np.array(naive_pred[:-H]) - mids_a[H:]).mean()
print(f"\n[Kalman] {sym} 最新 mid={mids[-1]:.4f} 滤波 state={kalman.state} | 10 秒外推 MAE: Kalman {err_k:.4f} vs 朴素 {err_n:.4f}")
print("        注：模拟器中间价是 GBM（鞅），趋势外推理论上不会优于朴素预测，此处只演示接口；Kalman 的价值在有漂移/延迟外推场景与真实数据上体现。")
print(f"[OU] 价差 bps 最新={snap.spread_bps:.2f} 5 秒后条件期望={ou.predict(5 * US):.2f} 参数={ {k: round(v, 4) for k, v in ou.params.items()} }")

# 2) 控制器：TWAP 目标进度 vs 实际进度，用参与率作控制量（随机成交量扰动，seed 固定）
def track(controller, steps=120, base_rate=0.10, seed=1):
    rng = np.random.default_rng(seed)
    filled, rate, errs = 0.0, base_rate, []
    for k in range(1, steps + 1):
        filled = min(1.0, filled + rate / base_rate * rng.uniform(0.4, 1.4) / steps)
        err = k / steps - filled
        errs.append(err)
        if controller is not None:
            rate = float(np.clip(base_rate + controller.step(err, US), 0.01, 0.30))
    return float(np.sqrt(np.mean(np.square(errs)))), filled

for name, ctrl in [("无控制", None), ("PID", make_controller("crossquant.PIDParticipationController", kp=0.8, ki=0.05)), ("LQR", make_controller("crossquant.LQRTrackingController", q=1.0, r=0.1))]:
    rmse, done = track(ctrl)
    print(f"[控制] {name:<5} 进度跟踪 RMSE={rmse:.4f} 最终完成={done:.3f}")

# 3) 冲击模型：平方根律 vs Almgren–Chriss 期望成本
sqrt_model = make_impact_model("crossquant.SquareRootImpactModel")
ac_model = make_impact_model("crossquant.AlmgrenChrissImpactModel")
print(f"\n[冲击] {sym} ADV={snap.adv:,.0f} σ={snap.volatility:.3f} 价差={snap.spread_bps:.2f}bps")
print(f"{'数量':>8} {'时长':>8} {'平方根律 bps':>14} {'AC bps':>10}")
for qty in (5_000, 50_000, 200_000):
    for label, dur in (("5min", 300 * US), ("30min", 1800 * US), ("全天", 23_400 * US)):
        print(f"{qty:>8} {label:>8} {sqrt_model.expected_cost_bps(qty, snap, dur):>14.2f} {ac_model.expected_cost_bps(qty, snap, dur):>10.2f}")
