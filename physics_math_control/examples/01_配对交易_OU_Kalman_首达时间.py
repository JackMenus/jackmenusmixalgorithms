"""
示例 1：配对交易 —— Langevin/OU（物理）+ Kalman 动态对冲（控制）+ 首达时间（扩散方程）+ ADF（数学）
运行：python examples/01_配对交易_OU_Kalman_首达时间.py
"""
import numpy as np

import _common  # noqa: F401
from crossquant.pipelines.pairs_trading import pairs_trading_pipeline
from crossquant.control.optimal_stopping import ou_optimal_entry_exit

rng = np.random.default_rng(42)
T = 2000
px = 100 + np.cumsum(rng.normal(0, 1, T))                 # 资产 X：随机游走
spread = np.zeros(T)
for t in range(1, T):                                       # 协整残差：AR(1) ≈ OU
    spread[t] = 0.92 * spread[t - 1] + 0.6 * rng.normal()
py = 1.4 * px + 20 + spread                                 # 资产 Y = β X + α + 价差

res = pairs_trading_pipeline(py, px, train_frac=0.6, entry_z=1.5, exit_z=0.0, cost=0.01, use_kalman=True, seed=0)
print("=== 对冲比率 (OLS, 训练集) ===", {k: round(v, 3) for k, v in res["hedge"].items()})
print("=== ADF 单位根检验 ===  统计量 %.2f，5%% 临界值 %.2f，拒绝单位根: %s" % (res["adf"]["stat"], res["adf"]["critical_values"]["5%"], res["adf"]["reject"]["5%"]))
print("=== OU 参数 ===", {k: round(v, 3) for k, v in res["ou"].items()})
print("=== 从 +1.5σ 回到均值的首达时间 (MC) ===", {k: round(v, 2) for k, v in res["revert_time"].items()})
print("=== 样本外回测 ===", {k: round(v, 3) for k, v in res["backtest"].items()})
print("Kalman 动态 β 末值: %.3f（真值 1.4）" % res["kalman"]["beta"][-1])

# 用最优停止理论替代固定阈值：Leung–Li 双重停止给出最优进出场水平
ou = res["ou"]
oe = ou_optimal_entry_exit(theta=ou["theta"], kappa=ou["kappa"], sigma=ou["sigma"], r=0.02, cost=0.01, dt=1.0, n_grid=301)
print("=== 最优停止阈值 ===  出场 b* = %.3f（%.2fσ），进场 d* = %.3f（%.2fσ）" % (
    oe["exit_threshold"], (oe["exit_threshold"] - ou["theta"]) / ou["stationary_std"],
    oe["entry_threshold"], (oe["entry_threshold"] - ou["theta"]) / ou["stationary_std"]))

plt = _common.try_plot()
if plt is not None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(res["zscore"], lw=0.8); ax[0].axhline(1.5, c="r", ls="--"); ax[0].axhline(-1.5, c="r", ls="--"); ax[0].axvline(res["n_train"], c="k", ls=":")
    ax[0].set_title("价差 z-score（OU 参数来自训练集，虚线为训练/测试分界）")
    ax[1].plot(res["kalman"]["beta"], lw=0.8); ax[1].axhline(1.4, c="r", ls="--"); ax[1].set_title("Kalman 动态对冲比率 β_t")
    fig.tight_layout(); fig.savefig(_common.out_path("01_pairs.png"), dpi=120)
    print("图已保存:", _common.out_path("01_pairs.png"))
