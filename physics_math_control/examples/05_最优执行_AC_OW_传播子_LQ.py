"""
示例 5：最优执行 —— Almgren–Chriss（LQ 变分）、Obizhaeva–Wang（线性响应/瞬时冲击）、Gârleanu–Pedersen（动态规划）
运行：python examples/05_最优执行_AC_OW_传播子_LQ.py
"""
import numpy as np

import _common  # noqa: F401
from crossquant.control.execution import almgren_chriss, almgren_chriss_frontier, obizhaeva_wang, optimal_schedule_transient, power_law_kernel, square_root_impact
from crossquant.control.lqr import GarleanuPedersen

X, T, N = 1_000_000, 5, 5
print("=== Almgren–Chriss（AC 2000 原文参数）===")
for lam in [0.0, 1e-6, 1e-5]:
    r = almgren_chriss(X, T, N, sigma=0.95, eta=2.5e-6, lam=lam, gamma=2.5e-7, epsilon=0.0625)
    print(f"λ={lam:.0e}  κ={r['kappa']:.3f}  持仓轨迹 {np.round(r['holdings'] / 1e3).astype(int)} (千股)  E[成本]=${r['expected_cost']:,.0f}  σ[成本]=${np.sqrt(r['variance']):,.0f}")
fr = almgren_chriss_frontier(X, T, N, 0.95, 2.5e-6, np.logspace(-8, -5, 6), gamma=2.5e-7, epsilon=0.0625)
print("效率前沿 (E, σ):", [(round(e / 1e3), round(np.sqrt(v) / 1e3)) for e, v in zip(fr["expected_cost"], fr["variance"])], "(千美元)")

print("\n=== Obizhaeva–Wang：指数回弹的块状订单簿 ===")
ow = obizhaeva_wang(1.0, T=1.0, N=10, rho=2.0)
print("交易比例:", np.round(ow["trades"], 3), " ⇒ 首尾大块 + 中间匀速")
times = ow["times"]
gss = optimal_schedule_transient(times, power_law_kernel(1.0, 0.1, 0.5), 1.0)
print("幂律核 (β=0.5) 下的最优调度:", np.round(gss["trades"], 3), " ⇒ 更强的首尾集中（U 形）")
print("平方根冲击律：买入 5%% ADV，σ_daily=2%%  ⇒ 冲击 ≈ %.1f bps" % (1e4 * square_root_impact(0.05, 1.0, 0.02)))

print("\n=== Gârleanu–Pedersen：有成本的动态组合 ===")
Sigma = np.diag([0.04, 0.09]); B = np.array([[0.02, 0.0], [0.0, 0.03]]); Phi = np.diag([0.1, 0.5])  # 因子2 衰减快
gp = GarleanuPedersen(Sigma, Lambda=5 * Sigma, B=B, Phi=Phi, gamma=2.0, rho=0.0)
f = np.array([1.0, 1.0])
print("Markowitz 目标:", np.round(gp.markowitz(f), 3), " aim（按因子衰减打折）:", np.round(gp.aim(f), 3))
print("交易矩阵 Λ^{-1}A_xx 的对角（每期向 aim 靠近的比例）:", np.round(np.diag(gp.trade_matrix), 3))
rng = np.random.default_rng(0); fpath = np.zeros((50, 2)); fpath[0] = f
for t in range(1, 50):
    fpath[t] = fpath[t - 1] - Phi @ fpath[t - 1] + 0.1 * rng.standard_normal(2)
xs = gp.simulate(fpath)
print("50 期持仓路径的换手（Σ|Δx|）: %.3f，相比每期直接持有 Markowitz 组合: %.3f" % (
    np.abs(np.diff(xs, axis=0)).sum(), np.abs(np.diff(np.array([gp.markowitz(ff) for ff in fpath]), axis=0)).sum()))
