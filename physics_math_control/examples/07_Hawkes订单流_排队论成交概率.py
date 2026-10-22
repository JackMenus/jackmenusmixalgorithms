"""
示例 7：订单流 —— Hawkes 自激励过程（地震学）与排队论（电话交换机）看限价单成交
运行：python examples/07_Hawkes订单流_排队论成交概率.py
"""
import numpy as np

import _common  # noqa: F401
from crossquant.physics.hawkes import fit_hawkes, fit_multivariate_hawkes, hawkes_residuals, simulate_hawkes, simulate_multivariate_hawkes
from crossquant.mathematics.queueing import expected_time_to_front, fill_probability_pure_death, prob_mid_up

print("=== 一维 Hawkes：市价单到达 ===")
ev = simulate_hawkes(mu=0.5, alpha=1.0, beta=1.5, T=4000, seed=1)
f = fit_hawkes(ev, 4000)
res = hawkes_residuals(ev[:2000], f["mu"], f["alpha"], f["beta"])
print(f"事件数 {len(ev)}，MLE: μ={f['mu']:.3f} α={f['alpha']:.3f} β={f['beta']:.3f}，分支比 n=α/β={f['branching_ratio']:.3f}（真值 0.667）；残差 KS p={res['ks_pvalue']:.2f}")
print("分支比 n 越接近 1，市场越“临界”：一个订单平均引发 n/(1−n) = %.1f 个后续订单" % (f["branching_ratio"] / (1 - f["branching_ratio"])))

print("\n=== 二维 Hawkes：买方/卖方市价单互相激励 ===")
mt, mk = simulate_multivariate_hawkes([0.3, 0.3], [[0.5, 0.2], [0.2, 0.5]], [[2.0, 2.0], [2.0, 2.0]], T=3000, seed=2)
mf = fit_multivariate_hawkes(mt, mk, 3000)
print("α 估计:\n", np.round(mf["alpha"], 2), "\n分支矩阵谱半径 %.3f（<1 平稳）" % mf["spectral_radius"])

print("\n=== 排队论：我的限价单前面还有 q 份 ===")
lam_mo, lam_c, nu = 1.0, 0.05, 0.4   # 市价单率、每单撤单率、价格变动率（每秒）
for q in [1, 3, 5, 10, 20]:
    print(f"q={q:2d}  在价格变动前成交概率 {fill_probability_pure_death(q, lam_mo, lam_c, nu):.3f}  期望排到队首 {expected_time_to_front(q, lam_mo, lam_c):5.2f} s")
print("订单簿不平衡 ⇒ 中间价方向：P(上涨 | q_bid=8, q_ask=2) = %.2f，P(上涨 | q_bid=2, q_ask=8) = %.2f" % (
    prob_mid_up(8, 2, 1.0, 1.0, 0.1, 2000, seed=0), prob_mid_up(2, 8, 1.0, 1.0, 0.1, 2000, seed=0)))
