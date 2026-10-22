"""
示例 3：组合风险 —— 随机矩阵理论去噪（物理）+ HRP/风险平价（复杂网络）样本外比较
运行：python examples/03_组合风险_RMT_HRP.py
"""
import numpy as np

import _common  # noqa: F401
from crossquant.physics.random_matrix import clip_eigenvalues, fit_mp_sigma2, mp_bounds
from crossquant.physics.networks import minimum_spanning_tree, correlation_distance
from crossquant.pipelines.portfolio_risk import allocation_comparison

rng = np.random.default_rng(7)
T, N = 750, 60
# 三个“行业”因子 + 市场因子 + 特异噪声
market = rng.normal(0, 0.01, T); sectors = rng.normal(0, 0.008, (T, 3))
beta = rng.normal(1, 0.3, N); sec_id = rng.integers(0, 3, N)
R = np.outer(market, beta) + sectors[:, sec_id] * rng.uniform(0.5, 1.5, N) + rng.normal(0, 0.012, (T, N))

train, test = R[:500], R[500:]
corr = np.corrcoef(train, rowvar=False)
q = N / 500
fit = fit_mp_sigma2(np.linalg.eigvalsh(corr), q)
print("=== Marchenko–Pastur ===  q=%.2f  拟合 σ²=%.2f  λ+=%.2f  显著因子数=%d（真值 4：市场+3 行业）" % (q, fit["sigma2"], fit["lambda_plus"], fit["n_signal"]))

mst = minimum_spanning_tree(correlation_distance(corr))
same_sector = np.mean([sec_id[i] == sec_id[j] for i, j, _ in mst["edges"]])
print("=== 最小生成树 ===  边数 %d，归一化树长 %.3f，同一行业内的边占比 %.0f%%（随机约 33%%）" % (len(mst["edges"]), mst["normalized_length"], 100 * same_sector))

print("\n=== 样本外年化波动（训练集估计协方差，测试集评估）===")
rows = allocation_comparison(train, test)
for k in sorted(rows, key=lambda k: rows[k]["oos_vol_annual"]):
    v = rows[k]
    print(f"{k:18s}  OOS vol {v['oos_vol_annual']:.4f}  HHI {v['herfindahl']:.3f}  max w {v['max_weight']:.3f}")
print("\n解读：去噪协方差（clip/rie/lw）+ 层次/风险平价配置通常比样本协方差 + 最小方差更稳、更分散。")
