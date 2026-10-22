"""
示例 2：做市 —— Avellaneda–Stoikov 与 GLFT 的 HJB 随机控制解，对比对称报价
运行：python examples/02_做市_AS_GLFT_随机控制.py
"""
import numpy as np

import _common  # noqa: F401
from crossquant.control.market_making import (as_strategy, glft_asymptotic_quotes, glft_closed_form_quotes,
                                              glft_strategy, simulate_market_making, symmetric_strategy)

gamma, sigma, k, A, T = 0.1, 2.0, 1.5, 140.0, 1.0
print("=== GLFT 渐近最优报价距离（库存 q = −5..5）===")
asy = glft_asymptotic_quotes(gamma, sigma, k, A, Q=5)
cf = glft_closed_form_quotes(asy["q"], gamma, sigma, k, A)
for q, db, da, cb, ca in zip(asy["q"], asy["delta_b"], asy["delta_a"], cf["delta_b"], cf["delta_a"]):
    print(f"q={q:+d}  精确: δ_b={db:5.3f} δ_a={da:5.3f} | 闭式近似: δ_b={cb:5.3f} δ_a={ca:5.3f}")

print("\n=== 1000 次蒙特卡洛：终端 PnL 的均值 / 标准差 / 终端库存绝对值 ===")
strategies = {"AS 库存策略": as_strategy(gamma, sigma, k, T), "GLFT 渐近(Q=5)": glft_strategy(gamma, sigma, k, A, 5),
              "对称固定价差": symmetric_strategy(0.7)}
for name, fn in strategies.items():
    pnl, inv = [], []
    for s in range(1000):
        r = simulate_market_making(fn, S0=100, sigma=sigma, T=T, dt=0.005, A=A, k=k, seed=s)
        pnl.append(r["final_pnl"]); inv.append(abs(r["final_inventory"]))
    print(f"{name:12s}  PnL 均值 {np.mean(pnl):6.2f}  标准差 {np.std(pnl):6.2f}  |库存| 均值 {np.mean(inv):4.2f}")
print("\n解读：AS/GLFT 用 γσ²(T−t)·q 的保留价格把库存风险“定价”进报价，牺牲少量期望收益换取方差大幅下降。")
