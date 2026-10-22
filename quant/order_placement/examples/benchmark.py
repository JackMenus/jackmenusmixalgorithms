"""一键基准：做市模型对比、执行算法对比、RL vs TWAP。运行：python examples/benchmark.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from order_placement import (AvellanedaStoikov, GLFT, AdaptiveGLFT, MarketMakerConfig, LatencyProfile,
                             ParentOrder, Side, ScheduledExecutor)
from order_placement.execution import ACParams, almgren_chriss, obizhaeva_wang, twap
from order_placement.simulation import MarketMakingSim, ExecutionSim
from order_placement.rl import ExecutionEnv, ExecEnvConfig, QLearningExecutor


def mm_bench(seeds=range(5)):
    print("\n== 做市：600 秒，σ=0.02，A=1，k=50，含 OU alpha 漂移 ==")
    base = dict(gamma=0.2, sigma=0.02, A=1.0, k=50.0, max_inventory=20)
    cases = {
        "AS(无alpha)": lambda: AvellanedaStoikov(MarketMakerConfig(**base, alpha_weight=0.0)),
        "GLFT(无alpha)": lambda: GLFT(MarketMakerConfig(**base, alpha_weight=0.0)),
        "GLFT+alpha": lambda: GLFT(MarketMakerConfig(**base, alpha_weight=1.0)),
        "AdaptiveGLFT+alpha": lambda: AdaptiveGLFT(MarketMakerConfig(**base, alpha_weight=1.0)),
    }
    print(f"{'策略':<22}{'PnL均值':>10}{'PnL标准差':>10}{'平均|库存|':>12}{'成交数':>8}")
    for name, mk in cases.items():
        res = [MarketMakingSim(T=600, alpha_kappa=0.05, alpha_eta=0.002, seed=s).run(mk()).stats for s in seeds]
        p = np.array([r["final_pnl"] for r in res])
        print(f"{name:<22}{p.mean():>10.3f}{p.std():>10.3f}{np.mean([r['mean_abs_inventory'] for r in res]):>12.2f}"
              f"{np.mean([r['n_fills'] for r in res]):>8.0f}")
    # 延迟实验关闭 alpha：否则完美已知的漂移会让“多成交=多方向性敞口”掩盖延迟成本
    print("\n-- 延迟（无 alpha）：不同单向延迟 × 是否按 z·σ·√RTT 加宽 --")
    for ms in (0.0, 200.0, 1000.0):
        lat = LatencyProfile(one_way_ms=ms) if ms else None
        row = []
        for z in (0.0, 1.0):
            res = [MarketMakingSim(T=600, latency=lat, seed=s)
                   .run(GLFT(MarketMakerConfig(**base, latency_z=z))).stats["final_pnl"] for s in seeds]
            row.append(np.mean(res))
        print(f"单向 {ms:>6.0f}ms   不加宽 PnL {row[0]:7.3f}   加宽(z=1) PnL {row[1]:7.3f}")


def exec_bench(seeds=range(20)):
    print("\n== 执行：买入 2000 股 / 200 秒 / 20 区间 ==")
    X, T, N = 2000, 200.0, 20
    scheds = {
        "TWAP": twap(X, N),
        "Almgren-Chriss(λ=1e-4)": almgren_chriss(X, T, N, ACParams(sigma=0.02, eta=1e-3, risk_aversion=1e-4)),
        "Obizhaeva-Wang(ρ=0.1)": obizhaeva_wang(X, T, N, rho=0.1),
    }
    for name, n in scheds.items():
        res = [ExecutionSim(seed=s).run(ScheduledExecutor(ParentOrder("SIM", Side.BUY, X, 0.0, T), n)) for s in seeds]
        sf = np.array([r["shortfall_bps"] for r in res])
        print(f"{name:<26} IS 均值 {sf.mean():6.2f} bps  标准差 {sf.std():5.2f}  完成率 {np.mean([r['completion'] for r in res]):.2f}")


def rl_bench():
    print("\n== RL 执行（表格 Q-learning，可观测漂移）==")
    cfg = ExecEnvConfig(drift_vol=0.05, sigma=0.02)
    agent = QLearningExecutor(cfg).train(ExecutionEnv(cfg, seed=0), episodes=20000)
    print("Q-learning:", agent.evaluate(ExecutionEnv(cfg, seed=1), 2000, "q"))
    print("TWAP      :", agent.evaluate(ExecutionEnv(cfg, seed=1), 2000, "twap"))


if __name__ == "__main__":
    mm_bench()
    exec_bench()
    rl_bench()
