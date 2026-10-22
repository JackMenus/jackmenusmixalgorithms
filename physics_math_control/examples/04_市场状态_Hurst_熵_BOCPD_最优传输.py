"""
示例 4：市场状态仪表盘 —— Hurst/DFA（标度物理）+ 排列熵（信息论）+ BOCPD（贝叶斯变点）+ W₁ 距离（最优传输）
运行：python examples/04_市场状态_Hurst_熵_BOCPD_最优传输.py
"""
import numpy as np

import _common  # noqa: F401
from crossquant.pipelines.regime import regime_dashboard
from crossquant.physics.rough_volatility import fbm_davies_harte

rng = np.random.default_rng(3)
# 三段 regime：随机游走 → 趋势持续(H=0.7 的 fGn) → 高波动均值回复(H=0.3)
seg1 = rng.normal(0, 0.01, 600)
seg2 = np.diff(fbm_davies_harte(600, 0.7, seed=1)) * 0.01 / np.std(np.diff(fbm_davies_harte(600, 0.7, seed=1)))
seg3 = np.diff(fbm_davies_harte(600, 0.3, seed=2)); seg3 = seg3 / seg3.std() * 0.025
r = np.concatenate([seg1, seg2, seg3])

rd = regime_dashboard(r, window=150, step=10)
def seg_mean(arr, lo, hi):
    m = (rd["index"] >= lo) & (rd["index"] < hi)
    return float(np.nanmean(arr[m]))
print("指标              | 随机游走段 | 趣势段(H=0.7) | 高波动反持续段(H=0.3)")
for name, arr in [("Hurst(DFA)", rd["hurst"]), ("排列熵", rd["permutation_entropy"]), ("W1 距离", rd["wasserstein"]), ("Fisher–Rao", rd["fisher_rao"]), ("BOCPD 变点概率", rd["bocpd_change_prob"])]:
    print(f"{name:16s} | {seg_mean(arr, 150, 600):10.3f} | {seg_mean(arr, 750, 1200):13.3f} | {seg_mean(arr, 1350, 1800):18.3f}")
print("\n解读：Hurst 跟随记忆性切换，W1/Fisher–Rao 在波动率跳升时抬高，BOCPD 的 run length 在段边界归零。")

plt = _common.try_plot()
if plt is not None:
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(np.cumsum(r)); ax[0].set_title("累计收益（三段 regime）")
    ax[1].plot(rd["index"], rd["hurst"]); ax[1].axhline(0.5, c="k", ls=":"); ax[1].set_title("滚动 Hurst (DFA)")
    ax[2].plot(rd["index"], rd["wasserstein"]); ax[2].set_title("滚动 W1 距离（相对全样本）")
    ax[3].plot(rd["index"], rd["bocpd_run_length"]); ax[3].set_title("BOCPD 最大后验 run length")
    fig.tight_layout(); fig.savefig(_common.out_path("04_regime.png"), dpi=120); print("图已保存:", _common.out_path("04_regime.png"))
