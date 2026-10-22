"""
示例 6：路径签名 —— 粗糙路径理论的签名特征、Lévy 面积、签名交易（Futter–Horvath–Wiese 简化版）
运行：python examples/06_路径签名特征与签名交易.py
"""
import numpy as np

import _common  # noqa: F401
from crossquant.mathematics.signatures import SignatureTrading, add_time, lead_lag, levy_area, signature_flat, sliding_signature_features

rng = np.random.default_rng(0)
# 领先滞后对：y 滞后 x 两期 ⇒ Lévy 面积有确定符号
x = np.cumsum(rng.normal(size=500)); y = np.roll(x, 2) + 0.3 * rng.normal(size=500)
path = np.stack([x, y], 1)[5:]
print("二维路径 (x, y) 的 Lévy 面积 A_xy = %.1f（符号反映 x 领先 y 的“旋转方向”）" % levy_area(path)[0, 1])
feat = sliding_signature_features(x, window=20, depth=3, transform=add_time)
print("滚动签名特征矩阵形状:", feat["features"].shape, "（含时间增广，3 阶，2 维路径 ⇒ 2+4+8=14 维）")
print("lead-lag 变换后的签名维数:", signature_flat(lead_lag(x[:20, None]), 2).shape[0])

# 签名交易：在均值回复序列上训练，样本外回测
s = np.zeros(3000)
for t in range(1, 3000):
    s[t] = 0.9 * s[t - 1] + rng.normal()
for depth in [1, 2, 3]:
    st = SignatureTrading(window=10, depth=depth, risk_aversion=1.0, ridge=1e-3).fit(s[:2000])
    bt = st.backtest(s[2000:])
    print(f"签名交易 depth={depth}: 样本外每期 Sharpe {bt['sharpe']:.3f}  累计 PnL {bt['pnl'].sum():.1f}")
print("解读：depth=1 的签名就是“最近增量”，更高阶引入路径的非线性（面积、曲率）信息。")
