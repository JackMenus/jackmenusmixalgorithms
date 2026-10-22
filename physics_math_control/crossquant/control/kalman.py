"""
卡尔曼滤波（Kalman Filter）及其在量化中的应用
================================================

学科来源
--------
控制论 / 航天工程（Kalman 1960，Apollo 导航）。线性高斯状态空间模型的最小方差递推估计器：

    状态方程   x_t = F x_{t-1} + B u_t + w_t,   w_t ~ N(0, Q)
    观测方程   z_t = H_t x_t + v_t,             v_t ~ N(0, R)

量化映射
--------
1. **动态对冲比率 / 配对交易**：把回归系数 β_t、α_t 看成随机游走的隐状态，观测方程
   y_t = β_t x_t + α_t + v_t，Kalman 增益自动给出“自适应 β”，新息 e_t 及其标准差
   sqrt(S_t) 直接构成 z-score 交易信号（Chan 2013, *Algorithmic Trading* 第 3 章）。
2. **隐含趋势 / 局部水平模型**：价格 = 趋势 + 噪声，用 KF 做低延迟平滑（替代移动平均）。
3. **随机波动率 QML**：Harvey–Ruiz–Shephard (1994) 把 log(r_t^2) 线性化后用 KF 估计 SV 模型。
4. **延迟 / 价差的隐状态估计**：路由层对交易所往返时延做滤波（见 ``crossquant.adapters``）。

核心公式
--------
预测：  x̂_{t|t-1} = F x̂_{t-1},  P_{t|t-1} = F P_{t-1} Fᵀ + Q
更新：  e_t = z_t − H x̂_{t|t-1},  S_t = H P_{t|t-1} Hᵀ + R,  K_t = P_{t|t-1} Hᵀ S_t^{-1}
        x̂_t = x̂_{t|t-1} + K_t e_t,  P_t = (I − K_t H) P_{t|t-1}
对数似然：  ℓ = −½ Σ_t [ log|S_t| + e_tᵀ S_t^{-1} e_t + m log 2π ]

参考文献
--------
- Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems.
- Harvey, Ruiz & Shephard (1994). Multivariate Stochastic Variance Models. RES.
- Chan, E. (2013). Algorithmic Trading: Winning Strategies and Their Rationale. Ch. 3.
- Rauch, Tung & Striebel (1965). Maximum likelihood estimates of linear dynamic systems（RTS 平滑）。
- Julier & Uhlmann (1997). A New Extension of the Kalman Filter to Nonlinear Systems（UKF）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .._utils import as_1d


@dataclass
class KalmanFilter:
    """线性高斯卡尔曼滤波器（支持时变观测矩阵 H_t）。

    参数
    ----
    F : (n, n) 状态转移矩阵
    Q : (n, n) 状态噪声协方差
    R : (m, m) 观测噪声协方差
    H : (m, n) 观测矩阵；若观测矩阵随时间变化，在 ``filter`` 中传入 ``Hs``
    x0, P0 : 初始状态均值与协方差
    B : (n, k) 控制输入矩阵（可选）
    """

    F: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    H: Optional[np.ndarray] = None
    x0: Optional[np.ndarray] = None
    P0: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None

    def __post_init__(self):
        self.F = np.atleast_2d(np.asarray(self.F, dtype=float))
        self.Q = np.atleast_2d(np.asarray(self.Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(self.R, dtype=float))
        n = self.F.shape[0]
        if self.H is not None:
            self.H = np.atleast_2d(np.asarray(self.H, dtype=float))
        self.x = np.zeros(n) if self.x0 is None else np.asarray(self.x0, dtype=float).copy()
        self.P = np.eye(n) if self.P0 is None else np.asarray(self.P0, dtype=float).copy()

    # ----------------------------------------------------------------- 单步接口
    def predict(self, u: Optional[np.ndarray] = None):
        """时间更新：返回 (x̂_{t|t-1}, P_{t|t-1})。"""
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.asarray(u, dtype=float)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z, H: Optional[np.ndarray] = None):
        """观测更新：返回 (x̂_t, P_t, 新息 e_t, 新息协方差 S_t)。"""
        H = self.H if H is None else np.atleast_2d(np.asarray(H, dtype=float))
        z = np.atleast_1d(np.asarray(z, dtype=float))
        e = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T  # = P Hᵀ S^{-1}
        self.x = self.x + K @ e
        I_KH = np.eye(self.P.shape[0]) - K @ H
        # Joseph 形式，数值更稳定
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy(), self.P.copy(), e, S

    # ----------------------------------------------------------------- 批量接口
    def filter(self, zs, Hs: Optional[np.ndarray] = None, us: Optional[np.ndarray] = None) -> dict:
        """对整段观测序列滤波。

        参数
        ----
        zs : (T, m) 或 (T,) 观测
        Hs : (T, m, n) 时变观测矩阵（可选）
        us : (T, k) 控制输入（可选）

        返回 dict：``x_pred, P_pred, x_filt, P_filt, innovations, S, loglik``
        """
        zs = np.asarray(zs, dtype=float)
        if zs.ndim == 1:
            zs = zs[:, None]
        T, m = zs.shape
        n = self.F.shape[0]
        x_pred = np.zeros((T, n)); P_pred = np.zeros((T, n, n))
        x_filt = np.zeros((T, n)); P_filt = np.zeros((T, n, n))
        innov = np.zeros((T, m)); S_all = np.zeros((T, m, m))
        loglik = 0.0
        for t in range(T):
            u = None if us is None else us[t]
            self.predict(u)
            x_pred[t], P_pred[t] = self.x, self.P
            H = None if Hs is None else Hs[t]
            _, _, e, S = self.update(zs[t], H)
            x_filt[t], P_filt[t] = self.x, self.P
            innov[t], S_all[t] = e, S
            sign, logdet = np.linalg.slogdet(S)
            loglik += -0.5 * (logdet + e @ np.linalg.solve(S, e) + m * np.log(2 * np.pi))
        return dict(x_pred=x_pred, P_pred=P_pred, x_filt=x_filt, P_filt=P_filt,
                    innovations=innov, S=S_all, loglik=loglik)

    def smooth(self, filt: dict) -> dict:
        """Rauch–Tung–Striebel 固定区间平滑（需先调用 ``filter``）。返回 ``x_smooth, P_smooth``。"""
        x_f, P_f, x_p, P_p = filt["x_filt"], filt["P_filt"], filt["x_pred"], filt["P_pred"]
        T, n = x_f.shape
        x_s = x_f.copy(); P_s = P_f.copy()
        for t in range(T - 2, -1, -1):
            C = P_f[t] @ self.F.T @ np.linalg.pinv(P_p[t + 1])
            x_s[t] = x_f[t] + C @ (x_s[t + 1] - x_p[t + 1])
            P_s[t] = P_f[t] + C @ (P_s[t + 1] - P_p[t + 1]) @ C.T
        return dict(x_smooth=x_s, P_smooth=P_s)


# --------------------------------------------------------------------------- 应用 1：动态对冲比率
def dynamic_hedge_ratio(y, x, delta: float = 1e-4, r: float = 1e-3,
                        beta0: Optional[np.ndarray] = None, P0_scale: float = 1.0) -> dict:
    """Kalman 动态对冲比率（配对交易经典用法）。

    模型：  y_t = β_t x_t + α_t + v_t,  (β_t, α_t) 服从随机游走，Q = δ/(1−δ) I，R = r。

    参数
    ----
    y, x : 两条价格（或对数价格）序列
    delta : 状态噪声比例（越大 β 变化越快，典型 1e-5 ~ 1e-3）
    r : 观测噪声方差

    返回 dict：
        beta, alpha : 滤波后的对冲比率与截距 (T,)
        spread      : 新息 e_t = y_t − (β x_t + α)，即“预测残差”
        spread_std  : sqrt(S_t)，新息标准差
        zscore      : e_t / sqrt(S_t)，常用交易信号（|z|>1~2 开仓，回到 0 平仓）
        loglik      : 对数似然（可用于选择 delta）
    """
    y = as_1d(y, "y"); x = as_1d(x, "x")
    T = len(y)
    Hs = np.stack([x, np.ones(T)], axis=1)[:, None, :]  # (T,1,2)
    Qv = delta / (1.0 - delta)
    kf = KalmanFilter(F=np.eye(2), Q=Qv * np.eye(2), R=np.array([[r]]),
                      x0=np.zeros(2) if beta0 is None else beta0, P0=P0_scale * np.eye(2))
    out = kf.filter(y, Hs=Hs)
    spread = out["innovations"][:, 0]
    spread_std = np.sqrt(out["S"][:, 0, 0])
    return dict(beta=out["x_filt"][:, 0], alpha=out["x_filt"][:, 1], spread=spread,
                spread_std=spread_std, zscore=spread / spread_std, loglik=out["loglik"])


def optimize_noise_ratio(y, x, deltas=None, r: float = 1e-3) -> dict:
    """用对数似然在网格上选择 ``dynamic_hedge_ratio`` 的 delta（最大似然）。"""
    deltas = np.logspace(-7, -1, 25) if deltas is None else np.asarray(deltas)
    lls = np.array([dynamic_hedge_ratio(y, x, d, r)["loglik"] for d in deltas])
    return dict(deltas=deltas, loglik=lls, best_delta=float(deltas[np.argmax(lls)]))


# --------------------------------------------------------------------------- 应用 2：局部水平/趋势模型
def local_level_filter(z, q_ratio: float = 0.01) -> dict:
    """局部水平模型（随机游走 + 观测噪声），q_ratio = Q/R 控制平滑程度。

    返回 ``level``（滤波后的水平）与 ``smooth``（RTS 平滑后的水平）。
    信号/噪声比 q_ratio 越小越平滑；稳态下等价于一个 EWMA，但 KF 自动处理起始段与缺失值。
    """
    z = as_1d(z, "z")
    R = np.var(np.diff(z)) if len(z) > 2 else 1.0
    kf = KalmanFilter(F=np.eye(1), Q=np.array([[q_ratio * R]]), R=np.array([[R]]),
                      H=np.eye(1), x0=np.array([z[0]]), P0=np.array([[R]]))
    out = kf.filter(z)
    sm = kf.smooth(out)
    return dict(level=out["x_filt"][:, 0], smooth=sm["x_smooth"][:, 0], loglik=out["loglik"])


# --------------------------------------------------------------------------- 应用 3：随机波动率 QML
def sv_qml_filter(returns, params: Optional[dict] = None, estimate: bool = True) -> dict:
    """Harvey–Ruiz–Shephard (1994) 准极大似然：用 Kalman 滤波估计随机波动率模型。

    SV 模型：  r_t = exp(h_t/2) ε_t,   h_t = μ + φ(h_{t-1} − μ) + σ_η η_t
    线性化：  log r_t² = h_t + log ε_t²，其中 E[log ε²] = −1.2704，Var = π²/2（ε ~ N(0,1)）。
    于是 y_t = log r_t² + 1.2704 = h_t + ξ_t，ξ_t 近似为均值 0、方差 π²/2 的噪声 → 标准 KF。

    返回 dict：``h_filt``（滤波对数方差）、``vol``（exp(h/2)）、``params``（μ, φ, σ_η）、``loglik``。
    """
    r = as_1d(returns, "returns")
    y = np.log(np.maximum(r ** 2, 1e-12)) + 1.2704
    var_xi = np.pi ** 2 / 2

    def run(mu, phi, sig_eta):
        kf = KalmanFilter(F=np.array([[phi]]), Q=np.array([[sig_eta ** 2]]), R=np.array([[var_xi]]),
                          H=np.eye(1), x0=np.array([mu]), P0=np.array([[sig_eta ** 2 / max(1 - phi ** 2, 1e-6)]]))
        # 带常数项的转移：h_t − μ = φ(h_{t−1} − μ) ⇒ 对 (y − μ) 滤波
        out = kf.filter(y - mu)
        return out

    if estimate:
        def nll(theta):
            mu, phi, log_sig = theta
            phi = np.tanh(phi)
            return -run(mu, phi, np.exp(log_sig))["loglik"]
        theta0 = np.array([np.mean(y), np.arctanh(0.9), np.log(0.3)])
        from scipy import optimize  # 延迟导入，保持模块加载轻量

        res = optimize.minimize(nll, theta0, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-4, fatol=1e-4))
        mu, phi, sig_eta = res.x[0], float(np.tanh(res.x[1])), float(np.exp(res.x[2]))
    else:
        mu, phi, sig_eta = params["mu"], params["phi"], params["sigma_eta"]
    out = run(mu, phi, sig_eta)
    h = out["x_filt"][:, 0] + mu
    return dict(h_filt=h, vol=np.exp(h / 2), params=dict(mu=float(mu), phi=phi, sigma_eta=sig_eta),
                loglik=out["loglik"])


# --------------------------------------------------------------------------- 非线性：UKF
@dataclass
class UnscentedKalmanFilter:
    """无迹卡尔曼滤波（加性噪声版本）。

    f(x) 状态转移函数，h(x) 观测函数，Q/R 噪声协方差。sigma 点参数 alpha, beta, kappa。
    用途：非线性定价/对冲状态（如隐含波动率曲面参数）、非线性延迟模型。
    """

    f: Callable[[np.ndarray], np.ndarray]
    h: Callable[[np.ndarray], np.ndarray]
    Q: np.ndarray
    R: np.ndarray
    x0: np.ndarray
    P0: np.ndarray
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    x: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x = np.asarray(self.x0, dtype=float).copy()
        self.P = np.asarray(self.P0, dtype=float).copy()
        n = len(self.x)
        lam = self.alpha ** 2 * (n + self.kappa) - n
        self._lam = lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)

    def _sigma_points(self):
        n = len(self.x)
        S = np.linalg.cholesky((n + self._lam) * (self.P + 1e-12 * np.eye(n)))
        pts = np.vstack([self.x, self.x + S.T, self.x - S.T])
        return pts

    def step(self, z):
        """一次预测 + 更新，返回 (x̂_t, P_t)。"""
        pts = self._sigma_points()
        fx = np.array([self.f(p) for p in pts])
        x_pred = self.Wm @ fx
        P_pred = (fx - x_pred).T @ np.diag(self.Wc) @ (fx - x_pred) + self.Q
        hx = np.array([np.atleast_1d(self.h(p)) for p in fx])
        z_pred = self.Wm @ hx
        Pzz = (hx - z_pred).T @ np.diag(self.Wc) @ (hx - z_pred) + self.R
        Pxz = (fx - x_pred).T @ np.diag(self.Wc) @ (hx - z_pred)
        K = Pxz @ np.linalg.inv(Pzz)
        self.x = x_pred + K @ (np.atleast_1d(z) - z_pred)
        self.P = P_pred - K @ Pzz @ K.T
        return self.x.copy(), self.P.copy()

    def filter(self, zs) -> np.ndarray:
        zs = np.asarray(zs, dtype=float)
        return np.array([self.step(z)[0] for z in zs])
