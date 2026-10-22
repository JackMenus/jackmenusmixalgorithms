"""组合层新算法：``DenoisedRiskParityConstructor``（实现 quantstack ``PortfolioConstructor`` 协议）。

把三件事拼在一起：
  * crossquant.physics.random_matrix.denoise_covariance —— 随机矩阵理论（Marchenko–Pastur / RIE）去噪的协方差；
  * crossquant.physics.networks.hierarchical_risk_parity —— 层次风险平价（HRP）作为"风险配置骨架"；
  * quantstack 参考实现 LongShortConstructor 的多空选股、单票上限、换手门槛与紧急度规则；
  * （可选）crossquant.mathematics.kelly.kelly_multi_asset —— 分数 Kelly 决定总杠杆上限。

权重 = 风险平价权重^(1−tilt) × 信号强度^tilt，再归一到各腿目标敞口并裁剪单票上限。
与 quantstack 流水线配合时需先 ``fit(panel, as_of)``（或 ``update_returns``）喂入历史收益用于估协方差；
没有历史收益时自动退化为按信号加权（等价于参考实现）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .._paths import available, require


def _cap_weights(raw: np.ndarray, cap: float, rounds: int = 10) -> np.ndarray:
    raw = raw.copy()
    for _ in range(rounds):
        over = raw > cap
        if not over.any():
            break
        excess = (raw[over] - cap).sum()
        raw[over] = cap
        under = ~over
        if under.any() and raw[under].sum() > 0:
            raw[under] += excess * raw[under] / raw[under].sum()
        else:
            break
    return raw


class DenoisedRiskParityConstructor:
    """RMT 去噪 + HRP 风险配置 + 信号倾斜 的多空组合构造器。

    参数
      top_k            每腿最多标的数
      gross_target     总敞口 / NAV；net_target 净敞口 / NAV
      max_weight       单票权重上限
      lookback         估协方差用的历史收益长度（bar）
      denoise          协方差去噪方法：rie / clip / lw / none
      allocator        权重配置：mv（去噪协方差 → 剥离市场模式 → 腿内 w ∝ μ/特质方差，可选 β 中性）/ hrp / ivp（逆方差）/ equal；
                       hrp/ivp/equal 为风险权重，再按 tilt 与信号强度几何混合；mv 本身已含信号，不再用 tilt
      neutralize       仅 mv：dollar（按 net_target 分配两腿，默认）/ beta（用多空规模抵消共同因子暴露；单因子合成市场上增益很小且 β 估计有噪声）
      tilt             信号倾斜强度 ∈ [0,1]：0 纯风险平价，1 纯信号加权
      kelly_fraction   >0 时用分数 Kelly 的总杠杆 Σ|f*|·fraction 作为总敞口上限（不超过 gross_target）
      cost_threshold_bps 期望收益绝对值低于此值的信号不交易；默认 0（关闭）。弱 alpha 的期望收益往往只有 2–3 bps/σ，
                       硬门槛会把两腿压到一两只股票，使风险配置失去意义，因此门槛改由搜索空间决定
    """

    def __init__(self, top_k: int = 10, gross_target: float = 1.0, net_target: float = 0.0, max_weight: float = 0.08, lookback: int = 120,
                 denoise: str = "rie", allocator: str = "hrp", tilt: float = 0.5, kelly_fraction: float = 0.0, cost_threshold_bps: float = 0.0, neutralize: str = "dollar",
                 min_trade_weight: float = 0.001, default_participation: float = 0.1, lot_size: int = 1, min_obs: int = 30):
        self.top_k, self.gross_target, self.net_target, self.max_weight = int(top_k), float(gross_target), float(net_target), float(max_weight)
        self.lookback, self.denoise, self.allocator, self.tilt = int(lookback), denoise, allocator, float(np.clip(tilt, 0.0, 1.0))
        self.kelly_fraction, self.cost_threshold_bps, self.neutralize = float(kelly_fraction), float(cost_threshold_bps), str(neutralize)
        self.min_trade_weight, self.default_participation, self.lot_size, self.min_obs = float(min_trade_weight), float(default_participation), int(lot_size), int(min_obs)
        self._returns: Optional[pd.DataFrame] = None
        self.last_diagnostics: Dict[str, float] = {}

    # ---------------------------------------------------------------- 历史数据
    def fit(self, panel, as_of=None, **kwargs) -> None:
        rets = panel["returns"] if "returns" in panel else panel["close"].pct_change(fill_method=None)
        if as_of is not None:
            ts = pd.Timestamp(int(as_of) * 1000, unit="ns") if isinstance(as_of, (int, np.integer)) else pd.Timestamp(as_of)
            rets = rets.loc[:ts]
        self.update_returns(rets)

    def update_returns(self, returns: pd.DataFrame) -> None:
        self._returns = returns.iloc[-self.lookback:].copy()

    # ---------------------------------------------------------------- 协方差与风险配置
    def _covariance(self, symbols: List[str]) -> Optional[np.ndarray]:
        if self._returns is None or len(symbols) < 2:
            return None
        cols = [s for s in symbols if s in self._returns.columns]
        if len(cols) < len(symbols):
            return None
        R = self._returns[cols].dropna(how="all").fillna(0.0).to_numpy(float)
        if R.shape[0] < self.min_obs:
            return None
        if self.denoise in ("rie", "clip", "lw") and available("crossquant"):
            from crossquant.physics.random_matrix import denoise_covariance

            try:
                out = denoise_covariance(R, method=self.denoise)
                info = out.get("info", {})
                self.last_diagnostics = {k: float(v) for k, v in info.items() if isinstance(v, (int, float, np.floating, np.integer))}
                cov = np.asarray(out["cov"], float)
            except Exception:
                cov = np.cov(R, rowvar=False)
        else:
            cov = np.cov(R, rowvar=False)
        cov = cov + 1e-10 * np.eye(len(cols))
        return cov

    def _risk_weights(self, cov: Optional[np.ndarray], n: int) -> np.ndarray:
        if cov is None or n == 1:
            return np.full(n, 1.0 / n)
        if self.allocator == "hrp" and available("crossquant") and n >= 3:
            from crossquant.physics.networks import hierarchical_risk_parity

            try:
                return np.asarray(hierarchical_risk_parity(cov)["weights"], float)
            except Exception:
                pass
        if self.allocator in ("hrp", "ivp"):
            iv = 1.0 / np.clip(np.diag(cov), 1e-12, None)
            return iv / iv.sum()
        return np.full(n, 1.0 / n)

    def _leg(self, leg, gross: float, sign: int, nav_hint: float) -> Dict[str, float]:
        if not leg or gross <= 0:
            return {}
        syms = [s.symbol for s in leg]
        cov = self._covariance(syms)
        w_risk = self._risk_weights(cov, len(syms))
        strength = np.array([abs(s.score) for s in leg], float)
        strength = strength / strength.mean() if strength.mean() > 0 else np.ones_like(strength)
        w = np.power(np.clip(w_risk, 1e-9, None), 1.0 - self.tilt) * np.power(strength, self.tilt)
        w = w / w.sum() * gross
        w = _cap_weights(w, self.max_weight)
        self.last_diagnostics[f"leg{sign:+d}_n"] = float(len(syms))
        self.last_diagnostics[f"leg{sign:+d}_hhi"] = float(np.sum((w / max(w.sum(), 1e-12)) ** 2))
        return {s: float(sign * x) for s, x in zip(syms, w)}

    def _mv_weights(self, longs, shorts, long_gross: float, short_gross: float) -> Optional[Dict[str, float]]:
        """带腿约束的去噪均值-方差：
        1) 在两腿并集上取去噪协方差 Σ̂，最大特征对 (λ₁, v₁) 视为共同因子（市场模式），β̂ = √λ₁·v₁，特质方差 σ²_ε = diag(Σ̂) − β̂²；
        2) 腿内权重 w_i ∝ μ_i / σ²_ε,i（对角均值-方差；不剔除、不反向，保留分散度）；
        3) neutralize="beta" 时用两腿规模抵消共同因子暴露 Σ w_i β̂_i = 0（每腿限制在 [0.35, 0.65]·总敞口），否则按 net_target 分配；
        4) 迭代裁剪单票上限。
        一因子市场里这与完整的 Σ̂⁻¹μ 解等价于同一思想，但不会因对冲项把弱信号标的置零。"""
        sel = list(longs) + list(shorts)
        if len(sel) < 3:
            return None
        syms = [s.symbol for s in sel]
        cov = self._covariance(syms)
        if cov is None:
            return None
        n = len(sel)
        mu = np.array([s.expected_return_bps for s in sel], float)
        if not np.any(np.abs(mu) > 1e-12):
            mu = np.array([s.score for s in sel], float)
        evals, evecs = np.linalg.eigh(cov)
        lam1, v1 = float(evals[-1]), evecs[:, -1]
        if v1.sum() < 0:
            v1 = -v1
        beta = np.sqrt(max(lam1, 0.0)) * v1
        var_idio = np.clip(np.diag(cov) - beta ** 2, 0.2 * np.diag(cov), None)
        raw = np.abs(mu) / var_idio
        sign = np.array([1.0] * len(longs) + [-1.0] * len(shorts))
        gross = long_gross + short_gross
        legs = {+1: sign > 0, -1: sign < 0}
        leg_w = {sg: raw[m] / raw[m].sum() if raw[m].sum() > 0 else np.full(m.sum(), 1.0 / max(m.sum(), 1)) for sg, m in legs.items() if m.any()}
        if self.neutralize == "beta" and len(leg_w) == 2:
            bL = float(leg_w[+1] @ beta[legs[+1]])
            bS = float(leg_w[-1] @ beta[legs[-1]])
            share_long = bS / (bL + bS) if (bL + bS) > 1e-12 else 0.5
            share_long = float(np.clip(share_long, 0.35, 0.65))
            grosses = {+1: gross * share_long, -1: gross * (1.0 - share_long)}
            self.last_diagnostics["beta_long"] = bL
            self.last_diagnostics["beta_short"] = bS
            self.last_diagnostics["long_share"] = share_long
        else:
            grosses = {+1: long_gross, -1: short_gross}
        self.last_diagnostics["lambda1_share"] = float(lam1 / max(np.trace(cov), 1e-18))
        out: Dict[str, float] = {}
        for sg, m in legs.items():
            if sg not in leg_w or grosses.get(sg, 0.0) <= 0:
                continue
            lw = _cap_weights(leg_w[sg] * grosses[sg], self.max_weight)
            leg_syms = [syms[i] for i in np.flatnonzero(m)]
            self.last_diagnostics[f"leg{sg:+d}_n"] = float(len(leg_syms))
            self.last_diagnostics[f"leg{sg:+d}_hhi"] = float(np.sum((lw / max(lw.sum(), 1e-12)) ** 2))
            out.update({sym: float(sg * x) for sym, x in zip(leg_syms, lw) if x > 0})
        return out

    def _kelly_scale(self, longs, shorts, weights: Dict[str, float]) -> float:
        if self.kelly_fraction <= 0 or not available("crossquant"):
            return 1.0
        sel = list(longs) + list(shorts)
        syms = [s.symbol for s in sel]
        cov = self._covariance(syms)
        if cov is None:
            return 1.0
        from crossquant.mathematics.kelly import kelly_multi_asset

        mu = np.array([s.expected_return_bps * 1e-4 for s in sel], float)
        try:
            f = kelly_multi_asset(mu, cov)
        except np.linalg.LinAlgError:
            return 1.0
        lever = float(self.kelly_fraction * np.abs(f).sum())
        gross_now = sum(abs(v) for v in weights.values()) or 1.0
        target = float(np.clip(lever, 0.25 * self.gross_target, self.gross_target))
        self.last_diagnostics["kelly_gross"] = target
        # Kelly 只用于去杠杆：永不把权重放大到超过 gross_target / max_weight 的上限
        return float(min(target / gross_now, 1.0))

    # ---------------------------------------------------------------- 协议
    def construct(self, signals, portfolio, market: Dict, limits) -> List:
        require("quantstack")
        from quantstack.contracts import TargetPosition

        nav = portfolio.nav
        usable = [s for s in signals.signals if s.symbol in market and market[s.symbol].mid > 0 and abs(s.expected_return_bps) >= self.cost_threshold_bps]
        ranked = sorted(usable, key=lambda s: s.score, reverse=True)
        longs = [s for s in ranked if s.score > 0][: self.top_k]
        shorts = [s for s in ranked[::-1] if s.score < 0][: self.top_k] if limits.allow_short else []
        if shorts:
            long_gross = self.gross_target * (1 + self.net_target) / 2
            short_gross = self.gross_target * (1 - self.net_target) / 2
        else:
            long_gross, short_gross = min(self.gross_target, 1.0), 0.0
        self.last_diagnostics = {}
        weights: Optional[Dict[str, float]] = self._mv_weights(longs, shorts, long_gross, short_gross) if self.allocator == "mv" else None
        if weights is None:
            weights = {}
            weights.update(self._leg(longs, long_gross, +1, nav))
            weights.update(self._leg(shorts, short_gross, -1, nav))
        scale = self._kelly_scale(longs, shorts, weights)
        if scale != 1.0:
            weights = {k: v * scale for k, v in weights.items()}
        sig_by_sym = signals.by_symbol()
        targets: List = []
        for sym, pos in portfolio.positions.items():
            if pos.quantity != 0 and sym not in weights and sym in market:
                targets.append(TargetPosition(symbol=sym, target_quantity=0, target_weight=0.0, urgency=0.4, max_participation=self.default_participation, reason="exit: 不再在目标集合"))
        for sym, w in weights.items():
            px = market[sym].mid
            qty = int(round(w * nav / px / self.lot_size)) * self.lot_size
            s = sig_by_sym[sym]
            cur_w = portfolio.quantity(sym) * px / nav if nav else 0.0
            if abs(w - cur_w) < self.min_trade_weight:
                continue
            hl = float(s.metadata.get("half_life_bars", 3.0)) if s.metadata else 3.0
            urgency = float(np.clip(0.25 + 0.4 * s.confidence + 0.35 * min(1.0, 1.0 / max(hl, 0.5)), 0.05, 0.95))
            targets.append(TargetPosition(
                symbol=sym, target_quantity=qty, target_weight=w, urgency=urgency,
                max_participation=min(self.default_participation, limits.max_participation), benchmark="arrival",
                alpha_bps=abs(s.expected_return_bps), alpha_horizon_us=s.horizon_us,
                reason=f"score={s.score:+.2f} er={s.expected_return_bps:+.1f}bps conf={s.confidence:.2f} {self.allocator}/{self.denoise} tilt={self.tilt:.2f}",
            ))
        return targets

    def describe(self) -> Dict[str, object]:
        return {
            "library": "autoalgo", "family": "portfolio.denoised_riskparity", "top_k": self.top_k, "gross_target": self.gross_target,
            "net_target": self.net_target, "max_weight": self.max_weight, "lookback": self.lookback, "denoise": self.denoise,
            "allocator": self.allocator, "tilt": self.tilt, "kelly_fraction": self.kelly_fraction, "neutralize": self.neutralize, "cost_threshold_bps": self.cost_threshold_bps,
            "n_returns": 0 if self._returns is None else int(len(self._returns)), "last_diagnostics": dict(self.last_diagnostics),
        }
