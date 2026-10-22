# 自主生成算法索引

由 `autoalgo` 生成管线自动维护。每个算法目录含 `algorithm.py`（可直接 import）、中文说明 `.md`、`blueprint.json` 与 `evaluation.json`。

| 算法 | 协议 | 模板族 | 主目标 | 本算法 | 基线 | 通过 | 生成时间 |
|---|---|---|---|---|---|---|---|
| [`cross_operator_mined_alpha`](cross_operator_mined_alpha/cross_operator_mined_alpha.md) | AlphaModel | `alpha.cross_operator_mined` | `rank_ic` | 0.02217 | 0.03408 | ✓ | 2026-10-22T11:37:57+00:00 |
| [`regime_gated_alpha`](regime_gated_alpha/regime_gated_alpha.md) | AlphaModel | `alpha.regime_gated` | `rank_ic` | 0.02338 | 0.03408 | ✓ | 2026-10-22T11:36:10+00:00 |
| [`signature_alpha`](signature_alpha/signature_alpha.md) | AlphaModel | `alpha.signature` | `rank_ic` | 0.0219 | 0.03408 | ✓ | 2026-10-22T11:36:39+00:00 |
| [`kalman_pid_execution`](kalman_pid_execution/kalman_pid_execution.md) | ExecutionStrategy | `execution.kalman_pid` | `shortfall_bps` | -1.625 | 3.214 | ✓ | 2026-10-22T11:39:17+00:00 |
| [`denoised_riskparity_portfolio`](denoised_riskparity_portfolio/denoised_riskparity_portfolio.md) | PortfolioConstructor | `portfolio.denoised_riskparity` | `sharpe` | 4.036 | 3.94 | ✓ | 2026-10-22T11:38:44+00:00 |
| [`bayesian_venue_router`](bayesian_venue_router/bayesian_venue_router.md) | Router | `router.bayesian_venue` | `shortfall_bps` | 1.858 | 9.438 | ✓ | 2026-10-22T12:03:18+00:00 |
| [`bandit_strategy_selector`](bandit_strategy_selector/bandit_strategy_selector.md) | StrategySelector | `selector.bandit` | `learned_rounds_shortfall_bps` | -1.603 | 0.5395 | ✓ | 2026-10-22T11:42:06+00:00 |
