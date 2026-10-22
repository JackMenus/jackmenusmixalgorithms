"""autoalgo 组件库：建立在五个兄弟库之上的新算法实现（按 quantstack 协议分层）。

- regime      市场状态探测（Hurst / BOCPD）
- operators   跨学科因子算子（注入 factormine 算子表）
- alpha       AlphaModel：RegimeGatedAlpha / SignatureAlpha / CrossOperatorMinedAlpha
- portfolio   PortfolioConstructor：DenoisedRiskParityConstructor
- execution   ExecutionStrategy / Factory：KalmanPIDExecution / AutoExecutionFactory
- selector    StrategySelector：BanditStrategySelector
- routing     Router：BayesianVenueRouter
"""
