# 挂单与执行算法综述（含前沿方向）

标注：✅ 已在本库实现　🧩 提供接口/环境，需外部模型（如深度网络）　📌 文献追踪，尚未实现

## 一、做市（Market Making）

| 方法 | 核心思想 | 状态 |
|---|---|---|
| Avellaneda & Stoikov (2008) *High-frequency trading in a limit order book*, QF | 中间价布朗运动 + 指数成交强度，CARA 效用 HJB 近似解：保留价 r = s − qγσ²τ，总价差 γσ²τ + (2/γ)ln(1+γ/k) | ✅ `AvellanedaStoikov` |
| Guéant, Lehalle & Fernandez-Tapia (2013) *Dealing with the inventory risk*, MFE；Guéant (2017) *Optimal market making*, AMF | 库存有界，HJB 化为线性 ODE 组；无限期限渐近闭式解 δ = c1 ± (2q±1)/2·c2，可加漂移项 | ✅ `GLFT`、`glft_coefficients` |
| Cartea, Jaimungal & Penalva (2015) *Algorithmic and High-Frequency Trading*（书） | 加入 alpha 漂移、逆向选择、库存惩罚的统一框架 | ✅ 通过 `alpha_weight` 引入漂移 |
| Cartea, Jaimungal & Ricci (2014) *Buy low, sell high*, SIAM JFM | 自激 Hawkes 订单流下的做市 | 📌 |
| Cartea, Donnelly & Jaimungal (2017) *Algorithmic trading with model uncertainty*, SIAM JFM | 模型不确定（ambiguity aversion）下的稳健做市 | 📌 |
| Bergault & Guéant (2021) *Size matters for OTC market makers*, SIAM JFM | 多资产、多交易规模，降维近似 | 📌 |
| Spooner 等 (2018) *Market making via reinforcement learning*, AAMAS | 表格/线性函数近似 RL 直接学报价 | 🧩 可复用 `simulation` |
| Falces Marín 等 (2022) *A RL approach to improve the performance of the AS market-making algorithm*, PLOS ONE | 用 DQN 动态调整 AS 的 γ 与偏斜 | ✅ 规则版 `AdaptiveGLFT`（EWMA 波动 + 库存相关 γ）；🧩 RL 版 |
| 延迟感知报价（工程实践） | 报价在 RTT 内失效的逆向选择 ≈ σ√RTT，按 z 倍加宽 | ✅ `latency_z` |

## 二、最优执行（Optimal Execution）

| 方法 | 核心思想 | 状态 |
|---|---|---|
| Almgren & Chriss (2000) *Optimal execution of portfolio transactions*, J. Risk | 线性永久+临时冲击，均值-方差，sinh 轨迹与有效前沿 | ✅ `almgren_chriss`（含漂移的三对角 QP）、`almgren_chriss_closed_form`、`efficient_frontier` |
| Obizhaeva & Wang (2013) *Optimal trading strategy and supply/demand dynamics*, JFM | 订单簿指数恢复的瞬时冲击，首尾大单 + 中间匀速 | ✅ `obizhaeva_wang` |
| Bouchaud 等 (2004) 传播子模型；Gatheral (2010) *No-dynamic-arbitrage and market impact*, QF | 冲击按核 G(t) 衰减（幂律最常见），min ½nᵀΓn | ✅ `propagator_optimal`（指数/幂律核，非负约束主动集） |
| TWAP / VWAP / POV | 行业基准 | ✅ |
| Nevmyvaka, Feng & Kearns (2006) *RL for optimized trade execution*, ICML | 表格 Q-learning，状态=剩余时间/库存/盘口 | ✅ `QLearningExecutor` |
| Ning, Lin & Jaimungal (2021) *Double deep Q-learning for optimal execution*, AMF | DDQN，连续状态 | 🧩 `ExecutionEnv` 输出连续向量 |
| *Reinforcement Learning for Trade Execution with Market and Limit Orders*（arXiv:2507.06345, 2025） | RL 同时决定市价与限价比例 | 📌 方向与 `ScheduledExecutor` 的“被动+追单”逻辑一致，可作为 RL 版替换 |

## 三、队列位置与挂单放置

| 方法 | 核心思想 | 状态 |
|---|---|---|
| Cont & de Larrard (2013) *Price dynamics in a Markovian LOB market*, SIAM JFM | 最优档队列为生灭过程；重流量极限下 P(上涨)=1−(2/π)arctan(Qa/Qb) | ✅ `prob_up_diffusion` |
| Huang, Lehalle & Rosenbaum (2015) *Simulating and analyzing order book data: the queue-reactive model*, JASA | 事件强度依赖队列长度 | 📌（可扩展 `simulation`） |
| Moallemi & Yuan (2017) *A model for queue position valuation in a LOB* | 队列靠前的价值 = 成交概率 × 价差捕获 − 逆向选择 | ✅ 简化版 `queue_value` |
| 概率队列模型（开源回测器 hftbacktest 的思路） | 撤单按 f(x)=xⁿ/(xⁿ+(1−x)ⁿ) 分配到我前/后 | ✅ `QueuePositionTracker` |
| Cont & Kukanov (2017) *Optimal order placement in limit order markets*, QF | 多交易所市价/限价拆分，随机近似求解 | ✅ `cont_kukanov_allocation`（与路由库衔接） |

## 四、LOB 微结构信号（亦可回馈因子库）

| 方法 | 状态 |
|---|---|
| OFI：Cont, Kukanov & Stoikov (2014) *The price impact of order book events*, JFEc | ✅ `ofi_step` |
| 多层 OFI：Xu, Gould & Howison (2019) | ✅ `ofi_series` |
| 积分 OFI / 跨资产冲击：Cont, Cucuringu & Zhang (2023) *Cross-impact of order flow imbalance in equity markets*, QF | ✅ `integrated_ofi`（PCA 第一主成分） |
| 微价格：Stoikov (2018) *The micro-price*, QF | ✅ 经验近似 `EmpiricalMicroprice` |
| 深度 LOB 预测：Zhang, Zohren & Roberts (2019) *DeepLOB*, IEEE TSP；Sirignano & Cont (2019) *Universal features of price formation*, QF | 🧩 `feature_vector` 可作输入 |

## 五、仿真与数据生成（训练 RL / 压力测试）

| 方法 | 状态 |
|---|---|
| ABIDES 多智能体市场仿真（Byrd 等 2020） | 📌 |
| JAX-LOB：GPU 加速 LOB 仿真（Frey 等 2023, ICAIF） | 📌 适合大规模 RL |
| 扩散模型生成 LOB：*Painting the market*（arXiv:2509.05107, 2025）；*DiffVolume*（arXiv:2508.08698, 2025） | 📌 用于生成逼真的反事实行情 |

## 六、建议的下一步

1. 用真实 L2/L3 数据标定 σ、A、k（`microstructure.intensity`）、η 与衰减核（`kyle_lambda` + 传播子拟合）。
2. 把 `simulation` 替换为撮合级回测器（如 hftbacktest 或自建 L3 重放），重点验证队列模型与延迟。
3. 在 `ExecutionEnv` / 做市仿真上接入 PPO / DDQN（需要 PyTorch/JAX）。
4. Hawkes 订单流与 queue-reactive 仿真，作为更真实的 RL 训练环境。

> 文献清单以作者知识为主，并辅以少量 2025 年 arXiv 检索；2025–2026 年的最新工作请在使用前再核对原文。
