# crossquant —— 物理 · 数学 · 控制论 → 量化交易 跨学科算法库

> 专门收集 **物理学、数学、控制论** 中可以借鉴到量化交易的算法。每个模块都回答三个问题：
> **它在原学科解决什么问题？在交易里对应什么问题？怎么用（公式 + 代码）？**
>
> 纯 `numpy + scipy` 实现（约 7000 行），45 个单元测试，7 个端到端示例，全部中文文档与 docstring。

- 代码位置：`/mnt/project-files/physics_math_control/`
- Python 包名：`crossquant`
- 兄弟库：因子挖掘 `factormine`、挂单策略 `order_placement`、路由延迟 `latency_routing`、全栈系统 `quantstack`（见 [docs/05_与其他库的接口.md](docs/05_与其他库的接口.md)）

---

## 1. 目录结构

```
physics_math_control/
├── crossquant/
│   ├── control/          控制论：滤波、LQR/MPC、HJB 随机控制、最优执行、做市、变点、多臂老虎机、最优停止、PID、RLS
│   ├── physics/          物理：OU/Langevin、Hawkes、随机矩阵、Hurst/DFA、熵与传递熵、复杂网络/HRP、首达时间、
│   │                           粗糙波动率、MCMC/HMC、模拟退火、PDE 有限差分、Ising 羊群模型
│   ├── mathematics/      数学：最优传输、路径签名、信息几何、SDE/QMC、分数阶差分、时间序列检验、Koopman/DMD/SSA、
│   │                           拓扑数据分析、极值理论、稀疏恢复、Kelly、谱方法/FFT 定价、排队论、Copula、混沌动力系统、
│   │                           流形学习、小波
│   ├── pipelines/        端到端流水线：配对交易、组合风险（去噪 → 配置 → 样本外比较）、市场状态仪表盘
│   ├── adapters.py       对接全栈系统 quantstack 的 StateEstimator / Controller / ImpactModel 实现
│   └── _utils.py
├── docs/                 中文文档（总览、分学科说明、前沿文献索引、接口说明）
├── examples/             7 个可直接运行的示例脚本（图输出到 examples/output/）
├── tests/                pytest 单元测试
├── pyproject.toml / requirements.txt
└── README.md
```

## 2. 安装与运行

```bash
cd /mnt/project-files/physics_math_control
pip install -r requirements.txt            # numpy>=1.24, scipy>=1.10（可选 matplotlib / pandas / pytest）
pip install -e .                           # 或者直接把本目录加入 PYTHONPATH

python -m pytest tests -q                  # 45 passed
```

> **依赖提醒**：`scipy` 是必需依赖。全栈系统 `quantstack` 的 `discover()` 在导入失败时会静默回退到参考实现，若发现 crossquant 的实现没有被登记，先确认 `python -c "import scipy"` 成功。
> `import crossquant` 只加载适配器所需的少数轻量模块（约 0.2 秒），四个子包按需懒加载。

```bash
python examples/01_配对交易_OU_Kalman_首达时间.py
```

## 3. 快速开始

```python
import numpy as np
import crossquant as cq
from crossquant.control import kalman, execution, market_making, changepoint
from crossquant.physics import random_matrix, hawkes, networks
from crossquant.mathematics import optimal_transport, signatures

rng = np.random.default_rng(0)

# (1) 控制论：Kalman 动态对冲比率（时变回归系数的隐状态估计）
x = np.cumsum(rng.normal(size=1000)); y = 1.5 * x + rng.normal(scale=0.5, size=1000)
beta_path = kalman.dynamic_hedge_ratio(y, x, delta=1e-4)["beta"]        # (T, 2)：截距、斜率

# (2) 物理：随机矩阵去噪协方差 → 层次风险平价
R = rng.normal(size=(500, 50)) @ rng.normal(size=(50, 50)) * 0.01
cov = random_matrix.denoise_covariance(R, method="rie")["cov"]           # 'clip' | 'rie' | 'lw'
w = networks.hierarchical_risk_parity(cov)["weights"]

# (3) 控制论：Almgren–Chriss 最优执行与 GLFT 做市报价
ac = execution.almgren_chriss(X=1e6, T=5, N=5, sigma=0.95, eta=2.5e-6, lam=1e-6, gamma=2.5e-7)
print(ac["holdings"], ac["expected_cost"], np.sqrt(ac["variance"]))   # 轨迹、期望成本、成本标准差
q = market_making.glft_closed_form_quotes(q=np.arange(-5, 6), gamma=0.1, sigma=2.0, k=1.5, A=140.0)

# (4) 物理 + 数学：Hawkes 订单流拟合、分布漂移（Wasserstein）、路径签名特征
t = hawkes.simulate_hawkes(mu=1.0, alpha=0.6, beta=1.2, T=2000, seed=1)
fit = hawkes.fit_hawkes(t, T=2000); print(fit["mu"], fit["alpha"], fit["beta"])   # 参数回收
d = optimal_transport.wasserstein_1d(rng.normal(size=500), rng.normal(0.3, 1.2, size=500))
S = signatures.signature(np.c_[np.linspace(0, 1, 50), np.cumsum(rng.normal(size=50))], depth=3)

# (5) 对接全栈系统的协议实现
est = cq.KalmanTrendEstimator(); est.update(100.0, ts_us=0); est.update(100.2, ts_us=1_000_000)
print(est.predict(horizon_us=5_000_000))                                 # 5 秒后的公允价外推
```

## 4. 算法索引（学科 → 算法 → 量化应用 → 接口）

### 4.1 控制论 `crossquant.control`

| 模块 | 算法 | 学科来源 | 量化应用 | 主要接口 |
|---|---|---|---|---|
| `kalman` | 卡尔曼滤波（Joseph 形式）、RTS 平滑、UKF、SV 的 QML 滤波 | 导航/航天状态估计 | 动态对冲比率、公允价/趋势估计、随机波动率滤波 | `KalmanFilter`、`dynamic_hedge_ratio`、`local_level_filter`、`sv_qml_filter`、`UnscentedKalmanFilter` |
| `particle_filter` | 自举粒子滤波（系统重采样）、PMMH | 序贯蒙特卡洛 | 非线性/非高斯隐状态（随机波动率）估计与参数后验 | `ParticleFilter`、`sv_particle_filter`、`pmmh_sv` |
| `lqr` | 离散 LQR/DARE、有限期 Riccati、Gârleanu–Pedersen 动态交易 | 最优控制 | 含交易成本的多资产动态组合、执行跟踪 | `lqr_discrete`、`execution_tracking_lqr`、`GarleanuPedersen` |
| `mpc` | 多期组合 MPC（Boyd 等，SLSQP） | 过程控制 | 多期收益预测 + 交易成本 + 约束的滚动优化 | `MultiPeriodMPC.plan/run` |
| `execution` | Almgren–Chriss、Bertsimas–Lo、Obizhaeva–Wang、瞬时冲击传播子、平方根冲击律 | 二次最优控制 / 线性系统响应 | 最优执行调度、冲击成本估计、实施缺口 | `almgren_chriss(_frontier)`、`obizhaeva_wang`、`optimal_schedule_transient`、`propagator_price_path`、`square_root_impact` |
| `market_making` | Avellaneda–Stoikov、GLFT（闭式 / 精确 expm / 渐近 Perron 向量） | HJB 随机控制 | 库存感知的双边报价、做市仿真 | `avellaneda_stoikov_quotes`、`glft_*_quotes`、`simulate_market_making` |
| `hjb` | Merton 问题闭式解、通用一维 HJB 显式迎风求解器 | 动态规划 / 偏微分方程 | 连续时间最优配置、任意一维控制问题的数值验证 | `merton_optimal_fraction`、`solve_hjb_1d`、`merton_hjb_numeric` |
| `changepoint` | CUSUM 滤波、Page CUSUM、贝叶斯在线变点（BOCPD，Normal–Gamma） | 统计过程控制 / 贝叶斯推断 | 事件采样、结构突变监测、regime 切换概率 | `cusum_filter`、`page_cusum`、`bocpd` |
| `bandits` | UCB1、Thompson 采样、EXP3、LinUCB | 序贯决策 / 对偶控制 | 交易场所选择、策略/参数在线分配 | `UCB1`、`ThompsonGaussian`、`EXP3`、`LinUCB`、`simulate_bandit` |
| `optimal_stopping` | Longstaff–Schwartz LSM、CRR 二叉树、OU 最优进出场（值迭代） | 最优停止理论 | 美式期权、均值回复价差的最优平仓/建仓阈值 | `american_option_lsm`、`crr_binomial_american`、`ou_optimal_exit`、`ou_optimal_entry_exit` |
| `pid` | PID（抗积分饱和） | 经典反馈控制 | 执行进度 / 参与率 / 库存的反馈调节 | `PID`、`simulate_execution_progress` |
| `rls` | 带遗忘因子的递推最小二乘 | 自适应控制 | 在线回归（时变 beta、在线 AR） | `RecursiveLeastSquares`、`adaptive_regression` |

### 4.2 物理 `crossquant.physics`

| 模块 | 算法 | 学科来源 | 量化应用 | 主要接口 |
|---|---|---|---|---|
| `ou_process` | OU/Langevin 过程的模拟、精确 MLE、半衰期、信号回测 | 布朗粒子的 Langevin 方程 | 配对交易 / 统计套利的价差建模 | `simulate_ou`、`fit_ou`、`ou_zscore`、`hedge_ratio_ols`、`ou_signal_backtest` |
| `hawkes` | 单变量/多变量指数核 Hawkes：Ogata 稀疏化模拟、递推 MLE、残差检验 | 地震余震（ETAS） | 订单流聚簇、成交强度预测、跨资产互激 | `simulate_hawkes`、`fit_hawkes`、`hawkes_intensity`、`fit_multivariate_hawkes` |
| `random_matrix` | Marchenko–Pastur、特征值裁剪、RIE（Ledoit–Péché）、Ledoit–Wolf、吸收比 | 核物理能级统计 | 协方差去噪、系统性风险监测 | `mp_bounds`、`clip_eigenvalues`、`rie_cleaning`、`denoise_covariance`、`absorption_ratio` |
| `hurst` | R/S、DFA、MF-DFA、广义 Hurst、Lo–MacKinlay 方差比 | 水文学 / 分形与标度律 | 趋势 vs 均值回复的判别、多重分形风险 | `rescaled_range_hurst`、`dfa`、`mfdfa`、`variance_ratio`、`rolling_hurst` |
| `entropy` | Shannon/Rényi/Tsallis 熵、KSG 互信息、排列/样本/谱熵、传递熵与有效传递熵 | 统计力学 / 信息论 | 市场效率度量、非线性因子筛选、信息流向网络 | `permutation_entropy`、`mutual_information_ksg`、`transfer_entropy`、`transfer_entropy_matrix`、`max_entropy_weights` |
| `networks` | Mantegna 距离、Prim MST、层次风险平价（HRP）、等风险贡献 | 复杂网络 | 资产相关结构、稳健配置 | `minimum_spanning_tree`、`hierarchical_risk_parity`、`risk_parity_weights` |
| `first_passage` | 首达时间密度（逆高斯）、反射原理障碍概率、双障碍、三重障碍 MC | 扩散方程的吸收边界 | 止盈/止损概率、持有期分布、障碍校准 | `bm_first_passage_pdf`、`bm_prob_hit_before_T`、`bm_two_barrier`、`triple_barrier_mc`、`barrier_calibration` |
| `rough_volatility` | 分形布朗运动（Davies–Harte）、Hurst 粗糙度估计、rBergomi 混合方案、fBm 预测 | 分形随机过程 | 粗糙波动率建模与预测、期权定价蒙特卡洛 | `fbm_davies_harte`、`hurst_from_realized_variance`、`rbergomi_simulate`、`fbm_forecast` |
| `mcmc` | Metropolis–Hastings、HMC（leapfrog）、ESS、Gelman–Rubin | 统计物理的哈密顿动力学 | 厚尾收益的贝叶斯参数后验、不确定性量化 | `metropolis_hastings`、`hamiltonian_monte_carlo`、`bayesian_student_t` |
| `annealing` | 模拟退火、并行回火 | 金属退火 / Metropolis 算法 | 基数约束组合、离散策略参数搜索 | `simulated_annealing`、`parallel_tempering`、`cardinality_constrained_portfolio` |
| `pde` | Black–Scholes PDE 有限差分（显式/隐式/CN，美式投影）、Fokker–Planck 密度演化 | 热传导方程 | 期权定价与希腊值、价格分布随时间的演化 | `black_scholes_price`、`bs_pde_fd`、`fokker_planck_fd` |
| `ising` | Bornholdt 自旋模型 | Ising 铁磁模型 | 羊群效应与波动聚簇的生成机制（教学） | `bornholdt_ising` |

### 4.3 数学 `crossquant.mathematics`

| 模块 | 算法 | 学科来源 | 量化应用 | 主要接口 |
|---|---|---|---|---|
| `optimal_transport` | 1D Wasserstein、Sinkhorn（log 域）、Bures W2、Wasserstein DRO、MMD/能量距离 | 最优传输 | 分布漂移监测、分布鲁棒组合优化、两样本检验 | `wasserstein_1d`、`sinkhorn`、`gaussian_w2`、`wasserstein_dro_mean_variance`、`mmd_permutation_test` |
| `signatures` | 截断路径签名（Chen 恒等式）、对数签名、Lévy 面积、lead-lag、签名交易 | 粗糙路径理论 | 路径依赖特征、均值-方差签名策略 | `signature`、`log_signature`、`lead_lag`、`sliding_signature_features`、`SignatureTrading` |
| `information_geometry` | Fisher–Rao 距离、KL/JS/Hellinger/α/Bregman 散度、自然梯度、EG 与 Cover 通用组合 | 信息几何 | 分布变化度量、在线组合选择（镜像下降） | `fisher_rao_gaussian`、`natural_gradient_step`、`exponentiated_gradient`、`universal_portfolio` |
| `sde` | Euler–Maruyama、Milstein、GBM/CIR/Heston/Merton 跳跃、Halton + 布朗桥 QMC、对偶/控制变量 | 随机分析 / 数值 SDE | 情景生成、衍生品蒙特卡洛与方差缩减 | `euler_maruyama`、`simulate_heston`、`qmc_normals`、`control_variate` |
| `fracdiff` | 固定窗分数阶差分（FFD）、最小 d 搜索 | 分数阶微积分 | 平稳化同时保留记忆的特征工程 | `frac_diff_ffd`、`find_min_d` |
| `timeseries` | ADF（MacKinnon 2010 临界值）、Ljung–Box、AR(1) 半衰期 | 时间序列统计 | 协整/平稳性检验（无 statsmodels 依赖） | `adf_test`、`ljung_box`、`ar1_half_life` |
| `koopman` | DMD、Hankel-DMD、SSA 分解与预测 | Koopman 算子理论 / 流体力学 | 线性化非线性动力学、趋势-周期分解 | `dmd`、`hankel_dmd`、`ssa`、`ssa_forecast` |
| `tda` | Vietoris–Rips 持续同调（H0/H1）、持续景观、崩盘指标 | 代数拓扑 | 市场结构突变 / 崩盘预警（Gidea–Katz） | `vietoris_rips_persistence`、`persistence_landscape`、`tda_crash_indicator` |
| `extreme_value` | Hill 估计、GPD 超阈值 VaR/ES、GEV 块极大值、平均超出图 | 极值理论 | 尾部风险度量 | `hill_estimator`、`pot_var_es`、`fit_gev_block_maxima` |
| `sparse` | OMP、LASSO/ISTA（FISTA）、稀疏指数追踪 | 压缩感知 | 因子选择、低换手指数复制 | `omp`、`lasso_ista`、`sparse_index_tracking` |
| `kelly` | 二元/连续/多资产 Kelly、回撤概率、遍历性模拟 | 信息论 / 增长最优 | 仓位规模与杠杆、分数 Kelly 的回撤权衡 | `kelly_continuous`、`kelly_multi_asset`、`drawdown_probability`、`simulate_growth` |
| `spectral` | 周期图、Hilbert 相位、Carr–Madan FFT 定价（BS/Heston 特征函数） | 调和分析 | 周期检测、快速期权定价 | `periodogram`、`dominant_cycles`、`hilbert_phase`、`carr_madan_call` |
| `queueing` | 纯死亡过程成交概率、排队竞速 MC、马尔可夫 LOB 模拟、中间价上涨概率 | 排队论 | 限价单成交概率、挂单 vs 吃单决策 | `fill_probability_pure_death`、`queue_race_mc`、`simulate_markov_lob`、`prob_mid_up` |
| `copulas` | 高斯/t copula（ν 轮廓似然）、尾部相关系数 | 多元统计 | 依赖结构与联合尾部风险 | `gaussian_copula_fit`、`t_copula_fit`、`tail_dependence_t` |
| `dynamical_systems` | Takens 嵌入、相关维数、Rosenstein 最大 Lyapunov 指数、递归量化分析 | 非线性动力学 / 混沌 | 可预测性度量、状态转换的递归结构 | `takens_embedding`、`correlation_dimension`、`largest_lyapunov_rosenstein`、`rqa` |
| `manifold` | 扩散映射、谱聚类 | 流形学习 / 谱图论 | 资产低维嵌入与聚类 | `diffusion_maps`、`spectral_clustering`、`asset_clusters` |
| `wavelets` | Haar/db2 DWT、MODWT、小波去噪、小波方差 / 相关、多分辨率分析 | 调和分析 | 多时间尺度分解、去噪、尺度依赖的相关性 | `dwt`、`modwt`、`wavelet_denoise`、`wavelet_variance`、`multiresolution` |

### 4.4 流水线 `crossquant.pipelines` 与适配器 `crossquant.adapters`

| 接口 | 作用 |
|---|---|
| `pipelines.pairs_trading_pipeline` | 对冲比率（OLS / Kalman）→ ADF 协整检验 → OU 拟合 → z-score 信号回测 → 首达时间持有期分布 |
| `pipelines.eigenportfolio_residuals` | PCA 特征组合残差（Avellaneda–Lee 式统计套利的残差构造） |
| `pipelines.allocation_comparison` | 样本协方差 vs 裁剪 vs RIE vs Ledoit–Wolf × 最小方差 / HRP / 风险平价 / 等权 的样本外比较 |
| `pipelines.regime_dashboard` | 滚动 Hurst、排列熵、Wasserstein 漂移、Fisher–Rao 距离、BOCPD 变点概率 |
| `adapters.KalmanTrendEstimator` / `OUSpreadEstimator` | `StateEstimator` 协议：`update(observation, ts_us)`、`predict(horizon_us)` |
| `adapters.PIDParticipationController` / `LQRTrackingController` | `Controller` 协议：`step(error, dt_us)` |
| `adapters.SquareRootImpactModel` / `AlmgrenChrissImpactModel` | `ImpactModel` 协议：`expected_cost_bps(quantity, snapshot, duration_us)` |
| `adapters.registry_entries()` | 协议名 → {注册名: 类}，可直接登记进 `quantstack/core/registry.py` |

## 5. 示例脚本

| 脚本 | 内容 |
|---|---|
| `examples/01_配对交易_OU_Kalman_首达时间.py` | 协整价差的 OU 拟合、Kalman 动态对冲、首达时间持有期分布、回测 |
| `examples/02_做市_AS_GLFT_随机控制.py` | AS 与 GLFT 报价曲线、精确 / 渐近 / 闭式解对比、做市 PnL 仿真 |
| `examples/03_组合风险_RMT_HRP.py` | Marchenko–Pastur 拟合、特征值裁剪 / RIE、HRP / 风险平价样本外比较 |
| `examples/04_市场状态_Hurst_熵_BOCPD_最优传输.py` | 滚动 Hurst / 排列熵 / Wasserstein / Fisher–Rao / BOCPD 状态仪表盘 |
| `examples/05_最优执行_AC_OW_传播子_LQ.py` | AC 效率前沿、Obizhaeva–Wang 与幂律传播子调度、平方根冲击、Gârleanu–Pedersen |
| `examples/06_路径签名特征与签名交易.py` | 签名 / 对数签名 / Lévy 面积、lead-lag 变换、签名均值-方差策略 |
| `examples/07_Hawkes订单流_排队论成交概率.py` | Hawkes 模拟与 MLE、残差 KS 检验、纯死亡过程成交概率 vs 蒙特卡洛 |

## 6. 测试与数值验证

`python -m pytest tests -q` → 45 passed。除单元测试外，实现过程中对照了以下闭式解 / 已知结果：

- Gârleanu–Pedersen 单资产闭式 aim 组合；Almgren–Chriss 原文参数 κ≈0.61；Obizhaeva–Wang 与广义传播子调度一致；
- Merton 问题：HJB 数值解的最优比例 π* 与闭式解相等；LSM 美式看跌 4.475 vs CRR 4.487；
- Hawkes MLE 参数回收；Marchenko–Pastur 边界；DMD 特征值精确恢复；Carr–Madan 与 BS 闭式误差 < 2e-3；
- 逻辑映射 Lyapunov 指数 0.691 vs ln 2；纯死亡过程成交概率 vs 蒙特卡洛；fBm 方差 / 自相关 vs 理论；
- rBergomi 的 Var(Y_T)≈1 与 S 的鞅性；签名的 Chen 恒等式（深度 4）与圆周路径对数签名面积 = π；
- 小波 DWT 完美重构、MRA 可加性与方差分解。

## 7. 文档

- [docs/00_总览.md](docs/00_总览.md)：设计思路、按“量化问题”反查算法、使用约定、路线图
- [docs/01_控制论.md](docs/01_控制论.md)：控制论模块逐一说明（来源 / 映射 / 公式 / 接口 / 注意事项）
- [docs/02_物理.md](docs/02_物理.md)：物理模块逐一说明
- [docs/03_数学.md](docs/03_数学.md)：数学模块逐一说明
- [docs/04_前沿文献索引.md](docs/04_前沿文献索引.md)：经典 + 2023–2026 前沿文献，按主题分组，并标注本库对应实现
- [docs/05_与其他库的接口.md](docs/05_与其他库的接口.md)：如何被因子挖掘 / 挂单 / 路由 / agent 调用；协议适配器与注册

## 8. 设计原则

1. **先讲物理图像，再给公式与代码**：每个模块 docstring 都写明学科来源与量化映射，便于在因子挖掘、执行、风控中选对工具。
2. **只依赖 numpy / scipy**：无深度学习框架、无 statsmodels；需要神经网络的前沿方向（神经 Hawkes、Koopman-Transformer、PINN、RL 做市）在文献索引中给出路线，但本库提供其经典核心作为基线与校验器。
3. **可验证**：凡有闭式解的问题都提供数值 / 闭式对照（见第 6 节），通用求解器（HJB、PDE、签名、小波）都附带恒等式检验。
4. **约定统一**：时间升序、形状 `(T,)` / `(T, n)`、随机数经 `seed` / `rng` 控制、成本用基点、时间戳微秒整数（适配器层）。

> 免责声明：本库用于研究与教学；示例参数多为合成数据或文献中的玩具参数，实盘前须在真实数据上重新校准并做好风控。
