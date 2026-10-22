# 挂单策略算法库 `order_placement`

全栈量化交易系统中的“挂单 / 执行层”。上游接因子挖掘库的 alpha 信号，下游把子单交给路由延迟优化库发送，并根据路由层回传的延迟画像调整报价。纯 Python + NumPy，无其他依赖。

```
因子挖掘库 ──AlphaSignal──▶  order_placement  ──ChildOrder──▶  路由延迟优化库
                               ▲      │                              │
                               └─Fill─┴──────LatencyProfile──────────┘
```

## 目录

| 模块 | 内容 | 关键文献 |
|---|---|---|
| `interfaces.py` | 跨库数据契约：`BookSnapshot` `AlphaSignal` `LatencyProfile` `ChildOrder` `Fill` `ParentOrder`，协议 `OrderPlacementStrategy` `Router` `AlphaProvider` | — |
| `microstructure/` | 失衡度、加权中间价、单/多层 OFI、积分 OFI、经验微价格、Kyle λ、已实现波动、队列扩散上涨概率；成交强度 λ(δ)=A·e^{−kδ} 估计 | Cont-Kukanov-Stoikov 2014；Xu-Gould-Howison 2019；Cont-Cucuringu-Zhang 2023；Stoikov 2018；Cont-de Larrard 2013 |
| `market_making/` | Avellaneda-Stoikov（含 alpha 漂移）、GLFT 闭式解（含漂移）、自适应 GLFT；统一处理库存上限、post-only、tick 舍入、延迟加宽、微价格参考价 | Avellaneda-Stoikov 2008；Guéant-Lehalle-Fernandez-Tapia 2013；Guéant 2017；Cartea-Jaimungal-Penalva 2015 |
| `execution/` | Almgren-Chriss（含漂移，三对角 QP 求解 + 闭式解校验 + 有效前沿）、Obizhaeva-Wang、任意衰减核的传播子最优切分（指数/幂律核）、TWAP/VWAP/POV；`ScheduledExecutor` 把轨迹落地为被动挂单 + 落后追单 + alpha 加速 | Almgren-Chriss 2000；Obizhaeva-Wang 2013；Bouchaud 等 2004；Gatheral 2010 |
| `queue/` | 队列位置跟踪（概率撤单分配）、排队成交概率、挂单 vs 吃单价值比较、Cont-Kukanov 多交易所市价/限价最优拆分 | Moallemi-Yuan 2017；Cont-Kukanov 2017 |
| `rl/` | Gym 风格执行环境（离散 + 连续状态）、表格 Q-learning 基线，可直接替换为 DDQN/PPO | Nevmyvaka-Feng-Kearns 2006；Ning-Lin-Jaimungal 2021 |
| `simulation/` | 合成 L2 生成器、做市仿真（OU alpha、延迟）、执行仿真（冲击 + 排队） | — |
| `engine.py` | `OrderPlacementEngine`：集成 agent 的统一入口 | — |

更详细的算法综述与前沿方向见 [docs/ALGORITHMS.md](docs/ALGORITHMS.md)，接口契约见 [docs/INTERFACES.md](docs/INTERFACES.md)。

## 快速开始

```bash
pip install numpy pytest
python -m pytest -q tests          # 40 个测试（含 quantstack 适配）
python examples/benchmark.py       # 做市 / 执行 / RL 基准
```

```python
from order_placement import GLFT, MarketMakerConfig, StrategyState, AlphaSignal, LatencyProfile
from order_placement.simulation import make_book

mm = GLFT(MarketMakerConfig(gamma=0.2, sigma=0.02, A=1.0, k=50.0, max_inventory=20))
book = make_book("BTCUSDT", ts=0.0, mid=100.0, tick=0.01, bid_q=10, ask_q=10)
orders = mm.on_book(book, StrategyState(inventory=3),
                    alpha=AlphaSignal("BTCUSDT", 0.0, value=2.0, horizon_s=5.0),   # 5 秒看涨 2bps
                    latency=LatencyProfile(one_way_ms=3.0))
```

母单执行：

```python
from order_placement import ParentOrder, Side, ScheduledExecutor
from order_placement.execution import ACParams, almgren_chriss

parent = ParentOrder("600519.SH", Side.BUY, qty=20000, start_ts=0, end_ts=1800)
n = almgren_chriss(20000, 1800, 30, ACParams(sigma=0.05, eta=2e-4, risk_aversion=1e-6))
executor = ScheduledExecutor(parent, n)       # 每次行情调用 executor.on_book(...)
```

## 基准结果（`examples/benchmark.py`，合成数据，仅用于验证实现方向）

| 做市（600 秒，5 个种子） | PnL 均值 | PnL 标准差 | 平均 \|库存\| |
|---|---|---|---|
| AS（不用 alpha） | 6.28 | 0.29 | 0.59 |
| GLFT（不用 alpha） | 7.47 | 1.87 | 2.06 |
| GLFT + alpha | 26.00 | 11.64 | 11.85 |
| 自适应 GLFT + alpha | 24.76 | 6.66 | 9.40 |

延迟（无 alpha）：单向 1000ms 时不加宽 PnL −12.42，按 σ√RTT 加宽后 4.89；200ms 时 6.44 → 6.77。

执行（买 2000 股 / 200 秒，20 种子）：TWAP 0.86bps（σ 17.5），AC 1.35bps（σ 15.9，用少量期望成本换方差），OW 0.99bps。
RL（可观测漂移）：Q-learning −0.79bps vs TWAP 3.61bps。

> 仿真器是简化模型（无真实撮合、无信息型订单流），数字只说明各模块方向正确，不代表实盘收益。实盘前需用真实 L2/L3 数据（如 LOBSTER、交易所逐笔）重新标定 σ、A、k、η 等参数并在撮合级回测器中验证。
