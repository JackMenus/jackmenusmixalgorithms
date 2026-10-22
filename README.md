# jackmenusmixalgorithms

量化算法交易算法库合集：四个独立算法库 + 一个把它们串成全栈交易系统的集成层（agent）。

| 目录 | 包名 | 内容 |
|---|---|---|
| `factor_mining/` | `factormine` | 因子挖掘（GP / RL / LLM / 神经网络因子、因子评估与组合） |
| `quant/order_placement/` | `order_placement` | 下单与执行（TWAP/VWAP/POV/Almgren-Chriss/Obizhaeva-Wang/做市/RL 执行） |
| `routing_latency/` | `latency_routing` | 延迟感知的智能路由 |
| `physics_math_control/` | `crossquant` | 物理 / 数学 / 控制论交叉算法库 |
| `trading_system/` | `quantstack` | 集成层与 LLM agent，统一契约见 `trading_system/docs/INTERFACES.md` |

## 快速开始

```bash
pip install -r requirements.txt

# 各库测试
(cd factor_mining && pytest -q)
(cd quant/order_placement && pytest -q)
(cd routing_latency && pytest -q)
(cd physics_math_control && pytest -q)
(cd trading_system && pytest -q)

# 端到端离线演示（自动发现同级的四个库）
cd trading_system && python -m quantstack.agent.cli demo --kinds auto
```

`quantstack` 默认在 `trading_system/` 的上一级目录寻找兄弟库，可用环境变量 `QUANT_PROJECT_ROOT` 覆盖。
各库的详细说明见各自目录下的 `README.md` 与 `docs/`。
