# 全栈量化交易系统集成层 · quantstack

把 **因子挖掘库（factormine）→ 组合层 → 挂单策略库（order_placement）→ 路由延迟优化库（latency_routing）→ 撮合/网关** 串成一条
可回测、可模拟、可实盘的流水线，并由一个 LLM 编排 agent 在研究、执行规划、复盘、事故四类决策点上介入。
**跨学科算法库（crossquant，物理 / 数学 / 控制）** 以状态估计器（Kalman / OU）、控制器（PID / LQR）、冲击模型（平方根律 / Almgren–Chriss）三类钩子接入执行与路由层。

* 架构：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* 跨库接口契约：[docs/INTERFACES.md](docs/INTERFACES.md)
* 模型路由（Fable 5.1 最高推理 / Opus 5.5 自适应）：[docs/MODEL_POLICY.md](docs/MODEL_POLICY.md)

## 快速开始

```bash
cd /mnt/project-files/trading_system
pip install anthropic pydantic numpy scipy pandas pytest  # anthropic 仅真实模型调用时需要；scipy 为 crossquant 所需

python -m quantstack.agent.cli policy                   # 查看模型路由表
python -m quantstack.agent.cli plugins                  # 列出四个库登记的全部实现
python -m quantstack.agent.cli demo --kinds reference   # 离线演示：参考实现 + 离线决策基线（约 2 秒）
python -m quantstack.agent.cli demo --kinds auto        # 离线演示：自动接入三个兄弟库（约 1 分钟，含因子挖掘）
python -m quantstack.agent.cli demo --kinds auto --llm  # 真实 Claude：需要 ANTHROPIC_API_KEY 或 `ant auth login`
python examples/run_crossquant_hooks.py                 # 跨学科库钩子演示：Kalman 外推 / OU 价差 / PID·LQR 跟踪 / 冲击成本
python -m pytest tests -q                               # 23 个测试（含四库联调，库不可用时自动跳过）
```

Python 内使用：

```python
import sys; sys.path.insert(0, "/mnt/project-files/trading_system")
from quantstack.core.registry import build_sim_pipeline
from quantstack.sim.market import make_synthetic_panel
from quantstack.agent import TradingAgent

panel = make_synthetic_panel(300, 40, seed=0)
pipe = build_sim_pipeline(panel, kinds={"alpha": "auto", "execution": "auto", "router": "auto"})
agent = TradingAgent(pipe)                      # 默认离线 ScriptedBackend；传 backend=ClaudeBackend() 走真实模型
out = agent.run_daily_cycle(panel)
print(out["cycle"]["tca"], out["report"]["headline"])
```

## 目录

```
quantstack/
  contracts/   跨库契约：market / signals / portfolio / orders / risk / tca / protocols / serialization
  core/        clock · events（事件总线+日志）· oms · risk_gate · tca · pipeline（事件循环）· registry（插件发现：四个兄弟库 + 参考实现）
  reference/   参考实现：alpha（反转+量价）· portfolio（多空）· execution（TWAP/Almgren-Chriss/POV）· routing（成本-延迟同步路由）· latency
  sim/         market（日频面板 + 日内多场所 L1 行情/陈旧报价/延迟/冲击）· venue（撮合网关）
  adapters/    latency_routing_adapter（factormine / order_placement / crossquant 直接实现契约，无需适配）
  agent/       model_policy · decisions · backend（Claude / Scripted）· tools · roles · orchestrator · cli
docs/          ARCHITECTURE · INTERFACES · MODEL_POLICY
examples/      run_end_to_end.py · run_agent_offline.py · run_agent_live.py · run_crossquant_hooks.py
tests/         contracts · core · pipeline · agent · siblings（四库联调）
```

## 当前状态（2026-10-22）

| 项 | 状态 |
|---|---|
| 契约层 + 三库接入 | 完成；factormine / order_placement / latency_routing 均按契约实现并在流水线中跑通 |
| 跨学科库 crossquant 接入 | 完成；Kalman / OU 状态估计器、PID / LQR 控制器、平方根律 / Almgren–Chriss 冲击模型按契约登记（`registry.make_state_estimator / make_controller / make_impact_model`），冲击模型已供 agent 成本估算工具使用 |
| 流水线 | 完成；合成市场 10 张母单成交率 100%，加权实施缺口个位数 bps（单路径，见限制） |
| agent | 完成；每日周期 / 事故 / 研究流程，离线基线后端可无凭据运行；真实 Claude 后端按 SDK 2.x 编写，未在本环境联网验证 |
| 实盘网关 / 实时行情 | 接口已定义，未实现 |
| 多种子评估、组合优化器、L2 模拟 | 下一步 |

## 如何替换某一层

实现对应协议（见 INTERFACES.md）后，两种方式：

1. 放进兄弟库并暴露 `make_*` 工厂，`registry.discover()` 自动发现；
2. 直接构造 `TradingPipeline(alpha_model=..., portfolio=..., execution_factory=..., router=..., gateway=..., feed=...)`。

## 安全边界

* LLM 不直接下单：输出结构化决策 → 确定性校验 → 流水线执行。
* 风控是代码：母单/子单/盈亏三道闸门；kill switch 只能由 agent 触发，不能由 agent 解除。
* agent 可调参数只有 `price_band_bps / max_participation / max_order_notional`，且限定区间。
