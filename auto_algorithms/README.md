# auto_algorithms —— 自主生成算法库（autoalgo）

在仓库里已有的五个库之上**自动开发新算法**：按 `quantstack` 的协议把 `factormine`（因子挖掘）、`order_placement`（挂单策略）、
`latency_routing`（路由延迟）、`crossquant`（物理 / 数学 / 控制）的组件拼装成新的 AlphaModel / PortfolioConstructor /
ExecutionStrategy / StrategySelector / Router，在搜索空间内提议参数、用统一评估台与基线成对比较、选优后生成：

```
generated/<algo_id>/
├── algorithm.py        可直接 import：make(**overrides) 返回算法实例；PARAMS / BLUEPRINT / REGISTRY_KIND 元数据
├── <algo_id>.md        中文可解释性文档（概述 / 协议层 / 假设 / 组件来源 / 流程公式 / 参数 / 评估 vs 基线 / 风险 / 接入 / 复现血缘）
├── blueprint.json      算法蓝图
└── evaluation.json     评估报告（指标、基线、是否通过）
generated/index.json    索引（quantstack 注册表读取它）；generated/README.md 为索引的可读版本
```

**每生成一个算法，就同时生成一份可解释性 md** —— 这是 `codegen.emit` 的硬性行为，没有"只出代码"的路径。

## 首批算法（7 个模板族）

| 协议层 | algo_id | 一句话 | 建立在 |
|---|---|---|---|
| AlphaModel | `regime_gated_alpha` | DFA-Hurst 识别市场状态，分状态估计因子权重，BOCPD 变点概率收缩置信度 | factormine 合成 + crossquant Hurst / BOCPD |
| AlphaModel | `signature_alpha` | [时间, log 价, log 量] 路径的深度-2 签名特征 + 池化岭回归 | crossquant 路径签名 |
| AlphaModel | `cross_operator_mined_alpha` | 把 6 个跨学科算子注入 factormine 算子表后做 GP 因子挖掘 | factormine GP + crossquant（分数阶差分 / 方差比 / Hurst / 排列熵 / Kalman / Lévy 面积） |
| PortfolioConstructor | `denoised_riskparity_portfolio` | 随机矩阵去噪协方差 → 剥离市场模式后按 μ/特质方差做均值-方差（或 HRP + 信号倾斜）→（可选）分数 Kelly 去杠杆 | crossquant RMT / HRP / Kelly + 参考实现规则 |
| ExecutionStrategy | `kalman_pid_execution`（策略名 `kalman_pid`） | Almgren–Chriss 轨迹 × Kalman 公允价外推 × PID/LQR 进度反馈 | order_placement 执行基类 + crossquant 估计器 / 控制器 |
| StrategySelector | `bandit_strategy_selector` | LinUCB 上下文老虎机按母单特征选执行策略，奖励 = −TCA 实现偏差 | crossquant 老虎机 + quantstack TCA / 事件总线 |
| Router | `bayesian_venue_router` | Thompson 采样倾斜场所份额 + BOCPD 延迟退化监测 的路由叠加层 | latency_routing 适配器 + crossquant |

评估结论（含是否优于基线）见各算法的 md 与 `generated/README.md`。**未通过评估的算法也会写出**（md 中明确标注"不建议上线"），便于复核与再生成。

## 快速开始

```bash
pip install -r requirements.txt            # numpy / pandas / scipy（anthropic 可选）
pytest -q                                   # 组件 + 管线 + generated/ 回归测试

# 生成（默认合成面板 400 日 × 60 资产；--budget quick 几分钟，standard 约半小时）
python -m autoalgo generate --budget quick --proposer evolution --n-candidates 3
python -m autoalgo generate --families alpha.signature execution.kalman_pid --budget standard --seed 1
python -m autoalgo generate --proposer claude --model claude-fable-5-1 --effort max   # 需要 anthropic + API 凭据

python -m autoalgo list                     # 索引
python -m autoalgo show --algo-id kalman_pid_execution
python -m autoalgo rerender                 # 模板改动后只重排代码 / md / 索引，不重新评估
python -m autoalgo evaluate --kind autoalgo.signature_alpha --budget quick
python -m autoalgo families -v              # 模板族与搜索空间
python -m autoalgo components               # 可用组件目录
```

Python 侧：

```python
import sys; sys.path.insert(0, "/mnt/project-files/auto_algorithms")
import autoalgo
from autoalgo.pipeline import GenerationConfig, generate
results = generate(panel, GenerationConfig(budget="quick", n_candidates=3, proposer="evolution", seed=0))
for r in results:
    print(r.summary(), r.paths["markdown"])

# 使用生成的算法
from autoalgo import registry
model = registry.make_alpha_model("autoalgo.regime_gated_alpha")
factory = registry.make_execution_factory("autoalgo.kalman_pid_execution")      # 含挂单库全部策略 + kalman_pid
router = registry.make_router("autoalgo.bayesian_venue_router", venues=["NYSE", "NASDAQ", "ARCA", "IEX"])
```

## 与 quantstack 的接入

`quantstack.core.registry` 把 `auto_algorithms/` 当作第五个兄弟库自动发现，`make_*` 工厂与 `build_sim_pipeline(kinds=...)` 接受：

- `autoalgo.<algo_id>`：`generated/` 中搜索出的参数；
- `autoalgo.family.<family>`：模板族默认参数（不依赖生成结果）；
- `autoalgo.auto`：该协议下第一个评估通过的生成算法。

```python
from quantstack.core.registry import build_sim_pipeline
pipe = build_sim_pipeline(panel, kinds={"alpha": "autoalgo.regime_gated_alpha", "portfolio": "autoalgo.denoised_riskparity_portfolio",
                                        "execution": "autoalgo.kalman_pid_execution", "selector": "autoalgo.bandit_strategy_selector",
                                        "router": "autoalgo.bayesian_venue_router"})
res = pipe.run_cycle(panel)
```

契约细节见 `trading_system/docs/INTERFACES.md` §9。

## 目录

```
auto_algorithms/
├── autoalgo/
│   ├── _paths.py        兄弟库定位（与 quantstack 同一约定，QUANT_PROJECT_ROOT 可覆盖）
│   ├── blueprint.py     AlgorithmBlueprint / EvaluationReport
│   ├── catalog.py       组件目录（可用积木）
│   ├── components/      新算法实现：regime / operators / alpha / portfolio / execution / selector / routing
│   ├── families.py      7 个模板族：构造方式 + 搜索空间 + 可解释性文本骨架
│   ├── search.py        随机 / 进化提议器
│   ├── llm.py           Claude 提议器（claude-fable-5-1，effort=max）与离线替身
│   ├── evaluate.py      评估台（各协议层 vs 基线）
│   ├── explain.py       中文 Markdown 渲染
│   ├── codegen.py       写出 generated/<algo_id>/ 与 index.json
│   ├── pipeline.py      generate(panel, GenerationConfig)
│   ├── registry.py      make_* 工厂与 kind 解析
│   └── cli.py           python -m autoalgo
├── generated/           生成的算法（代码 + md + 蓝图 + 评估）
├── docs/                中文文档（总览 / 生成管线 / 组件目录 / 可解释性文档规范 / 接入 quantstack）
├── examples/            示例脚本
└── tests/
```

## 设计原则

1. **只拼装，不重写**：数值实现尽量复用兄弟库（如执行策略继承挂单库基类、签名特征对照 crossquant 验证到 1e-15），新代码只负责组合与接口。
2. **无未来函数**：因子层 `predict(as_of)` 结构性截断；评估台的快路径 `score_panel` 逐行与截断预测等价（测试覆盖）。
3. **可复现**：`seed` 决定全部随机性；LLM 提议会完整记录在蓝图血缘中（模型、思考强度、原始理由、失败原因）。
4. **基线成对比较**：模拟器评估同种子、同母单对比基线，多种子取均值；通过条件写在报告里。
5. **可解释优先**：md 文档与代码同源（同一份蓝图渲染），参数、假设、风险、组件来源一一对应。

## 模型策略

与 `trading_system/docs/MODEL_POLICY.md` 一致：参数 / 设计提议是强推理阶段，`--proposer claude` 固定使用 `claude-fable-5-1` 并设置
`output_config.effort="max"`；评估、文档渲染、代码生成为确定性代码，不调用模型。无凭据或无网络时自动退回随机提议器并在血缘中注明。
