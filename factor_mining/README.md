# factormine：前沿因子挖掘算法库

量化交易全栈系统中的**因子层**。它把几条前沿的因子挖掘路线放进同一个框架，产出可持久化的因子库和标准化的交易信号，供挂单策略库、路由延迟库和全栈 agent 使用。

核心依赖只有 `numpy` 和 `pandas`。`anthropic` 是可选依赖，只在调用 Claude 时需要。

## 包含的算法

| 路线 | 模块 | 参考工作 |
|---|---|---|
| 公式算子体系（43 个算子） | `operators.py` | Alpha101、国泰君安 191、Qlib |
| 遗传规划 | `gp.py` | gplearn、AutoAlpha、AlphaEvolve |
| 强化学习（RPN + REINFORCE，以池合成 IC 增量为奖励） | `rl.py` | AlphaGen、QuantFactor REINFORCE、Alpha² |
| LLM 智能体（想法 → 公式 → 校验 → 回测 → 反馈，带经验记忆） | `llm.py` | Alpha-GPT、AlphaAgent、Alpha Jungle、FactorMiner、R&D-Agent-Quant |
| 神经网络因子（直接最大化横截面相关） | `neural.py` | GKX 2020、FactorVAE、MASTER |
| 因子合成、动态合成、正交化 | `combine.py` | Qian–Hua 最大化 IC_IR、AlphaForge、Löwdin 正交 |
| 评估 | `evaluate.py` | IC/RankIC/ICIR、分层、多空、换手、衰减 |

各路线的原理与文献见 [docs/前沿算法综述.md](docs/前沿算法综述.md)，对外接口见 [docs/接口规范.md](docs/接口规范.md)。

## 快速开始

```bash
cd /mnt/project-files/factor_mining
python tests/test_factormine.py      # 10 项测试，约 10 秒
python examples/demo.py              # 端到端演示（离线 MockLLM），约 2 分钟
python examples/demo.py --claude     # 用 Claude Fable 5.1（effort=max）做 LLM 挖掘
```

```python
import factormine as fm

panel = fm.make_synthetic_panel()          # 或 fm.validate_panel({"close": ..., "volume": ...})
lib, report = fm.mine_factors(panel, fm.MiningConfig(methods=("gp", "rl", "llm")), llm=fm.MockLLM())
model = fm.AlphaModel(lib, method="max_ir").fit(panel)
sig = model.signal(panel)
weights = sig.target_weights(gross=1.0, max_weight=0.03)
```

## 设计要点

1. **一个池，多个挖掘器。** GP、RL、LLM 都往同一个 `FactorPool` 提交候选。入池要求与池内因子的相关性不超过 0.7，池满时淘汰最弱的因子，所以最后得到的是一组互补的因子，而不是一堆同质因子。
2. **防止未来信息泄漏。** 算子只看历史数据（有测试检查截断前后结果一致）。适应度只在训练段计算，入库前要求样本外 IC 的方向与样本内一致。动态合成只使用已经实现的 IC。
3. **信号自带执行提示。** 信号带有半衰期和紧迫度，执行层可以据此决定挂单要多激进、路由的延迟预算是多少。
4. **模型分工。** 需要强推理的 LLM 环节（提出假设、精修公式）默认用 `claude-fable-5-1`、effort=max。批量改写可以用 `AnthropicClient(model="claude-opus-5-5", effort="medium")`。

## 目录

```
factormine/        库代码
  integration.py   quantstack 契约适配（make_alpha_model / make_miner）
docs/              中文综述与接口规范
examples/demo.py   端到端演示
tests/             单元测试
output/            演示产出（alphas.json、alpha_model.json、latest_signal.json、mining_report.csv）
```

## 已知局限

- 自带数据是合成数据，植入了短期反转和放量两个信号。接真实数据后需要重新调参，并加上交易成本、涨跌停和停牌处理。
- RL 策略是线性 softmax，换成 LSTM 或 Transformer 需要 PyTorch。神经网络因子只是基线实现。
- AlphaAgent 的“假设与公式一致性”评审和 Alpha Jungle 的 MCTS 精修还没有实现，接口已经留好。
