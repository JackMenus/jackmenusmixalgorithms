"""端到端演示：合成行情 → GP/RL/LLM 联合挖掘 → 样本外筛选 → 因子库 → AlphaModel → 交易信号。

运行：  python examples/demo.py            （离线，使用 MockLLM）
        python examples/demo.py --claude   （需要 anthropic SDK 与 API 凭据，使用 Fable 5.1 max effort）
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd  # noqa: E402

import factormine as fm  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def main(use_claude: bool = False):
    os.makedirs(OUT, exist_ok=True)
    panel = fm.make_synthetic_panel(n_days=500, n_assets=100, seed=0)
    # 接真实数据时：panel = fm.validate_panel({"open": df_open, "close": df_close, ...})

    llm = fm.AnthropicClient(model="claude-fable-5-1", effort="max") if use_claude else fm.MockLLM()
    cfg = fm.MiningConfig(
        gp=fm.GPConfig(population=100, generations=6),
        rl=fm.RLConfig(iterations=10, batch_size=24),
        llm=fm.LLMConfig(rounds=3, per_round=6),
    )
    seeds = ["neg(cs_rank(ts_pct(close, 5)))"]  # 热启动：可放入经典 Alpha101 公式
    lib, report = fm.mine_factors(panel, cfg, llm=llm, library=fm.FactorLibrary(os.path.join(OUT, "alphas.json")),
                                  seeds=seeds)
    lib.save()
    report.to_csv(os.path.join(OUT, "mining_report.csv"), index=False)
    pd.set_option("display.width", 200)
    print(report[["expression", "source", "train_rank_ic", "valid_rank_ic", "passed"]].head(15).to_string())

    # 合成为可上线模型，只用前 70% 拟合权重
    train = {k: v.iloc[:350] for k, v in panel.items()}
    model = fm.AlphaModel(lib, method="max_ir").fit(train)
    model.save(os.path.join(OUT, "alpha_model.json"))

    # 样本外合成信号表现
    comb = model.scores(panel)
    fwd = fm.forward_returns(panel)
    oos = fm.evaluate_factor(comb.iloc[350:], fwd.iloc[350:])
    print("\n合成信号（样本外）:", oos.summary())
    print(f"信号半衰期 ≈ {model.half_life_:.2f} 个 bar")

    # 交给挂单 / 路由模块的信号
    sig = model.signal(panel)
    w = sig.target_weights(gross=1.0, max_weight=0.03)
    with open(os.path.join(OUT, "latest_signal.json"), "w") as f:
        json.dump(sig.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"\n最新信号 {sig.timestamp.date()}  紧迫度={sig.urgency:.2f}  多头 {int((w > 0).sum())} 只 / 空头 {int((w < 0).sum())} 只")
    print("前 5 大多头:", w.nlargest(5).round(4).to_dict())


if __name__ == "__main__":
    main(use_claude="--claude" in sys.argv)
