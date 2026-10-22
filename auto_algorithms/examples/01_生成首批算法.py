"""示例 1：在合成面板上运行生成管线（quick 预算，进化提议器），并打印每个算法的结论与文档路径。

运行：python examples/01_生成首批算法.py [--budget quick|standard] [--proposer evolution|random|claude|mock]
产物写入 auto_algorithms/generated/（默认），每个算法一个目录：algorithm.py + 中文 md + blueprint.json + evaluation.json。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autoalgo  # noqa: E402
from autoalgo.pipeline import GenerationConfig, default_panel, generate  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="quick")
    ap.add_argument("--proposer", default="evolution")
    ap.add_argument("--n-candidates", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    autoalgo.ensure_siblings()
    panel = default_panel(a.seed, 400, 60)
    results = generate(panel, GenerationConfig(budget=a.budget, proposer=a.proposer, n_candidates=a.n_candidates, seed=a.seed, out_dir=a.out))
    print("\n==== 结果 ====")
    for r in results:
        print(r.summary())
        print("   文档:", r.paths.get("markdown"))
