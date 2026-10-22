"""命令行：python -m autoalgo <命令>

  generate     运行生成管线（--families / --protocols / --budget / --seed / --proposer / --n-candidates / --out）
  list         列出 generated/ 索引
  show         打印某个算法的中文说明（--algo-id）
  evaluate     用当前参数重新评估某个生成算法或模板族（--kind）
  families     列出模板族与搜索空间
  components   列出组件目录
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ._paths import GENERATED_DIR, ensure_siblings


def _p_generate(a):
    from .pipeline import GenerationConfig, default_panel, generate

    cfg = GenerationConfig(families=a.families, protocols=a.protocols, n_candidates=a.n_candidates, budget=a.budget, seed=a.seed, proposer=a.proposer,
                           model=a.model, effort=a.effort, out_dir=a.out, train_ratio=a.train_ratio, rounds=a.rounds)
    panel = default_panel(a.seed, a.n_days, a.n_assets)
    results = generate(panel, cfg)
    for r in results:
        print(r.summary())


def _p_list(a):
    from .codegen import load_index

    for e in load_index(a.out):
        print(f"{e['algo_id']:36s} {e['protocol']:20s} {e['family']:30s} {e.get('objective', ''):28s} {str(e.get('primary')):>12s} {'通过' if e.get('passed') else '未通过'}")


def _p_show(a):
    p = os.path.join(a.out or GENERATED_DIR, a.algo_id, f"{a.algo_id}.md")
    with open(p, encoding="utf-8") as f:
        print(f.read())


def _p_rerender(a):
    from .codegen import rerender

    for algo_id in rerender(a.out, a.algo_id or None):
        print(f"已重排 {algo_id}")


def _p_evaluate(a):
    from .evaluate import evaluate
    from .pipeline import default_panel
    from .registry import resolve

    how, obj = resolve(a.kind)
    if how == "generated":
        from .registry import load_module
        from .families import FAMILIES

        mod = load_module(obj)
        spec, params = FAMILIES[mod.FAMILY], dict(mod.PARAMS)
    else:
        spec, params = obj, dict(obj.default_params)
    rep = evaluate(spec, params, default_panel(a.seed, a.n_days, a.n_assets), budget=a.budget, seed=a.seed)
    print(rep.to_json())


def _p_families(a):
    from .families import FAMILIES

    for f in FAMILIES.values():
        print(f"{f.family:32s} {f.protocol:20s} {f.name_zh}")
        if a.verbose:
            print("   默认参数:", json.dumps(f.default_params, ensure_ascii=False, default=str))
            print("   搜索空间:", json.dumps({k: list(v) for k, v in f.search_space.items()}, ensure_ascii=False, default=str))


def _p_components(a):
    from .catalog import describe_catalog

    print(describe_catalog(a.library))


def main(argv=None) -> int:
    ensure_siblings()
    ap = argparse.ArgumentParser(prog="autoalgo", description="自主生成算法库")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="运行生成管线")
    g.add_argument("--families", nargs="*", default=None)
    g.add_argument("--protocols", nargs="*", default=None)
    g.add_argument("--n-candidates", type=int, default=3)
    g.add_argument("--rounds", type=int, default=2)
    g.add_argument("--budget", default="quick", choices=["quick", "standard", "full"])
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--proposer", default="random", choices=["random", "evolution", "claude", "mock"])
    g.add_argument("--model", default="claude-fable-5-1")
    g.add_argument("--effort", default="max")
    g.add_argument("--train-ratio", type=float, default=0.7)
    g.add_argument("--n-days", type=int, default=400)
    g.add_argument("--n-assets", type=int, default=60)
    g.add_argument("--out", default=None)
    g.set_defaults(fn=_p_generate)
    l = sub.add_parser("list", help="列出已生成算法")
    l.add_argument("--out", default=None)
    l.set_defaults(fn=_p_list)
    s = sub.add_parser("show", help="打印算法的中文说明")
    s.add_argument("--algo-id", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=_p_show)
    r = sub.add_parser("rerender", help="不重新评估，仅由 blueprint.json / evaluation.json 重排代码、中文说明与索引")
    r.add_argument("--algo-id", action="append", default=[], help="只重排指定算法（可多次）")
    r.add_argument("--out", default=None)
    r.set_defaults(fn=_p_rerender)
    e = sub.add_parser("evaluate", help="重新评估")
    e.add_argument("--kind", required=True)
    e.add_argument("--budget", default="quick")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--n-days", type=int, default=400)
    e.add_argument("--n-assets", type=int, default=60)
    e.set_defaults(fn=_p_evaluate)
    f = sub.add_parser("families", help="列出模板族")
    f.add_argument("-v", "--verbose", action="store_true")
    f.set_defaults(fn=_p_families)
    c = sub.add_parser("components", help="列出组件目录")
    c.add_argument("--library", default=None)
    c.set_defaults(fn=_p_components)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
