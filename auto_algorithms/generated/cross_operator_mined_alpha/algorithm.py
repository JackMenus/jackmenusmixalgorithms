"""跨学科算子遗传规划因子（autoalgo 自主生成，algo_id = cross_operator_mined_alpha）

协议：AlphaModel；模板族：alpha.cross_operator_mined；注册名：autoalgo.cross_operator_mined_alpha
因子挖掘的搜索空间由算子表决定。把物理/数学/控制中的 6 个工具（分数阶差分、方差比、Hurst、排列熵、Kalman 残差、Lévy 面积）改写成面板算子注入 factormine，遗传规划就能搜到传统算子表达不出的因子（如'价量 Lévy 面积为正且方差比显示均值回复'）。

评估：rank_ic = 0.02217（基线 0.03408），通过。

说明文档：cross_operator_mined_alpha.md（同目录）。本文件由 autoalgo.codegen 生成，参数来自生成管线的搜索结果；
手工修改请改 PARAMS 或在 make(**overrides) 传入覆盖值。
"""

from __future__ import annotations

import json
import os

ALGO_ID = 'cross_operator_mined_alpha'
PROTOCOL = 'AlphaModel'
FAMILY = 'alpha.cross_operator_mined'
REGISTRY_KIND = 'autoalgo.cross_operator_mined_alpha'
NAME_ZH = '跨学科算子遗传规划因子'
GENERATOR_VERSION = '0.1.0'

PARAMS = {
    "budget": 60,
    "seed": 0,
    "method": "max_ir",
    "extended_only": True,
    "pool_capacity": 12,
    "max_corr": 0.7,
    "min_ic": 0.011072141231976869
}

BLUEPRINT = {
    "algo_id": "cross_operator_mined_alpha",
    "name_zh": "跨学科算子遗传规划因子",
    "protocol": "AlphaModel",
    "family": "alpha.cross_operator_mined",
    "params": {
        "budget": 60,
        "seed": 0,
        "method": "max_ir",
        "extended_only": True,
        "pool_capacity": 12,
        "max_corr": 0.7,
        "min_ic": 0.011072141231976869
    },
    "components": [
        {
            "name": "Evaluator",
            "library": "factormine",
            "path": "factormine.Evaluator / factormine.parse",
            "role": "因子表达式解析与带缓存求值"
        },
        {
            "name": "fit_weights",
            "library": "factormine",
            "path": "factormine.fit_weights / combine",
            "role": "多因子合成权重（equal/ic/icir/max_ir/ridge）与标准化合成"
        },
        {
            "name": "register",
            "library": "factormine",
            "path": "factormine.register",
            "role": "算子注册入口（扩展算子注入点）"
        },
        {
            "name": "mine_factors",
            "library": "factormine",
            "path": "factormine.mine_factors / GPConfig / MiningConfig",
            "role": "遗传规划因子挖掘（样本外过滤入库）"
        },
        {
            "name": "AlphaModel",
            "library": "factormine",
            "path": "factormine.AlphaModel",
            "role": "因子库 → 合成信号（Grinold 期望收益、IC 半衰期）"
        }
    ],
    "hypothesis": "因子挖掘的搜索空间由算子表决定。把物理/数学/控制中的 6 个工具（分数阶差分、方差比、Hurst、排列熵、Kalman 残差、Lévy 面积）改写成面板算子注入 factormine，遗传规划就能搜到传统算子表达不出的因子（如'价量 Lévy 面积为正且方差比显示均值回复'）。",
    "mechanism": [
        "components.operators.register_all 把扩展算子注册进 factormine.OPERATORS，并为常数参数提供候选值。",
        "factormine.mine_factors（GP：种群 max(20,budget)，代数 max(2,budget//20)）在训练段挖掘，样本外 RankIC 同号且 ≥ min_ic 才入库。",
        "factormine.AlphaModel(library, method) 估合成权重，signal 给出分数、Grinold 期望收益与 IC 半衰期。",
        "describe 报告因子库中含扩展算子的因子比例（跨学科贡献度）。"
    ],
    "assumptions": [
        "训练段足够长（≥ 200 bar）让 GP 的样本外过滤有意义。",
        "扩展算子的窗口/常数候选覆盖了数据的时间尺度。"
    ],
    "risks": [
        "GP 本质是大规模多重检验，样本外过滤只能部分抑制过拟合；因子库应定期用新数据重挖。",
        "ts_hurst / ts_fracdiff 计算较慢，budget 过大时挖掘时间显著增加。",
        "随机种子影响结果；生成文档记录了 seed 以便复现。"
    ],
    "param_doc": {
        "budget": "计算预算（≈ 候选评估次数/10）",
        "seed": "随机种子",
        "method": "合成方法",
        "extended_only": "是否只用基础+扩展算子",
        "pool_capacity": "因子池容量",
        "max_corr": "池内因子最大相关",
        "min_ic": "入池最低 |RankIC|"
    },
    "search_space": {
        "budget": [
            "choice",
            [
                30,
                40,
                60
            ]
        ],
        "method": [
            "choice",
            [
                "max_ir",
                "icir",
                "ridge"
            ]
        ],
        "pool_capacity": [
            "choice",
            [
                8,
                12,
                16
            ]
        ],
        "max_corr": [
            "float",
            0.5,
            0.8
        ],
        "min_ic": [
            "float",
            0.005,
            0.02
        ]
    },
    "lineage": {
        "generator": "autoalgo.pipeline.generate",
        "generator_version": "0.1.0",
        "proposer": "evolution",
        "model": "",
        "effort": "",
        "llm_error": "",
        "seed": 0,
        "budget": "standard",
        "train_ratio": 0.7,
        "n_candidates": 9,
        "panel": {
            "n_days": 600,
            "n_assets": 60,
            "start": "2021-01-04",
            "end": "2023-04-21"
        },
        "candidates": [
            {
                "params": {
                    "budget": 40,
                    "seed": 0,
                    "method": "max_ir",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7,
                    "min_ic": 0.01
                },
                "source": "default",
                "objective": "rank_ic",
                "value": 0.0014805478244267112,
                "score": 0.0014805478244267112,
                "passed": False,
                "rationale": "",
                "wall_s": 5.7
            },
            {
                "params": {
                    "budget": 60,
                    "seed": 0,
                    "method": "max_ir",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7,
                    "min_ic": 0.011072141231976869
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.022173136237409213,
                "score": 0.022173136237409213,
                "passed": True,
                "rationale": "",
                "wall_s": 8.8
            },
            {
                "params": {
                    "budget": 40,
                    "seed": 0,
                    "method": "ridge",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7,
                    "min_ic": 0.01
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.0018878614636902566,
                "score": 0.0018878614636902566,
                "passed": False,
                "rationale": "",
                "wall_s": 6.0
            },
            {
                "params": {
                    "budget": 40,
                    "seed": 0,
                    "method": "ridge",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7,
                    "min_ic": 0.011185366180546025
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.0018878614636902566,
                "score": 0.0018878614636902566,
                "passed": False,
                "rationale": "",
                "wall_s": 6.4
            },
            {
                "params": {
                    "budget": 60,
                    "seed": 0,
                    "method": "ridge",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7,
                    "min_ic": 0.009087033515859764
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.018077957719478257,
                "score": 0.018077957719478257,
                "passed": False,
                "rationale": "",
                "wall_s": 10.3
            },
            {
                "params": {
                    "budget": 60,
                    "seed": 0,
                    "method": "max_ir",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7,
                    "min_ic": 0.009584409046650413
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.022173136237409213,
                "score": 0.022173136237409213,
                "passed": True,
                "rationale": "",
                "wall_s": 10.0
            },
            {
                "params": {
                    "budget": 60,
                    "seed": 0,
                    "method": "ridge",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7,
                    "min_ic": 0.01383745215226613
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.0165464956901436,
                "score": 0.0165464956901436,
                "passed": False,
                "rationale": "",
                "wall_s": 9.4
            },
            {
                "params": {
                    "budget": 60,
                    "seed": 0,
                    "method": "max_ir",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7031217384555931,
                    "min_ic": 0.005
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.00047809680218434457,
                "score": 0.00047809680218434457,
                "passed": False,
                "rationale": "",
                "wall_s": 11.1
            },
            {
                "params": {
                    "budget": 60,
                    "seed": 0,
                    "method": "max_ir",
                    "extended_only": True,
                    "pool_capacity": 12,
                    "max_corr": 0.7602376945505301,
                    "min_ic": 0.011072141231976869
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.022173136237409213,
                "score": 0.022173136237409213,
                "passed": True,
                "rationale": "",
                "wall_s": 10.0
            }
        ],
        "selected_score": 0.022173136237409213
    },
    "registry_kind": "autoalgo.cross_operator_mined_alpha",
    "created_at": "2026-10-22T11:37:57+00:00",
    "generator_version": "0.1.0"
}

_HERE = os.path.dirname(os.path.abspath(__file__))


def _ensure_siblings():
    """确保 autoalgo 及其兄弟库可导入（独立脚本直接 import 本文件时也能工作）。"""
    import sys

    lib_root = os.path.dirname(os.path.dirname(_HERE))
    if lib_root not in sys.path:
        sys.path.insert(0, lib_root)
    from autoalgo._paths import ensure_siblings

    ensure_siblings()


def make(**overrides):
    """实例化 跨学科算子遗传规划因子；overrides 覆盖 PARAMS。"""
    _ensure_siblings()
    from autoalgo.components.alpha import CrossOperatorMinedAlpha

    return CrossOperatorMinedAlpha(**{**PARAMS, **overrides})


def evaluation():
    """读取同目录的 evaluation.json（若存在）。"""
    p = os.path.join(_HERE, 'evaluation.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)
