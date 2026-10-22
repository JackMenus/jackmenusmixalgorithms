"""上下文老虎机执行策略选择器（autoalgo 自主生成，algo_id = bandit_strategy_selector）

协议：StrategySelector；模板族：selector.bandit；注册名：autoalgo.bandit_strategy_selector
没有一种执行策略在所有母单上最优：大单/紧急单适合前置轨迹，小单/低紧急度适合被动赚价差。参考规则是手写阈值，LinUCB 用母单特征作上下文、以实现偏差为奖励在线学习策略选择，长期应不劣于规则并随数据改进。

评估：learned_rounds_shortfall_bps = -1.603（基线 0.5395），通过。

说明文档：bandit_strategy_selector.md（同目录）。本文件由 autoalgo.codegen 生成，参数来自生成管线的搜索结果；
手工修改请改 PARAMS 或在 make(**overrides) 传入覆盖值。
"""

from __future__ import annotations

import json
import os

ALGO_ID = 'bandit_strategy_selector'
PROTOCOL = 'StrategySelector'
FAMILY = 'selector.bandit'
REGISTRY_KIND = 'autoalgo.bandit_strategy_selector'
NAME_ZH = '上下文老虎机执行策略选择器'
GENERATOR_VERSION = '0.1.0'

PARAMS = {
    "arms": [
        "twap",
        "almgren_chriss",
        "pov",
        "adaptive_passive",
        "kalman_pid"
    ],
    "alpha": 0.9194577669928894,
    "ridge": 0.3,
    "reward_scale_bps": 20.0,
    "warmup": 2,
    "epsilon": 0.05
}

BLUEPRINT = {
    "algo_id": "bandit_strategy_selector",
    "name_zh": "上下文老虎机执行策略选择器",
    "protocol": "StrategySelector",
    "family": "selector.bandit",
    "params": {
        "arms": [
            "twap",
            "almgren_chriss",
            "pov",
            "adaptive_passive",
            "kalman_pid"
        ],
        "alpha": 0.9194577669928894,
        "ridge": 0.3,
        "reward_scale_bps": 20.0,
        "warmup": 2,
        "epsilon": 0.05
    },
    "components": [
        {
            "name": "LinUCB",
            "library": "crossquant",
            "path": "crossquant.control.bandits.LinUCB",
            "role": "上下文线性老虎机"
        },
        {
            "name": "make_execution_factory",
            "library": "order_placement",
            "path": "order_placement.make_execution_factory",
            "role": "8 种基础执行策略工厂（被 AutoExecutionFactory 包裹）"
        },
        {
            "name": "choose_strategy",
            "library": "quantstack",
            "path": "quantstack.reference.execution.choose_strategy",
            "role": "确定性执行策略选择规则（冷启动与基线）"
        },
        {
            "name": "TCAEngine",
            "library": "quantstack",
            "path": "quantstack.core.tca.TCAEngine",
            "role": "实现偏差、成交率、maker 比例"
        }
    ],
    "hypothesis": "没有一种执行策略在所有母单上最优：大单/紧急单适合前置轨迹，小单/低紧急度适合被动赚价差。参考规则是手写阈值，LinUCB 用母单特征作上下文、以实现偏差为奖励在线学习策略选择，长期应不劣于规则并随数据改进。",
    "mechanism": [
        "上下文 x = [1, log10(%ADV), urgency, tanh(alpha/10), 价差, 时长, 波动率, %ADV]。",
        "前 warmup 次与 ε 探索沿用 quantstack.reference.execution.choose_strategy，并用其结果训练对应臂（冷启动不劣于规则）。",
        "其余按 LinUCB 选臂：θ_a = A_a⁻¹ b_a，score = θ_aᵀx + α√(xᵀA_a⁻¹x)。",
        "attach(pipeline) 订阅 order_done：reward = −clip(IS_bps, ±3s)/s（未成交按最差计），更新 A_a += xxᵀ, b_a += r·x。"
    ],
    "assumptions": [
        "同一流水线内母单特征与成本的关系平稳；成本以 TCA 实现偏差衡量。",
        "候选策略名存在于执行工厂 available() 中。"
    ],
    "risks": [
        "奖励噪声极大（单路径 IS 噪声可达 20 bps），需要数十笔以上母单才能显著优于规则。",
        "臂集合变化（新策略加入）需重新学习；线性模型不能表达特征交互。"
    ],
    "param_doc": {
        "arms": "候选策略",
        "alpha": "LinUCB 探索系数",
        "ridge": "岭惩罚",
        "reward_scale_bps": "奖励归一化尺度",
        "warmup": "冷启动次数",
        "epsilon": "ε-贪心探索概率"
    },
    "search_space": {
        "alpha": [
            "float",
            0.2,
            1.5
        ],
        "ridge": [
            "logfloat",
            0.3,
            5.0
        ],
        "reward_scale_bps": [
            "choice",
            [
                10.0,
                20.0,
                40.0
            ]
        ],
        "warmup": [
            "choice",
            [
                1,
                2,
                4
            ]
        ],
        "epsilon": [
            "float",
            0.0,
            0.15
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
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.6,
                    "ridge": 1.0,
                    "reward_scale_bps": 20.0,
                    "warmup": 2,
                    "epsilon": 0.05
                },
                "source": "default",
                "objective": "learned_rounds_shortfall_bps",
                "value": -1.438880762359299,
                "score": 1.438880762359299,
                "passed": True,
                "rationale": "",
                "wall_s": 32.0
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.6,
                    "ridge": 0.5223130174068923,
                    "reward_scale_bps": 20.0,
                    "warmup": 2,
                    "epsilon": 0.045948646998889585
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": 1.8737661741801115,
                "score": -1.8737661741801115,
                "passed": False,
                "rationale": "",
                "wall_s": 18.1
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.6299076640905532,
                    "ridge": 1.0,
                    "reward_scale_bps": 20.0,
                    "warmup": 2,
                    "epsilon": 0.05
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": -1.438880762359299,
                "score": 1.438880762359299,
                "passed": True,
                "rationale": "",
                "wall_s": 19.0
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.6299076640905532,
                    "ridge": 1.0,
                    "reward_scale_bps": 40.0,
                    "warmup": 2,
                    "epsilon": 0.01870996759785304
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": 4.896983961409864,
                "score": -4.896983961409864,
                "passed": False,
                "rationale": "",
                "wall_s": 17.3
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.6,
                    "ridge": 0.9883243918773088,
                    "reward_scale_bps": 20.0,
                    "warmup": 2,
                    "epsilon": 0.089673724093005
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": -1.438880762359299,
                "score": 1.438880762359299,
                "passed": True,
                "rationale": "",
                "wall_s": 16.5
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.6299076640905532,
                    "ridge": 1.0,
                    "reward_scale_bps": 20.0,
                    "warmup": 4,
                    "epsilon": 0.05
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": 8.72370101340761,
                "score": -8.72370101340761,
                "passed": False,
                "rationale": "",
                "wall_s": 17.8
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.9194577669928894,
                    "ridge": 0.3,
                    "reward_scale_bps": 20.0,
                    "warmup": 2,
                    "epsilon": 0.05
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": -1.6033286387195098,
                "score": 1.6033286387195098,
                "passed": True,
                "rationale": "",
                "wall_s": 17.2
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.6995168616703978,
                    "ridge": 1.0,
                    "reward_scale_bps": 20.0,
                    "warmup": 2,
                    "epsilon": 0.06113682253208777
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": -1.438880762359299,
                "score": 1.438880762359299,
                "passed": True,
                "rationale": "",
                "wall_s": 14.8
            },
            {
                "params": {
                    "arms": [
                        "twap",
                        "almgren_chriss",
                        "pov",
                        "adaptive_passive",
                        "kalman_pid"
                    ],
                    "alpha": 0.9194577669928894,
                    "ridge": 0.3,
                    "reward_scale_bps": 20.0,
                    "warmup": 1,
                    "epsilon": 0.0
                },
                "source": "evolution",
                "objective": "learned_rounds_shortfall_bps",
                "value": 4.152404898007752,
                "score": -4.152404898007752,
                "passed": False,
                "rationale": "",
                "wall_s": 16.3
            }
        ],
        "selected_score": 1.6033286387195098
    },
    "registry_kind": "autoalgo.bandit_strategy_selector",
    "created_at": "2026-10-22T11:42:06+00:00",
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
    """实例化 上下文老虎机执行策略选择器；overrides 覆盖 PARAMS。"""
    _ensure_siblings()
    from autoalgo.components.selector import BanditStrategySelector

    return BanditStrategySelector(**{**PARAMS, **overrides})


def evaluation():
    """读取同目录的 evaluation.json（若存在）。"""
    p = os.path.join(_HERE, 'evaluation.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)
