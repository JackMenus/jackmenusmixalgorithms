"""市场状态门控多因子模型（autoalgo 自主生成，algo_id = regime_gated_alpha）

协议：AlphaModel；模板族：alpha.regime_gated；注册名：autoalgo.regime_gated_alpha
同一组因子在不同市场状态下的有效性不同：均值回复状态（Hurst<0.5）里反转、量价背离因子更强，趋势状态（Hurst>0.5）里动量更强。用物理学中的 DFA-Hurst 指数识别状态、分状态估计合成权重，并在变点刚发生（BOCPD 概率高、状态标签不可靠）时收缩置信度。

评估：rank_ic = 0.02338（基线 0.03408），通过。

说明文档：regime_gated_alpha.md（同目录）。本文件由 autoalgo.codegen 生成，参数来自生成管线的搜索结果；
手工修改请改 PARAMS 或在 make(**overrides) 传入覆盖值。
"""

from __future__ import annotations

import json
import os

ALGO_ID = 'regime_gated_alpha'
PROTOCOL = 'AlphaModel'
FAMILY = 'alpha.regime_gated'
REGISTRY_KIND = 'autoalgo.regime_gated_alpha'
NAME_ZH = '市场状态门控多因子模型'
GENERATOR_VERSION = '0.1.0'

PARAMS = {
    "hurst_window": 180,
    "lo": 0.45,
    "hi": 0.55,
    "hazard": 0.008333333333333333,
    "method": "max_ir",
    "shrink_on_change": 0.4020333830856842,
    "min_regime_obs": 40
}

BLUEPRINT = {
    "algo_id": "regime_gated_alpha",
    "name_zh": "市场状态门控多因子模型",
    "protocol": "AlphaModel",
    "family": "alpha.regime_gated",
    "params": {
        "hurst_window": 180,
        "lo": 0.45,
        "hi": 0.55,
        "hazard": 0.008333333333333333,
        "method": "max_ir",
        "shrink_on_change": 0.4020333830856842,
        "min_regime_obs": 40
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
            "name": "dfa",
            "library": "crossquant",
            "path": "crossquant.physics.hurst.dfa",
            "role": "DFA Hurst 指数：趋势 / 均值回复强度"
        },
        {
            "name": "bocpd",
            "library": "crossquant",
            "path": "crossquant.control.changepoint.bocpd",
            "role": "贝叶斯在线变点概率"
        }
    ],
    "hypothesis": "同一组因子在不同市场状态下的有效性不同：均值回复状态（Hurst<0.5）里反转、量价背离因子更强，趋势状态（Hurst>0.5）里动量更强。用物理学中的 DFA-Hurst 指数识别状态、分状态估计合成权重，并在变点刚发生（BOCPD 概率高、状态标签不可靠）时收缩置信度。",
    "mechanism": [
        "用 factormine 求值 K 个因子表达式，并计算未来 h 期收益。",
        "等权市场收益 → 滚动 DFA Hurst（crossquant）→ 按阈值 lo/hi 标注 均值回复 / 中性 / 趋势。",
        "每个状态子样本上用 factormine.fit_weights(method) 估合成权重；样本不足回退全局权重。",
        "预测时按当前状态取权重合成分数 z，期望收益 E[r]=IC_state·σ_idio·z（Grinold）。",
        "BOCPD 给出'最近发生变点'的概率 p，置信度乘 (1−shrink·p)。"
    ],
    "assumptions": [
        "市场状态可由 Hurst 指数在 60–180 bar 窗口内识别，且状态持续时间长于窗口更新步长。",
        "各状态内因子权重稳定，训练期每个状态有足够样本（min_regime_obs）。",
        "面板为日频、横截面 ≥ 20 个资产，字段齐全（open/high/low/close/volume/returns）。"
    ],
    "risks": [
        "状态识别滞后：Hurst 估计需要较长窗口，状态切换初期会用错权重（变点收缩只能部分缓解）。",
        "分状态估计样本更少，权重估计噪声更大；过细的阈值（lo/hi 太靠近 0.5）会放大噪声。",
        "合成数据上的改善未必迁移到真实市场；应在真实面板上重新运行生成管线复核。"
    ],
    "param_doc": {
        "hurst_window": "滚动 Hurst 窗口（bar）",
        "lo": "均值回复阈值（H<lo）",
        "hi": "趋势阈值（H>hi）",
        "hazard": "BOCPD 先验变点概率",
        "method": "合成方法",
        "shrink_on_change": "变点概率对置信度的收缩强度",
        "min_regime_obs": "状态样本下限"
    },
    "search_space": {
        "hurst_window": [
            "choice",
            [
                60,
                90,
                120,
                180
            ]
        ],
        "lo": [
            "float",
            0.4,
            0.49
        ],
        "hi": [
            "float",
            0.51,
            0.6
        ],
        "hazard": [
            "logfloat",
            0.0025,
            0.025
        ],
        "method": [
            "choice",
            [
                "max_ir",
                "icir",
                "ridge",
                "equal"
            ]
        ],
        "shrink_on_change": [
            "float",
            0.0,
            0.9
        ],
        "min_regime_obs": [
            "choice",
            [
                30,
                40,
                60
            ]
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
                    "hurst_window": 120,
                    "lo": 0.45,
                    "hi": 0.55,
                    "hazard": 0.008333333333333333,
                    "method": "max_ir",
                    "shrink_on_change": 0.5,
                    "min_regime_obs": 40
                },
                "source": "default",
                "objective": "rank_ic",
                "value": 0.02109338255039808,
                "score": 0.02109338255039808,
                "passed": True,
                "rationale": "",
                "wall_s": 4.4
            },
            {
                "params": {
                    "hurst_window": 120,
                    "lo": 0.45,
                    "hi": 0.55,
                    "hazard": 0.00895960346620016,
                    "method": "max_ir",
                    "shrink_on_change": 0.5,
                    "min_regime_obs": 30
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.02109338255039808,
                "score": 0.02109338255039808,
                "passed": True,
                "rationale": "",
                "wall_s": 4.3
            },
            {
                "params": {
                    "hurst_window": 120,
                    "lo": 0.45650871098837076,
                    "hi": 0.55,
                    "hazard": 0.008333333333333333,
                    "method": "max_ir",
                    "shrink_on_change": 0.5,
                    "min_regime_obs": 40
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.021278101769423845,
                "score": 0.021278101769423845,
                "passed": True,
                "rationale": "",
                "wall_s": 3.8
            },
            {
                "params": {
                    "hurst_window": 120,
                    "lo": 0.45,
                    "hi": 0.55,
                    "hazard": 0.005125050902575196,
                    "method": "ridge",
                    "shrink_on_change": 0.5,
                    "min_regime_obs": 40
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.019608177938937103,
                "score": 0.019608177938937103,
                "passed": True,
                "rationale": "",
                "wall_s": 3.6
            },
            {
                "params": {
                    "hurst_window": 120,
                    "lo": 0.41465815704487174,
                    "hi": 0.55,
                    "hazard": 0.008333333333333333,
                    "method": "max_ir",
                    "shrink_on_change": 0.5074386762825038,
                    "min_regime_obs": 40
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.015513309873475096,
                "score": 0.015513309873475096,
                "passed": False,
                "rationale": "",
                "wall_s": 3.9
            },
            {
                "params": {
                    "hurst_window": 180,
                    "lo": 0.45,
                    "hi": 0.55,
                    "hazard": 0.008333333333333333,
                    "method": "max_ir",
                    "shrink_on_change": 0.4020333830856842,
                    "min_regime_obs": 40
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.023381106794097058,
                "score": 0.023381106794097058,
                "passed": True,
                "rationale": "",
                "wall_s": 3.7
            },
            {
                "params": {
                    "hurst_window": 120,
                    "lo": 0.45,
                    "hi": 0.5687652406499683,
                    "hazard": 0.008333333333333333,
                    "method": "max_ir",
                    "shrink_on_change": 0.5,
                    "min_regime_obs": 40
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.015931178896682968,
                "score": 0.015931178896682968,
                "passed": False,
                "rationale": "",
                "wall_s": 3.6
            },
            {
                "params": {
                    "hurst_window": 60,
                    "lo": 0.45650871098837076,
                    "hi": 0.55,
                    "hazard": 0.008333333333333333,
                    "method": "max_ir",
                    "shrink_on_change": 0.5,
                    "min_regime_obs": 40
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": -0.000707521176739037,
                "score": -0.000707521176739037,
                "passed": False,
                "rationale": "",
                "wall_s": 3.3
            },
            {
                "params": {
                    "hurst_window": 120,
                    "lo": 0.45,
                    "hi": 0.55,
                    "hazard": 0.008333333333333333,
                    "method": "max_ir",
                    "shrink_on_change": 0.4020333830856842,
                    "min_regime_obs": 40
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.02109338255039808,
                "score": 0.02109338255039808,
                "passed": True,
                "rationale": "",
                "wall_s": 3.4
            }
        ],
        "selected_score": 0.023381106794097058
    },
    "registry_kind": "autoalgo.regime_gated_alpha",
    "created_at": "2026-10-22T11:36:10+00:00",
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
    """实例化 市场状态门控多因子模型；overrides 覆盖 PARAMS。"""
    _ensure_siblings()
    from autoalgo.components.alpha import RegimeGatedAlpha

    return RegimeGatedAlpha(**{**PARAMS, **overrides})


def evaluation():
    """读取同目录的 evaluation.json（若存在）。"""
    p = os.path.join(_HERE, 'evaluation.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)
