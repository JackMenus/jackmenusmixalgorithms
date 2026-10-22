"""路径签名因子模型（autoalgo 自主生成，algo_id = signature_alpha）

协议：AlphaModel；模板族：alpha.signature；注册名：autoalgo.signature_alpha
粗路径理论中的截断签名是路径的'万能特征'：一阶项是净变化（动量/反转），二阶项（含 Lévy 面积）刻画价量之间谁领先、路径是凸还是凹。把每个资产近 window 根 bar 的 [时间, log 价格, log 成交量] 路径签名作为特征，截面标准化后与未来收益做池化岭回归。

评估：rank_ic = 0.0219（基线 0.03408），通过。

说明文档：signature_alpha.md（同目录）。本文件由 autoalgo.codegen 生成，参数来自生成管线的搜索结果；
手工修改请改 PARAMS 或在 make(**overrides) 传入覆盖值。
"""

from __future__ import annotations

import json
import os

ALGO_ID = 'signature_alpha'
PROTOCOL = 'AlphaModel'
FAMILY = 'alpha.signature'
REGISTRY_KIND = 'autoalgo.signature_alpha'
NAME_ZH = '路径签名因子模型'
GENERATOR_VERSION = '0.1.0'

PARAMS = {
    "window": 30,
    "ridge": 1.1745295787389274,
    "channels": [
        "logprice",
        "logvolume"
    ],
    "normalize": True
}

BLUEPRINT = {
    "algo_id": "signature_alpha",
    "name_zh": "路径签名因子模型",
    "protocol": "AlphaModel",
    "family": "alpha.signature",
    "params": {
        "window": 30,
        "ridge": 1.1745295787389274,
        "channels": [
            "logprice",
            "logvolume"
        ],
        "normalize": True
    },
    "components": [
        {
            "name": "signature",
            "library": "crossquant",
            "path": "crossquant.mathematics.signatures.signature / add_time",
            "role": "路径签名（深度 2）特征，Lévy 面积"
        }
    ],
    "hypothesis": "粗路径理论中的截断签名是路径的'万能特征'：一阶项是净变化（动量/反转），二阶项（含 Lévy 面积）刻画价量之间谁领先、路径是凸还是凹。把每个资产近 window 根 bar 的 [时间, log 价格, log 成交量] 路径签名作为特征，截面标准化后与未来收益做池化岭回归。",
    "mechanism": [
        "构造路径 X_t = [t/(w−1), log close, log(1+volume)]，每通道增量按窗口标准差归一（只保留形状）。",
        "按 Chen 恒等式向量化计算深度-2 签名：S¹ = ΣΔ，S² = Σ (Σ_{i<k}Δ_i)⊗Δ_k + ½ΣΔ_k⊗Δ_k（与 crossquant.signature 等价）。",
        "特征逐日截面 z 标准化、裁剪到 ±5；标签为截面标准化的未来收益。",
        "池化岭回归 β = (XᵀX + λnI)⁻¹Xᵀy；预测分数 z = cs_z(X_T β)。"
    ],
    "assumptions": [
        "价量路径形状与未来短期收益之间存在线性可分的关系，且在训练期与预测期稳定。",
        "窗口内 bar 数 ≥ 10；成交量为正。"
    ],
    "risks": [
        "二阶签名特征之间高度相关，λ 太小会过拟合；λ 太大退化为一阶动量/反转。",
        "特征对极端成交量敏感（已用 log1p 与裁剪缓解）。",
        "线性读出忽略了签名的高阶交互；深度 >2 会使特征维数指数增长。"
    ],
    "param_doc": {
        "window": "路径窗口长度（bar）",
        "ridge": "岭惩罚（相对样本量 ×1e-3）",
        "channels": "路径通道",
        "normalize": "增量是否按窗口标准差归一"
    },
    "search_space": {
        "window": [
            "choice",
            [
                10,
                15,
                20,
                30,
                40
            ]
        ],
        "ridge": [
            "logfloat",
            0.1,
            30.0
        ],
        "channels": [
            "choice",
            [
                [
                    "logprice"
                ],
                [
                    "logprice",
                    "logvolume"
                ]
            ]
        ],
        "normalize": [
            "choice",
            [
                True,
                False
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
                    "window": 20,
                    "ridge": 1.0,
                    "channels": [
                        "logprice",
                        "logvolume"
                    ],
                    "normalize": True
                },
                "source": "default",
                "objective": "rank_ic",
                "value": 0.007090734390837926,
                "score": 0.007090734390837926,
                "passed": False,
                "rationale": "",
                "wall_s": 3.3
            },
            {
                "params": {
                    "window": 20,
                    "ridge": 1.1745295787389274,
                    "channels": [
                        "logprice",
                        "logvolume"
                    ],
                    "normalize": True
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.007281662659242714,
                "score": 0.007281662659242714,
                "passed": False,
                "rationale": "",
                "wall_s": 3.5
            },
            {
                "params": {
                    "window": 20,
                    "ridge": 1.0,
                    "channels": [
                        "logprice",
                        "logvolume"
                    ],
                    "normalize": False
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.004305354839410703,
                "score": 0.004305354839410703,
                "passed": False,
                "rationale": "",
                "wall_s": 3.2
            },
            {
                "params": {
                    "window": 20,
                    "ridge": 1.0,
                    "channels": [
                        "logprice"
                    ],
                    "normalize": False
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.007779938871908862,
                "score": 0.007779938871908862,
                "passed": False,
                "rationale": "",
                "wall_s": 2.9
            },
            {
                "params": {
                    "window": 20,
                    "ridge": 0.801086464777667,
                    "channels": [
                        "logprice",
                        "logvolume"
                    ],
                    "normalize": True
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.007212742211135619,
                "score": 0.007212742211135619,
                "passed": False,
                "rationale": "",
                "wall_s": 3.3
            },
            {
                "params": {
                    "window": 40,
                    "ridge": 1.1745295787389274,
                    "channels": [
                        "logprice",
                        "logvolume"
                    ],
                    "normalize": True
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.018440255750743918,
                "score": 0.018440255750743918,
                "passed": False,
                "rationale": "",
                "wall_s": 3.7
            },
            {
                "params": {
                    "window": 20,
                    "ridge": 13.336805823447289,
                    "channels": [
                        "logprice",
                        "logvolume"
                    ],
                    "normalize": False
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.004783451641595042,
                "score": 0.004783451641595042,
                "passed": False,
                "rationale": "",
                "wall_s": 3.1
            },
            {
                "params": {
                    "window": 20,
                    "ridge": 0.5844031351515403,
                    "channels": [
                        "logprice"
                    ],
                    "normalize": True
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.008191288393268768,
                "score": 0.008191288393268768,
                "passed": False,
                "rationale": "",
                "wall_s": 3.1
            },
            {
                "params": {
                    "window": 30,
                    "ridge": 1.1745295787389274,
                    "channels": [
                        "logprice",
                        "logvolume"
                    ],
                    "normalize": True
                },
                "source": "evolution",
                "objective": "rank_ic",
                "value": 0.021901490327077198,
                "score": 0.021901490327077198,
                "passed": True,
                "rationale": "",
                "wall_s": 3.2
            }
        ],
        "selected_score": 0.021901490327077198
    },
    "registry_kind": "autoalgo.signature_alpha",
    "created_at": "2026-10-22T11:36:39+00:00",
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
    """实例化 路径签名因子模型；overrides 覆盖 PARAMS。"""
    _ensure_siblings()
    from autoalgo.components.alpha import SignatureAlpha

    return SignatureAlpha(**{**PARAMS, **overrides})


def evaluation():
    """读取同目录的 evaluation.json（若存在）。"""
    p = os.path.join(_HERE, 'evaluation.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)
