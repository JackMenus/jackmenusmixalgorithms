"""Thompson 采样 + 变点监测的场所路由叠加层（autoalgo 自主生成，algo_id = bayesian_venue_router）

协议：Router；模板族：router.bayesian_venue；注册名：autoalgo.bayesian_venue_router
延迟感知路由器按费用与延迟分位数做静态最优拆分，但场所的真实成交质量（滑点、隐藏流动性、拒单）只能事后观察，并会随时间漂移。把每个场所当成一条臂，用高斯 Thompson 采样在'探索冷门场所'与'利用便宜场所'间权衡；再用 BOCPD 监测延迟分布突变，退化场所降权或改投。

评估：shortfall_bps = 1.858（基线 9.438），通过。

说明文档：bayesian_venue_router.md（同目录）。本文件由 autoalgo.codegen 生成，参数来自生成管线的搜索结果；
手工修改请改 PARAMS 或在 make(**overrides) 传入覆盖值。
"""

from __future__ import annotations

import json
import os

ALGO_ID = 'bayesian_venue_router'
PROTOCOL = 'Router'
FAMILY = 'router.bayesian_venue'
REGISTRY_KIND = 'autoalgo.bayesian_venue_router'
NAME_ZH = 'Thompson 采样 + 变点监测的场所路由叠加层'
GENERATOR_VERSION = '0.1.0'

PARAMS = {
    "mix": 0.5,
    "obs_var_bps2": 4.28760031969563,
    "prior_var": 25.0,
    "min_share": 0.05,
    "max_share": 0.8,
    "latency_window": 40,
    "degrade_hazard": 0.05,
    "degrade_threshold": 0.5,
    "degrade_penalty": 0.6
}

BLUEPRINT = {
    "algo_id": "bayesian_venue_router",
    "name_zh": "Thompson 采样 + 变点监测的场所路由叠加层",
    "protocol": "Router",
    "family": "router.bayesian_venue",
    "params": {
        "mix": 0.5,
        "obs_var_bps2": 4.28760031969563,
        "prior_var": 25.0,
        "min_share": 0.05,
        "max_share": 0.8,
        "latency_window": 40,
        "degrade_hazard": 0.05,
        "degrade_threshold": 0.5,
        "degrade_penalty": 0.6
    },
    "components": [
        {
            "name": "bocpd",
            "library": "crossquant",
            "path": "crossquant.control.changepoint.bocpd",
            "role": "贝叶斯在线变点概率"
        },
        {
            "name": "ThompsonGaussian",
            "library": "crossquant",
            "path": "crossquant.control.bandits.ThompsonGaussian",
            "role": "高斯 Thompson 采样（场所选择）"
        },
        {
            "name": "make_router（经",
            "library": "latency_routing",
            "path": "latency_routing.make_router（经 quantstack.adapters.latency_routing_adapter）",
            "role": "延迟感知 SOR：扫单 + Cont-Kukanov 拆分 + 同步到达"
        }
    ],
    "hypothesis": "延迟感知路由器按费用与延迟分位数做静态最优拆分，但场所的真实成交质量（滑点、隐藏流动性、拒单）只能事后观察，并会随时间漂移。把每个场所当成一条臂，用高斯 Thompson 采样在'探索冷门场所'与'利用便宜场所'间权衡；再用 BOCPD 监测延迟分布突变，退化场所降权或改投。",
    "mechanism": [
        "底层路由器（latency_routing 适配器）给出基准分配 q_base。",
        "吃单切片：对每个场所从后验 N(m_v, v_v) 采样，平移为正后归一得 w_TS；最终份额 = (1−mix)·q_base/Q + mix·w_TS，裁剪到 [min_share, max_share] 与 1.3× 可见量。",
        "成交回报：奖励 = −(side·(成交价−路由时中间价)/中间价 + 费用)·1e4 bps，更新对应场所后验；拒单计 −reject_penalty。",
        "每个场所保留最近 latency_window 个回报延迟，log 标准化后 BOCPD；变点概率 > 阈值且延迟上行 → 退化：吃单份额 ×(1−penalty)，被动单改投后验均值最高的正常场所。"
    ],
    "assumptions": [
        "场所质量差异在同一会话内持续足够长，使得几十笔成交就能学到。",
        "Fill.latency_us 与 fee 由网关正确填写（§8）。"
    ],
    "risks": [
        "Thompson 采样会在早期把量导向噪声中偶然'便宜'的场所；prior_var 与 obs_var 控制探索速度。",
        "份额重分配可能超过某场所可见量导致部分未成交（已限 1.3×），底层的同步到达计划在重分配后仍沿用。",
        "BOCPD 对延迟突刺敏感，latency_window 太短会误判。"
    ],
    "param_doc": {
        "mix": "向 Thompson 权重倾斜的比例",
        "obs_var_bps2": "奖励观测方差",
        "prior_var": "先验方差",
        "min_share": "单场所最小份额",
        "max_share": "单场所最大份额",
        "latency_window": "延迟样本窗口",
        "degrade_hazard": "BOCPD 先验变点概率",
        "degrade_threshold": "退化判定阈值",
        "degrade_penalty": "退化场所份额惩罚"
    },
    "search_space": {
        "mix": [
            "float",
            0.2,
            0.9
        ],
        "obs_var_bps2": [
            "logfloat",
            1.0,
            16.0
        ],
        "prior_var": [
            "choice",
            [
                9.0,
                25.0,
                100.0
            ]
        ],
        "max_share": [
            "choice",
            [
                0.6,
                0.8,
                1.0
            ]
        ],
        "degrade_threshold": [
            "float",
            0.3,
            0.8
        ],
        "degrade_penalty": [
            "float",
            0.3,
            1.0
        ],
        "latency_window": [
            "choice",
            [
                20,
                40,
                80
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
                    "mix": 0.5,
                    "obs_var_bps2": 4.0,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.5,
                    "degrade_penalty": 0.6
                },
                "source": "default",
                "objective": "shortfall_bps",
                "value": 3.5463162730695714,
                "score": -3.5463162730695714,
                "passed": True,
                "rationale": "",
                "wall_s": 216.0
            },
            {
                "params": {
                    "mix": 0.651222877755304,
                    "obs_var_bps2": 4.0,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.5,
                    "degrade_penalty": 0.6
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 7.596486847801365,
                "score": -7.596486847801365,
                "passed": True,
                "rationale": "",
                "wall_s": 118.3
            },
            {
                "params": {
                    "mix": 0.5,
                    "obs_var_bps2": 4.28760031969563,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.5,
                    "degrade_penalty": 0.6
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 1.858486028167909,
                "score": -1.858486028167909,
                "passed": True,
                "rationale": "",
                "wall_s": 126.8
            },
            {
                "params": {
                    "mix": 0.5,
                    "obs_var_bps2": 4.0,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.35146248157364546,
                    "degrade_penalty": 0.5983853713638778
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 4.83611428890703,
                "score": -4.83611428890703,
                "passed": True,
                "rationale": "",
                "wall_s": 117.9
            },
            {
                "params": {
                    "mix": 0.3339992272992673,
                    "obs_var_bps2": 4.28760031969563,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.5,
                    "degrade_penalty": 0.6
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 6.942373475198412,
                "score": -6.942373475198412,
                "passed": True,
                "rationale": "",
                "wall_s": 132.0
            },
            {
                "params": {
                    "mix": 0.5,
                    "obs_var_bps2": 4.28760031969563,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.6816475940881144,
                    "degrade_penalty": 0.566938351820588
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 2.2002722268358577,
                "score": -2.2002722268358577,
                "passed": True,
                "rationale": "",
                "wall_s": 139.2
            },
            {
                "params": {
                    "mix": 0.5,
                    "obs_var_bps2": 4.0,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.5086619262988542,
                    "degrade_penalty": 0.6
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 3.5463162730695714,
                "score": -3.5463162730695714,
                "passed": True,
                "rationale": "",
                "wall_s": 142.6
            },
            {
                "params": {
                    "mix": 0.5,
                    "obs_var_bps2": 4.28760031969563,
                    "prior_var": 100.0,
                    "min_share": 0.05,
                    "max_share": 1.0,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.6816475940881144,
                    "degrade_penalty": 0.566938351820588
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 2.610185169597242,
                "score": -2.610185169597242,
                "passed": True,
                "rationale": "",
                "wall_s": 140.0
            },
            {
                "params": {
                    "mix": 0.5066741823696351,
                    "obs_var_bps2": 4.28760031969563,
                    "prior_var": 25.0,
                    "min_share": 0.05,
                    "max_share": 0.8,
                    "latency_window": 40,
                    "degrade_hazard": 0.05,
                    "degrade_threshold": 0.5,
                    "degrade_penalty": 0.6
                },
                "source": "evolution",
                "objective": "shortfall_bps",
                "value": 8.648428316906688,
                "score": -8.648428316906688,
                "passed": True,
                "rationale": "",
                "wall_s": 138.9
            }
        ],
        "selected_score": -1.858486028167909
    },
    "registry_kind": "autoalgo.bayesian_venue_router",
    "created_at": "2026-10-22T12:03:18+00:00",
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


def make(base=None, venues=None, **overrides):
    """实例化 Thompson 采样 + 变点监测的场所路由叠加层；overrides 覆盖 PARAMS。"""
    _ensure_siblings()
    from autoalgo.components.routing import BayesianVenueRouter

    if base is None:
        from quantstack.core.registry import make_router as _mk

        base = _mk('auto', list(venues or ()))
    return BayesianVenueRouter(base, list(venues or getattr(base, 'venues', [])), **{**PARAMS, **overrides})


def evaluation():
    """读取同目录的 evaluation.json（若存在）。"""
    p = os.path.join(_HERE, 'evaluation.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)
