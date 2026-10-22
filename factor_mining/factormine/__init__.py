"""factormine —— 前沿因子挖掘算法库。

模块一览：
  data        行情面板格式、合成数据
  operators   算子注册表（时序/横截面/逐元素）
  expression  公式表达式解析与带缓存求值
  evaluate    IC / RankIC / ICIR / 分层 / 换手 / 衰减
  library     因子池（挖掘期）与因子库（持久化）
  gp          遗传规划挖掘（AutoAlpha / AlphaEvolve 思路）
  rl          强化学习挖掘（AlphaGen / QuantFactor REINFORCE 思路）
  llm         LLM 智能体挖掘（Alpha-GPT / AlphaAgent / FactorMiner 思路）
  neural      神经网络因子（GKX 2020 / FactorVAE 等的接口样板）
  combine     多因子合成、动态合成（AlphaForge）、正交化
  interface   AlphaModel / AlphaSignal：对接挂单、路由与全栈 agent 的契约
  pipeline    一站式挖掘流水线
  integration quantstack 契约适配（make_alpha_model / make_miner）
"""
from .combine import DynamicCombiner, combine, fit_weights, orthogonalize, standardize
from .data import Panel, forward_returns, make_synthetic_panel, split_panel, validate_panel
from .evaluate import FactorReport, evaluate_factor, ic_decay, ic_series
from .expression import Evaluator, Expr, ParseError, parse
from .gp import GPConfig, GPMiner
from .interface import AlphaModel, AlphaSignal
from .integration import ContractAlphaModel, ContractFactorMiner, make_alpha_model, make_miner
from .library import FactorLibrary, FactorPool, FactorRecord
from .llm import AnthropicClient, LLMAlphaAgent, LLMConfig, MockLLM
from .neural import MLPConfig, NeuralFactor
from .operators import OPERATORS, register
from .pipeline import MiningConfig, mine_factors
from .rl import RLConfig, RLMiner

__version__ = "0.1.0"
