"""quantstack —— 全栈式量化交易系统的集成层。

把三个并行开发的算法库（因子挖掘 / 挂单策略 / 路由延迟优化）以及跨学科算法库
通过统一契约（``quantstack.contracts``）串联成一条可回测、可模拟、可实盘的流水线，
并由一个 LLM 编排 agent（``quantstack.agent``）负责研究、执行规划、复盘与风控决策。

分层：
    contracts  —— 唯一的跨库数据结构与协议（Protocol）定义
    core       —— 事件总线、时钟、OMS、风控闸门、TCA、流水线
    reference  —— 各层的参考实现（三库交付前即可跑通，交付后可被替换）
    sim        —— 合成行情与多场所撮合/延迟模拟器
    adapters   —— 把兄弟库的原生类型包装成 contracts 协议
    agent      —— 模型路由策略（Fable 5.1 max / Opus 5.5 自适应）、工具、编排器
"""
__version__ = "0.1.0"
