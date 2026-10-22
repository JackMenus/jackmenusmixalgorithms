"""端到端演示：喂入延迟样本 → 路由一笔急单 → 对抗仿真对比同步与不同步发送 → 一笔被动单。

运行：python3 examples/demo.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latency_routing import ParentOrder, RoutingService, VenueQuote  # noqa: E402
from latency_routing.sim.latency_arb_sim import SimVenue, simulate_sweep  # noqa: E402

rng = np.random.default_rng(42)
svc = RoutingService()
# 三个场所：同机房(60us)、同城(300us)、异地(900us)，异地带 1% 突刺
for venue, mu, sig in (("NEAR", 60, 0.15), ("MID", 300, 0.10), ("FAR", 900, 0.08)):
    x = rng.lognormal(np.log(mu), sig, 3000)
    if venue == "FAR":
        x[rng.random(3000) < 0.01] *= 8
    svc.latency.observe_many(venue, x)

print("== 延迟画像 ==")
print(json.dumps(svc.latency_report(), indent=2, ensure_ascii=False))

quotes = [
    VenueQuote("NEAR", 99.99, 100.00, 300, 300, taker_fee=0.003, maker_fee=-0.002, quote_update_rate=80),
    VenueQuote("MID", 99.99, 100.00, 400, 400, taker_fee=0.002, maker_fee=-0.001, quote_update_rate=50),
    VenueQuote("FAR", 99.98, 100.00, 500, 500, taker_fee=0.001, maker_fee=0.0, quote_update_rate=30),
]

print("\n== 急单：买 1000，urgency=0.9，因子 alpha=3bps ==")
dec = svc.route(ParentOrder("XYZ", +1, 1000, urgency=0.9, alpha_bps=3.0, order_id="P1"), quotes)
print(json.dumps(dec.to_dict(), indent=2, ensure_ascii=False))

ioc = [c for c in dec.children if c.order_type == "IOC"]
venues = {q.venue: SimVenue(q.venue, q.ask_size, svc.latency.sample(q.venue, 5000), q.quote_update_rate) for q in quotes}
synced = simulate_sweep(ioc, venues, reaction_window_us=150)
for c in ioc:
    c.send_delay_us = 0.0
naive = simulate_sweep(ioc, venues, reaction_window_us=150)
print("\n== 延迟套利对抗仿真（对手反应窗口 150us）==")
print("同步发送:", synced)
print("立即发送:", naive)

print("\n== 被动单：卖 800，urgency=0.1 ==")
print(json.dumps(svc.route(ParentOrder("XYZ", -1, 800, urgency=0.1, order_id="P2"), quotes).to_dict(), indent=2, ensure_ascii=False))
