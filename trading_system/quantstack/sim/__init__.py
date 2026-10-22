"""合成行情 + 多场所撮合/延迟模拟器，用于回测、集成测试与 agent 演示。"""
from .market import DEFAULT_VENUES, SyntheticMarket, VenueSpec, make_synthetic_panel
from .venue import VenueSimulator

__all__ = ["DEFAULT_VENUES", "SyntheticMarket", "VenueSpec", "make_synthetic_panel", "VenueSimulator"]
