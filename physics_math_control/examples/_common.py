"""示例公用：让示例在未安装包的情况下也能直接运行（把仓库根目录加入 sys.path）。"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def try_plot():
    """返回 matplotlib.pyplot（若可用），否则 None。示例默认把图保存到 examples/output/。"""
    try:
        import warnings
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 容器内可能没有中文字体：缺字形警告无害，这里静默处理；若本机有中文字体会自动使用
        warnings.filterwarnings("ignore", message="Glyph")
        for font in ("Noto Sans CJK SC", "SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei"):
            try:
                from matplotlib import font_manager
                if any(f.name == font for f in font_manager.fontManager.ttflist):
                    plt.rcParams["font.sans-serif"] = [font] + list(plt.rcParams["font.sans-serif"])
                    plt.rcParams["axes.unicode_minus"] = False
                    break
            except Exception:
                pass
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"), exist_ok=True)
        return plt
    except Exception:
        return None


def out_path(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", name)
