import os
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autoalgo._paths import ensure_siblings  # noqa: E402

STATUS = ensure_siblings()
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    import numpy as _np

    warnings.filterwarnings("ignore", category=_np.exceptions.RankWarning)
except Exception:  # pragma: no cover
    pass


def need(*libs):
    missing = [l for l in libs if not STATUS.get(l)]
    return pytest.mark.skipif(bool(missing), reason=f"兄弟库不可用: {missing}")


@pytest.fixture(scope="session")
def panel():
    from quantstack.sim.market import make_synthetic_panel

    return make_synthetic_panel(260, 24, seed=7)


@pytest.fixture(scope="session")
def small_panel():
    from quantstack.sim.market import make_synthetic_panel

    return make_synthetic_panel(160, 10, seed=11)


TINY_BUDGET = dict(alpha_stride=5, portfolio_every=5, exec_orders=2, exec_seeds=1, exec_duration_s=240, selector_rounds=2, selector_orders=2, selector_seeds=1,
                   router_orders=2, router_seeds=1, router_duration_s=240, step_s=2.0, mc_samples=100)
