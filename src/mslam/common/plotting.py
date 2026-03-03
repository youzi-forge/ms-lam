from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np


def setup_matplotlib_env(repo_root: Path, *, loky_safe: bool = False) -> None:
    """Keep matplotlib cache local and optionally cap loky core detection noise."""

    mpl_cache = repo_root / "results" / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))
    if loky_safe:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)


def robust_vmin_vmax(values: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi
