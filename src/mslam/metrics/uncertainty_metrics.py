from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UncertaintySummary:
    mean: float
    p95: float
    n_voxels: int


def prob_p1mp(p: np.ndarray) -> np.ndarray:
    """
    Simple uncertainty proxy for Bernoulli probability: p(1-p).

    Range: [0, 0.25], max at p=0.5.
    """

    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 0.0, 1.0)
    return p * (1.0 - p)


def prob_entropy(p: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """
    Binary entropy of probability p:
      H(p) = -p log p - (1-p) log(1-p)

    Range: [0, log(2)] in nats.
    """

    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def variance_across_probs(probs: list[np.ndarray]) -> np.ndarray:
    """
    Voxel-wise variance across multiple probability maps.
    """

    if len(probs) < 2:
        raise ValueError("Need at least 2 probability maps to compute variance.")
    stack = np.stack([np.asarray(p, dtype=np.float32) for p in probs], axis=0)
    stack = np.clip(stack, 0.0, 1.0)
    return np.var(stack, axis=0).astype(np.float32, copy=False)


def summarize_in_mask(x: np.ndarray, mask: np.ndarray, *, p: float = 95.0) -> UncertaintySummary:
    m = (mask > 0)
    vals = np.asarray(x, dtype=np.float32)[m]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return UncertaintySummary(mean=float("nan"), p95=float("nan"), n_voxels=0)
    return UncertaintySummary(mean=float(np.mean(vals)), p95=float(np.percentile(vals, p)), n_voxels=int(vals.size))


def make_boundary_band(
    lesion_mask: np.ndarray,
    *,
    brainmask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    radius_mm: float = 1.0,
) -> np.ndarray:
    """
    Conservative outer boundary band (within brain):
      boundary = dilate(lesion, radius_mm) & ~lesion & brainmask

    This avoids needing a full erosion implementation while still capturing
    the typical high-uncertainty region around lesion edges.
    """

    from mslam.metrics.longitudinal_metrics import dilate_binary_mm

    lesion = (lesion_mask > 0)
    brain = (brainmask > 0)
    if not bool(lesion.any()):
        return np.zeros_like(lesion, dtype=bool)

    if radius_mm <= 0.0:
        return np.zeros_like(lesion, dtype=bool)

    dil = dilate_binary_mm(lesion, spacing_xyz=spacing_xyz, radius_mm=float(radius_mm))
    band = np.logical_and(dil, np.logical_not(lesion))
    band = np.logical_and(band, brain)
    return band


def finite_quantile(values: list[float], q: float) -> float:
    """
    Quantile for a list of floats, ignoring NaNs/infs. Returns NaN if empty.
    """

    vals = np.array([float(v) for v in values], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    q = float(q)
    if not (0.0 <= q <= 1.0) or not math.isfinite(q):
        raise ValueError(f"Invalid quantile q={q}")
    return float(np.quantile(vals, q))

