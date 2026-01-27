from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IntensityNormStats:
    p_lo: float
    p_hi: float
    v_lo: float
    v_hi: float
    median: float


def dice_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    a = (a > 0)
    b = (b > 0)
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return 1.0
    if a_sum == 0 or b_sum == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return float((2.0 * inter) / (a_sum + b_sum))


def volume_mm3(mask: np.ndarray, voxel_vol_mm3: float) -> float:
    return float(int((mask > 0).sum()) * float(voxel_vol_mm3))


def percentile_normalize_in_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> tuple[np.ndarray, IntensityNormStats]:
    m = (mask > 0)
    vals = image[m].astype(np.float32, copy=False)
    if vals.size == 0:
        norm = np.zeros_like(image, dtype=np.float32)
        return norm, IntensityNormStats(p_lo=p_lo, p_hi=p_hi, v_lo=0.0, v_hi=1.0, median=0.0)

    v_lo = float(np.percentile(vals, p_lo))
    v_hi = float(np.percentile(vals, p_hi))
    if not (math.isfinite(v_lo) and math.isfinite(v_hi)) or v_hi <= v_lo:
        v_lo = float(np.min(vals))
        v_hi = float(np.max(vals))
        if not (math.isfinite(v_lo) and math.isfinite(v_hi)) or v_hi <= v_lo:
            v_lo, v_hi = 0.0, 1.0

    denom = (v_hi - v_lo) if (v_hi > v_lo) else 1.0
    norm = (image.astype(np.float32, copy=False) - v_lo) / float(denom)
    norm = np.clip(norm, 0.0, 1.0)
    median = float(np.median(vals))
    return norm, IntensityNormStats(p_lo=p_lo, p_hi=p_hi, v_lo=v_lo, v_hi=v_hi, median=median)


def intensity_change_stats(
    flair_t0: np.ndarray,
    flair_t1: np.ndarray,
    brainmask: np.ndarray,
    *,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
    diff_threshold: float = 0.2,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    bm = (brainmask > 0)
    n0, s0 = percentile_normalize_in_mask(flair_t0, bm, p_lo=p_lo, p_hi=p_hi)
    n1, s1 = percentile_normalize_in_mask(flair_t1, bm, p_lo=p_lo, p_hi=p_hi)

    diff = np.abs(n1 - n0)
    vals = diff[bm]
    if vals.size == 0:
        diff_mean, diff_p95, frac = 0.0, 0.0, 0.0
    else:
        diff_mean = float(np.mean(vals))
        diff_p95 = float(np.percentile(vals, 95))
        frac = float(np.mean(vals > float(diff_threshold)))

    t0 = {"median": s0.median, "p_lo": s0.p_lo, "p_hi": s0.p_hi, "v_lo": s0.v_lo, "v_hi": s0.v_hi}
    t1 = {"median": s1.median, "p_lo": s1.p_lo, "p_hi": s1.p_hi, "v_lo": s1.v_lo, "v_hi": s1.v_hi}
    diff_stats = {"mean": diff_mean, "p95": diff_p95, "frac_gt": frac, "frac_threshold": float(diff_threshold)}
    return t0, t1, diff_stats


def _ball_offsets_vox(
    spacing_xyz: tuple[float, float, float],
    radius_mm: float,
) -> list[tuple[int, int, int]]:
    sx, sy, sz = spacing_xyz
    rx = int(math.ceil(radius_mm / sx)) if sx > 0 else 0
    ry = int(math.ceil(radius_mm / sy)) if sy > 0 else 0
    rz = int(math.ceil(radius_mm / sz)) if sz > 0 else 0
    r2 = float(radius_mm * radius_mm)
    offsets: list[tuple[int, int, int]] = []
    for dx in range(-rx, rx + 1):
        for dy in range(-ry, ry + 1):
            for dz in range(-rz, rz + 1):
                d2 = (dx * sx) ** 2 + (dy * sy) ** 2 + (dz * sz) ** 2
                if d2 <= r2 + 1e-6:
                    offsets.append((dx, dy, dz))
    if (0, 0, 0) not in offsets:
        offsets.append((0, 0, 0))
    return offsets


def dilate_binary_mm(
    mask: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    mask = (mask > 0)
    if radius_mm <= 0.0:
        return mask.copy()
    if not bool(mask.any()):
        return mask.copy()

    offsets = _ball_offsets_vox(spacing_xyz, float(radius_mm))
    # Bounding box crop for speed.
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # Expand bbox by the maximum offset in voxels (safe).
    max_abs = np.max(np.abs(np.array(offsets, dtype=np.int64)), axis=0)
    lo = np.maximum(mins - max_abs, 0)
    hi = np.minimum(maxs + max_abs, np.array(mask.shape) - 1)

    sx0, sy0, sz0 = (int(lo[0]), int(lo[1]), int(lo[2]))
    sx1, sy1, sz1 = (int(hi[0]) + 1, int(hi[1]) + 1, int(hi[2]) + 1)
    sub = mask[sx0:sx1, sy0:sy1, sz0:sz1]
    out = np.zeros_like(sub, dtype=bool)

    for dx, dy, dz in offsets:
        src_s = []
        dst_s = []
        for dim, d in zip(sub.shape, (dx, dy, dz), strict=True):
            if d > 0:
                src_s.append(slice(0, dim - d))
                dst_s.append(slice(d, dim))
            elif d < 0:
                src_s.append(slice(-d, dim))
                dst_s.append(slice(0, dim + d))
            else:
                src_s.append(slice(0, dim))
                dst_s.append(slice(0, dim))
        out[tuple(dst_s)] |= sub[tuple(src_s)]

    full = np.zeros_like(mask, dtype=bool)
    full[sx0:sx1, sy0:sy1, sz0:sz1] = out
    return full


def remove_small_components(mask: np.ndarray, *, min_voxels: int, connectivity: int = 26) -> np.ndarray:
    """
    Remove connected components smaller than `min_voxels`.

    Notes:
    - Implemented in pure Python for small lesion masks (sparse); scales with #positive voxels.
    - `connectivity`: 6 or 26.
    """
    if min_voxels <= 1:
        return (mask > 0).copy()

    m = (mask > 0)
    coords = np.argwhere(m)
    if coords.size == 0:
        return m.copy()

    if connectivity == 6:
        neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:
        neigh = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

    remaining = {tuple(c) for c in coords}
    kept = np.zeros_like(m, dtype=bool)

    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = [seed]
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in neigh:
                n = (x + dx, y + dy, z + dz)
                if n in remaining:
                    remaining.remove(n)
                    stack.append(n)
                    comp.append(n)
        if len(comp) >= min_voxels:
            for v in comp:
                kept[v] = True

    return kept

