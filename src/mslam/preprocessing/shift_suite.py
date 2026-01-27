from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShiftSpec:
    """
    A deterministic, parameterized shift definition.

    Notes:
    - Shifts are intended to act on already coregistered images (same grid).
    - Geometric/resolution shifts must be applied synchronously to all modalities,
      and should be resampled back to the original reference grid.
    """

    name: str
    level: int
    params: dict[str, Any]


def stable_seed(*parts: str) -> int:
    """
    Stable, cross-run seed from string parts (independent of Python's hash randomization).
    """

    h = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def brain_percentiles(x: np.ndarray, brainmask: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    bm = brainmask > 0
    vals = x[bm]
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


def _clip_to_brain_range(x: np.ndarray, brainmask: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo, hi = brain_percentiles(x, brainmask, p_lo=p_lo, p_hi=p_hi)
    y = x.astype(np.float32, copy=False)
    return np.clip(y, lo, hi)


def apply_gamma_brightness(
    x: np.ndarray,
    *,
    brainmask: np.ndarray,
    gamma: float,
    brightness_shift: float = 0.0,
    clip_p_lo: float = 1.0,
    clip_p_hi: float = 99.0,
) -> np.ndarray:
    """
    Apply gamma + brightness shift in a percentile-scaled space within the brainmask.

    - We scale x to [0,1] using brainmask percentiles (p_lo, p_hi),
      apply gamma, then add a small brightness shift (in [0,1] units),
      and finally map back and clip to the original percentile range.
    """

    lo, hi = brain_percentiles(x, brainmask, p_lo=clip_p_lo, p_hi=clip_p_hi)
    denom = (hi - lo) if (hi > lo) else 1.0
    xn = (x.astype(np.float32, copy=False) - float(lo)) / float(denom)
    xn = np.clip(xn, 0.0, 1.0)
    xn = np.power(xn, float(gamma), dtype=np.float32)
    if brightness_shift != 0.0:
        xn = xn + float(brightness_shift)
    xn = np.clip(xn, 0.0, 1.0)
    y = xn * float(denom) + float(lo)
    return np.clip(y, lo, hi)


def apply_gaussian_noise(
    x: np.ndarray,
    *,
    brainmask: np.ndarray,
    sigma_rel: float,
    seed: int,
    clip_p_lo: float = 1.0,
    clip_p_hi: float = 99.0,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise, with sigma defined relative to (p_hi - p_lo)
    within the brainmask: sigma = sigma_rel * (p_hi - p_lo).
    """

    lo, hi = brain_percentiles(x, brainmask, p_lo=clip_p_lo, p_hi=clip_p_hi)
    sigma = float(sigma_rel) * float(hi - lo)
    if sigma <= 0:
        return _clip_to_brain_range(x, brainmask, p_lo=clip_p_lo, p_hi=clip_p_hi)
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    y = x.astype(np.float32, copy=False) + noise
    return np.clip(y, lo, hi)


def _resize_linear_1d_axis0(x: np.ndarray, out_n: int) -> np.ndarray:
    """
    Linear resize along axis 0. Input: (n, ...). Output: (out_n, ...).
    """

    n = int(x.shape[0])
    out_n = int(out_n)
    if out_n == n:
        return x.astype(np.float32, copy=False)
    if out_n < 2 or n < 2:
        # Degenerate: nearest
        idx = np.zeros(out_n, dtype=np.int64)
        if out_n > 1 and n > 1:
            idx = np.round(np.linspace(0, n - 1, out_n)).astype(np.int64)
        return x[idx].astype(np.float32, copy=False)

    coords = np.linspace(0.0, float(n - 1), out_n, dtype=np.float32)
    i0 = np.floor(coords).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, n - 1)
    w = (coords - i0.astype(np.float32)).astype(np.float32)

    x0 = x[i0].astype(np.float32, copy=False)
    x1 = x[i1].astype(np.float32, copy=False)
    w = w.reshape((out_n,) + (1,) * (x.ndim - 1))
    return (1.0 - w) * x0 + w * x1


def resize_trilinear(x: np.ndarray, out_shape: tuple[int, int, int]) -> np.ndarray:
    """
    Trilinear resize to out_shape using successive 1D linear resizes.
    """

    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    out_x, out_y, out_z = (int(out_shape[0]), int(out_shape[1]), int(out_shape[2]))
    y = x.astype(np.float32, copy=False)

    # axis 0
    y = _resize_linear_1d_axis0(y, out_x)
    # axis 1: swap axis1 to front
    y = np.swapaxes(y, 0, 1)
    y = _resize_linear_1d_axis0(y, out_y)
    y = np.swapaxes(y, 0, 1)
    # axis 2: swap axis2 to front
    y = np.swapaxes(y, 0, 2)
    y = _resize_linear_1d_axis0(y, out_z)
    y = np.swapaxes(y, 0, 2)
    return y


def apply_resolution_shift_down_up(
    x: np.ndarray,
    *,
    factor_xy: float,
    factor_z: float = 1.0,
) -> np.ndarray:
    """
    Simulate resolution loss by downsampling then upsampling back to original shape.

    This is a purely image-domain operation. The output remains on the original grid.
    """

    if factor_xy <= 1.0 and factor_z <= 1.0:
        return x.astype(np.float32, copy=False)

    nx, ny, nz = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    cx = max(2, int(round(nx / float(factor_xy))))
    cy = max(2, int(round(ny / float(factor_xy))))
    cz = max(2, int(round(nz / float(factor_z))))

    if (cx, cy, cz) == (nx, ny, nz):
        return x.astype(np.float32, copy=False)

    coarse = resize_trilinear(x, (cx, cy, cz))
    back = resize_trilinear(coarse, (nx, ny, nz))
    return back


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = int(math.ceil(3.0 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(xs * xs) / (2.0 * sigma * sigma)).astype(np.float32)
    s = float(k.sum())
    return (k / s) if s > 0 else np.array([1.0], dtype=np.float32)


def _convolve1d_reflect(x: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    kernel = kernel.astype(np.float32, copy=False)
    k = int(kernel.size)
    if k <= 1:
        return x.astype(np.float32, copy=False)
    pad = k // 2
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (pad, pad)
    xp = np.pad(x.astype(np.float32, copy=False), pad_width=pad_width, mode="reflect")
    windows = sliding_window_view(xp, window_shape=k, axis=axis)
    # windows has a trailing axis of size k.
    return np.tensordot(windows, kernel, axes=([-1], [0])).astype(np.float32, copy=False)


def apply_gaussian_blur_mm(
    x: np.ndarray,
    *,
    spacing_xyz: tuple[float, float, float],
    sigma_mm: float,
) -> np.ndarray:
    """
    Gaussian blur with sigma in millimeters (converted to voxel units via spacing).
    """

    sigma_mm = float(sigma_mm)
    if sigma_mm <= 0:
        return x.astype(np.float32, copy=False)

    sx, sy, sz = (float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2]))
    sigmas = (
        sigma_mm / sx if sx > 0 else 0.0,
        sigma_mm / sy if sy > 0 else 0.0,
        sigma_mm / sz if sz > 0 else 0.0,
    )
    y = x.astype(np.float32, copy=False)
    for axis, s in enumerate(sigmas):
        if s <= 0:
            continue
        k = _gaussian_kernel_1d(s)
        y = _convolve1d_reflect(y, k, axis=axis)
    return y


SHIFT_V1: dict[str, list[ShiftSpec]] = {
    # Intensity gamma (+ optional brightness_shift in normalized [0,1] space)
    "gamma": [
        ShiftSpec("gamma", 0, {"gamma": 1.0, "brightness_shift": 0.0}),
        ShiftSpec("gamma", 1, {"gamma": 0.8, "brightness_shift": 0.02}),
        ShiftSpec("gamma", 2, {"gamma": 1.2, "brightness_shift": -0.02}),
    ],
    # Noise sigma relative to (p99 - p1) within brainmask.
    "noise": [
        ShiftSpec("noise", 0, {"sigma_rel": 0.0}),
        ShiftSpec("noise", 1, {"sigma_rel": 0.01}),
        ShiftSpec("noise", 2, {"sigma_rel": 0.03}),
    ],
    # Blur sigma in mm.
    "blur": [
        ShiftSpec("blur", 0, {"sigma_mm": 0.0}),
        ShiftSpec("blur", 1, {"sigma_mm": 0.5}),
        ShiftSpec("blur", 2, {"sigma_mm": 1.0}),
    ],
    # Resolution loss via downsample+upsample. By default only in-plane (XY).
    "resolution": [
        ShiftSpec("resolution", 0, {"factor_xy": 1.0, "factor_z": 1.0}),
        ShiftSpec("resolution", 1, {"factor_xy": 1.5, "factor_z": 1.0}),
        ShiftSpec("resolution", 2, {"factor_xy": 2.0, "factor_z": 1.0}),
    ],
}


def get_shift_spec(shift_name: str, level: int) -> ShiftSpec:
    if shift_name not in SHIFT_V1:
        raise KeyError(f"Unknown shift '{shift_name}'. Available: {', '.join(sorted(SHIFT_V1))}")
    for spec in SHIFT_V1[shift_name]:
        if int(spec.level) == int(level):
            return spec
    raise KeyError(f"Unknown level {level} for shift '{shift_name}'. Levels: {[s.level for s in SHIFT_V1[shift_name]]}")


def apply_shift(
    x: np.ndarray,
    *,
    brainmask: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    shift: str,
    level: int,
    seed: int,
    clip_p_lo: float = 1.0,
    clip_p_hi: float = 99.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply one shift spec to a single 3D image array. Returns (shifted, meta).
    """

    spec = get_shift_spec(shift, level)
    params = dict(spec.params)
    meta: dict[str, Any] = {
        "shift": spec.name,
        "level": int(spec.level),
        "params": params,
        "clip_p_lo": float(clip_p_lo),
        "clip_p_hi": float(clip_p_hi),
        "seed": int(seed),
    }

    if spec.name == "gamma":
        y = apply_gamma_brightness(
            x,
            brainmask=brainmask,
            gamma=float(params["gamma"]),
            brightness_shift=float(params.get("brightness_shift", 0.0)),
            clip_p_lo=clip_p_lo,
            clip_p_hi=clip_p_hi,
        )
        return y, meta

    if spec.name == "noise":
        y = apply_gaussian_noise(
            x,
            brainmask=brainmask,
            sigma_rel=float(params["sigma_rel"]),
            seed=seed,
            clip_p_lo=clip_p_lo,
            clip_p_hi=clip_p_hi,
        )
        return y, meta

    if spec.name == "blur":
        y = apply_gaussian_blur_mm(x, spacing_xyz=spacing_xyz, sigma_mm=float(params["sigma_mm"]))
        y = _clip_to_brain_range(y, brainmask, p_lo=clip_p_lo, p_hi=clip_p_hi)
        return y, meta

    if spec.name == "resolution":
        y = apply_resolution_shift_down_up(
            x,
            factor_xy=float(params["factor_xy"]),
            factor_z=float(params.get("factor_z", 1.0)),
        )
        y = _clip_to_brain_range(y, brainmask, p_lo=clip_p_lo, p_hi=clip_p_hi)
        return y, meta

    raise RuntimeError(f"Unhandled shift '{spec.name}' (this is a bug).")
