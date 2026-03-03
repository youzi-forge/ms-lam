from __future__ import annotations

from pathlib import Path

import numpy as np

from mslam.common.plotting import robust_vmin_vmax


def save_patient_change_figure(
    *,
    patient_id: str,
    flair0: np.ndarray,
    flair1: np.ndarray,
    brain: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    gt: np.ndarray,
    new_mask: np.ndarray,
    diff_norm: np.ndarray,
    out_path: Path,
    title_suffix: str = "",
    scale: str = "per-timepoint",
) -> None:
    import matplotlib.pyplot as plt

    gray = plt.get_cmap("gray").copy()
    gray.set_bad(color="black")

    brain_mask = brain > 0
    lesion0 = mask0 > 0
    lesion1 = mask1 > 0
    gt_mask = gt > 0
    new_proxy = new_mask > 0

    union = gt_mask | new_proxy | lesion0 | lesion1
    if bool(union.any()):
        per_z = union.sum(axis=(0, 1))
        z_idx = int(np.argmax(per_z))
    else:
        per_z = brain_mask.sum(axis=(0, 1))
        z_idx = int(np.argmax(per_z)) if per_z.size else int(flair0.shape[2] // 2)

    flair0_slice = flair0[:, :, z_idx].astype(np.float32, copy=False).copy()
    flair1_slice = flair1[:, :, z_idx].astype(np.float32, copy=False).copy()
    brain_slice = brain_mask[:, :, z_idx]
    flair0_slice[~brain_slice] = np.nan
    flair1_slice[~brain_slice] = np.nan

    values0 = flair0_slice[brain_slice].reshape(-1)
    values1 = flair1_slice[brain_slice].reshape(-1)
    if scale == "shared":
        vmin, vmax = robust_vmin_vmax(np.concatenate([values0, values1]), 1.0, 99.0)
        vmin0, vmax0 = vmin, vmax
        vmin1, vmax1 = vmin, vmax
    else:
        vmin0, vmax0 = robust_vmin_vmax(values0, 1.0, 99.0)
        vmin1, vmax1 = robust_vmin_vmax(values1, 1.0, 99.0)

    diff_slice = diff_norm[:, :, z_idx].astype(np.float32, copy=False).copy()
    diff_slice[~brain_slice] = np.nan
    diff_values = diff_slice[brain_slice].reshape(-1)
    _, diff_vmax = robust_vmin_vmax(diff_values, 1.0, 99.5)

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    axes = fig.subplots(2, 2)
    for axis in axes.reshape(-1):
        axis.set_axis_off()

    axes[0, 0].imshow(flair0_slice.T, cmap=gray, vmin=vmin0, vmax=vmax0, origin="lower", interpolation="nearest")
    if bool(lesion0[:, :, z_idx].any()):
        axes[0, 0].contour(lesion0[:, :, z_idx].T.astype(float), levels=[0.5], colors="red", linewidths=1)
    axes[0, 0].set_title("t0 FLAIR + lesion(mask)", fontsize=12)

    axes[0, 1].imshow(flair1_slice.T, cmap=gray, vmin=vmin1, vmax=vmax1, origin="lower", interpolation="nearest")
    if bool(lesion1[:, :, z_idx].any()):
        axes[0, 1].contour(lesion1[:, :, z_idx].T.astype(float), levels=[0.5], colors="red", linewidths=1)
    axes[0, 1].set_title("t1 FLAIR + lesion(mask)", fontsize=12)

    axes[1, 0].imshow(flair1_slice.T, cmap=gray, vmin=vmin1, vmax=vmax1, origin="lower", interpolation="nearest")
    if bool(new_proxy[:, :, z_idx].any()):
        axes[1, 0].contour(new_proxy[:, :, z_idx].T.astype(float), levels=[0.5], colors="yellow", linewidths=1)
    if bool(gt_mask[:, :, z_idx].any()):
        axes[1, 0].contour(gt_mask[:, :, z_idx].T.astype(float), levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 0].set_title("t1 + new_proxy(yellow) + GT_change(cyan)", fontsize=12)

    image = axes[1, 1].imshow(diff_slice.T, cmap="magma", vmin=0.0, vmax=diff_vmax, origin="lower", interpolation="nearest")
    if bool(gt_mask[:, :, z_idx].any()):
        axes[1, 1].contour(gt_mask[:, :, z_idx].T.astype(float), levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 1].set_title("|norm(t1)-norm(t0)| + GT_change(cyan)", fontsize=12)
    fig.colorbar(image, ax=axes[1, 1], fraction=0.045, pad=0.02)

    fig.suptitle(f"{patient_id} (z={z_idx}){title_suffix}", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

