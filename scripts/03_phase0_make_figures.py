from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_manifest_checked(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("ok", "False") == "True"]


def _robust_range(img: np.ndarray, mask: np.ndarray | None, p_lo: float, p_hi: float) -> tuple[float, float]:
    x = img.astype(np.float32, copy=False)
    if mask is None:
        vals = x.reshape(-1)
    else:
        m = mask > 0
        if m.ndim == x.ndim - 1:
            m = np.broadcast_to(m, x.shape)
        elif m.shape != x.shape:
            try:
                m = np.broadcast_to(m, x.shape)
            except ValueError:
                m = None
        vals = x[m].reshape(-1) if m is not None else x.reshape(-1)
    if vals.size == 0:
        vals = x.reshape(-1)
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0: make example figures (t0/t1/diff/GT overlay)")
    parser.add_argument("--manifest", type=Path, required=True, help="phase0_manifest_checked.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="e.g. results/figures")
    parser.add_argument("--num-patients", type=int, default=3)
    parser.add_argument("--patients", type=str, default="", help="comma-separated patient ids to render (overrides --num-patients)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    out_dir = (repo_root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # local import (no need to install package)
    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.io.nifti import read_array

    rows = _read_manifest_checked(manifest_path)
    if args.patients.strip():
        wanted = {p.strip() for p in args.patients.split(",") if p.strip()}
        rows = [r for r in rows if r["patient_id"] in wanted]
    else:
        rows = rows[: max(0, args.num_patients)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for r in rows:
        pid = r["patient_id"]
        _, t0 = read_array(repo_root / r["t0_flair"])
        _, t1 = read_array(repo_root / r["t1_flair"])
        _, gt = read_array(repo_root / r["gt_change"])
        _, bm = read_array(repo_root / r["brainmask"])

        # pick z slice by max GT area (fallback to middle if empty)
        per_z = np.sum(gt > 0, axis=(0, 1))
        if int(np.max(per_z)) == 0:
            z = gt.shape[2] // 2
        else:
            z = int(np.argmax(per_z))

        t0_z = t0[:, :, z]
        t1_z = t1[:, :, z]
        diff_z = np.abs(t1_z.astype(np.float32) - t0_z.astype(np.float32))
        gt_z = (gt[:, :, z] > 0).astype(np.uint8)
        bm_z = (bm[:, :, z] > 0).astype(np.uint8)

        brain_mask = bm_z > 0

        # shared scaling for t0/t1 so they are visually comparable (within-patient only)
        lo, hi = _robust_range(np.stack([t0_z, t1_z], axis=0), brain_mask, 1, 99)

        # for diff map, focus scaling on brain voxels only
        _, dhi = _robust_range(diff_z, brain_mask, 0, 99)
        dlo = 0.0

        # "publication style": hide outside-brain in diff (and also t0/t1 for cleaner focus)
        t0_show = t0_z.astype(np.float32, copy=False).copy()
        t1_show = t1_z.astype(np.float32, copy=False).copy()
        diff_show = diff_z.astype(np.float32, copy=False).copy()
        t0_show[~brain_mask] = np.nan
        t1_show[~brain_mask] = np.nan
        diff_show[~brain_mask] = np.nan

        fig = plt.figure(figsize=(11, 9), layout="constrained")
        axes = fig.subplots(2, 2)
        for ax in axes.reshape(-1):
            ax.set_axis_off()

        gray = plt.get_cmap("gray").copy()
        gray.set_bad(color="black")
        magma = plt.get_cmap("magma").copy()
        magma.set_bad(color="black")

        axes[0, 0].imshow(t0_show.T, cmap=gray, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
        axes[0, 0].set_title("t0 FLAIR", fontsize=12)

        axes[0, 1].imshow(t1_show.T, cmap=gray, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
        axes[0, 1].imshow(np.ma.masked_where(gt_z.T == 0, gt_z.T), cmap="Reds", alpha=0.45, origin="lower")
        axes[0, 1].set_title("t1 FLAIR + GT(change)", fontsize=12)

        im = axes[1, 0].imshow(diff_show.T, cmap=magma, vmin=dlo, vmax=dhi, origin="lower", interpolation="nearest")
        axes[1, 0].imshow(np.ma.masked_where(gt_z.T == 0, gt_z.T), cmap="Reds", alpha=0.35, origin="lower")
        axes[1, 0].set_title("|t1 - t0| + GT(change)", fontsize=12)
        fig.colorbar(im, ax=axes[1, 0], fraction=0.045, pad=0.02)

        axes[1, 1].imshow(gt_z.T, cmap="gray", vmin=0, vmax=1, origin="lower")
        axes[1, 1].set_title("GT(change) mask", fontsize=12)

        fig.suptitle(f"{pid} (z={z})", fontsize=14)

        out_path = out_dir / f"phase0_{pid}_z{z:02d}.png"
        fig.savefig(out_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote {out_path.relative_to(repo_root).as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
