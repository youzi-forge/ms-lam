from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_manifest_checked(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("ok", "False") == "True"]


def _parse_patients_arg(patients: str) -> list[str]:
    return [p.strip() for p in patients.split(",") if p.strip()]


def _robust_vmin_vmax(vals: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, p_lo))
    hi = float(np.percentile(vals, p_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: overlay LST-AI lesion mask on FLAIR for a few patients")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/tables/phase0_manifest_checked.csv"),
        help="Checked manifest with t0/t1 FLAIR paths.",
    )
    parser.add_argument(
        "--lstai-root",
        type=Path,
        default=Path("data/processed/lstai_outputs"),
        help="Root of LST-AI outputs (patientXX/t0/lesion_mask.nii.gz).",
    )
    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient ids to render")
    parser.add_argument("--num-patients", type=int, default=2, help="Used when --patients is empty")
    parser.add_argument("--timepoints", choices=["t0", "t1", "both"], default="both", help="Which timepoints to render")
    parser.add_argument(
        "--scale",
        choices=["per-timepoint", "shared"],
        default="per-timepoint",
        help="Visualization scaling: per-timepoint improves readability; shared helps spot intensity shifts.",
    )
    parser.add_argument("--out", type=Path, default=Path("results/figures/phase1_lstai_overlay_examples.png"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    lstai_root = (repo_root / args.lstai_root).resolve() if not args.lstai_root.is_absolute() else args.lstai_root.resolve()
    out_path = (repo_root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.io.nifti import read_array

    rows = _read_manifest_checked(manifest_path)
    if args.patients.strip():
        wanted = set(_parse_patients_arg(args.patients))
        rows = [r for r in rows if r["patient_id"] in wanted]
    else:
        rows = rows[: max(0, args.num_patients)]

    if not rows:
        print("No patients selected.")
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    tps = ["t0", "t1"] if args.timepoints == "both" else [args.timepoints]
    ncols = len(tps)
    fig, axes = plt.subplots(n, ncols, figsize=(5.2 * ncols, 4.2 * n), layout="constrained")
    if n == 1 and ncols == 1:
        axes = np.array([[axes]])  # type: ignore[assignment]
    elif n == 1:
        axes = np.array([axes])  # type: ignore[assignment]
    elif ncols == 1:
        axes = axes.reshape(n, 1)  # type: ignore[assignment]

    gray = plt.get_cmap("gray").copy()
    gray.set_bad(color="black")

    for i, r in enumerate(rows):
        pid = r["patient_id"]
        flair0_path = repo_root / r["t0_flair"]
        flair1_path = repo_root / r["t1_flair"]
        bm_path = repo_root / r["brainmask"]

        mask0_path = lstai_root / pid / "t0" / "lesion_mask.nii.gz"
        mask1_path = lstai_root / pid / "t1" / "lesion_mask.nii.gz"

        if "t0" in tps and not mask0_path.exists():
            raise FileNotFoundError(
                f"Missing LST-AI outputs for {pid} t0. Expected: {mask0_path}. "
                f"(If you only ran one timepoint, set --timepoints t0 or t1.)"
            )
        if "t1" in tps and not mask1_path.exists():
            raise FileNotFoundError(
                f"Missing LST-AI outputs for {pid} t1. Expected: {mask1_path}. "
                f"(If you only ran one timepoint, set --timepoints t0 or t1.)"
            )

        _, flair0 = read_array(flair0_path)
        _, flair1 = read_array(flair1_path)
        _, bm = read_array(bm_path)
        mask0 = mask1 = None
        if mask0_path.exists():
            _, mask0 = read_array(mask0_path)
        if mask1_path.exists():
            _, mask1 = read_array(mask1_path)

        brain = bm > 0
        m_union = None
        if mask0 is not None and mask1 is not None:
            m_union = (mask0 > 0) | (mask1 > 0)
        elif mask0 is not None:
            m_union = (mask0 > 0)
        elif mask1 is not None:
            m_union = (mask1 > 0)
        else:
            raise RuntimeError(f"Internal error: no masks loaded for {pid}")
        per_z = np.sum(m_union, axis=(0, 1))
        z = int(np.argmax(per_z)) if int(np.max(per_z)) > 0 else int(flair0.shape[2] // 2)

        f0 = flair0[:, :, z].astype(np.float32, copy=False).copy()
        f1 = flair1[:, :, z].astype(np.float32, copy=False).copy()
        b = brain[:, :, z]
        f0[~b] = np.nan
        f1[~b] = np.nan

        v0 = f0[b].reshape(-1)
        v1 = f1[b].reshape(-1)
        if args.scale == "shared":
            vmin, vmax = _robust_vmin_vmax(np.concatenate([v0, v1]), 1, 99)
            vmin0, vmax0 = vmin, vmax
            vmin1, vmax1 = vmin, vmax
        else:
            vmin0, vmax0 = _robust_vmin_vmax(v0, 1, 99)
            vmin1, vmax1 = _robust_vmin_vmax(v1, 1, 99)

        for j, tp in enumerate(tps):
            ax = axes[i, j]
            ax.set_axis_off()
            if tp == "t0":
                m = (mask0[:, :, z] > 0) if mask0 is not None else np.zeros_like(b, dtype=bool)
                ax.imshow(f0.T, cmap=gray, vmin=vmin0, vmax=vmax0, origin="lower", interpolation="nearest")
                ax.imshow(np.ma.masked_where(m.T == 0, m.T), cmap="Reds", alpha=0.45, origin="lower")
                ax.set_title(f"{pid} t0 (z={z})", fontsize=12)
            else:
                m = (mask1[:, :, z] > 0) if mask1 is not None else np.zeros_like(b, dtype=bool)
                ax.imshow(f1.T, cmap=gray, vmin=vmin1, vmax=vmax1, origin="lower", interpolation="nearest")
                ax.imshow(np.ma.masked_where(m.T == 0, m.T), cmap="Reds", alpha=0.45, origin="lower")
                ax.set_title(f"{pid} t1 (z={z})", fontsize=12)

    fig.suptitle("Phase 1: LST-AI lesion mask overlays (FLAIR)", fontsize=14)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote {out_path.relative_to(repo_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
