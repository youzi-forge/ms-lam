from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_manifest_checked(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("ok", "False") == "True"]


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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: aggregate QC report (PDF) for LST-AI overlays over all patients")
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
    parser.add_argument("--out-pdf", type=Path, default=Path("results/reports/phase1_lstai_qc_overlays.pdf"))
    parser.add_argument("--out-csv", type=Path, default=Path("results/tables/phase1_mask_volumes.csv"))
    parser.add_argument("--patients-per-page", type=int, default=5, help="Rows per PDF page (each row has t0/t1)")
    parser.add_argument(
        "--scale",
        choices=["per-timepoint", "shared"],
        default="per-timepoint",
        help="Visualization scaling: per-timepoint improves readability; shared helps spot intensity shifts.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    lstai_root = (repo_root / args.lstai_root).resolve() if not args.lstai_root.is_absolute() else args.lstai_root.resolve()
    out_pdf = (repo_root / args.out_pdf).resolve() if not args.out_pdf.is_absolute() else args.out_pdf.resolve()
    out_csv = (repo_root / args.out_csv).resolve() if not args.out_csv.is_absolute() else args.out_csv.resolve()
    _ensure_parent(out_pdf)
    _ensure_parent(out_csv)

    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.io.nifti import read_array, read_header

    rows = _read_manifest_checked(manifest_path)
    if not rows:
        print("No ok patients found in manifest.")
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    gray = plt.get_cmap("gray").copy()
    gray.set_bad(color="black")

    csv_fieldnames = [
        "patient_id",
        "t0_mask_voxels",
        "t0_mask_volume_mm3",
        "t0_flair_median",
        "t0_flair_p99",
        "t1_mask_voxels",
        "t1_mask_volume_mm3",
        "t1_flair_median",
        "t1_flair_p99",
        "t1_over_t0_median_ratio",
        "delta_mask_volume_mm3",
        "t0_z_used",
        "t1_z_used",
        "qc_z_used",
        "qc_union_voxels_on_z",
        "t0_voxels_on_z",
        "t1_voxels_on_z",
    ]

    with out_csv.open("w", newline="") as fcsv, PdfPages(out_pdf) as pdf:
        writer = csv.DictWriter(fcsv, fieldnames=csv_fieldnames)
        writer.writeheader()

        per_page = max(1, int(args.patients_per_page))
        total = len(rows)
        for page_start in range(0, total, per_page):
            batch = rows[page_start : page_start + per_page]
            fig, axes = plt.subplots(len(batch), 2, figsize=(10, 4.0 * len(batch)), layout="constrained")
            if len(batch) == 1:
                axes = np.array([axes])  # type: ignore[assignment]

            for i, r in enumerate(batch):
                pid = r["patient_id"]
                flair0_path = (repo_root / r["t0_flair"]).resolve()
                flair1_path = (repo_root / r["t1_flair"]).resolve()
                bm_path = (repo_root / r["brainmask"]).resolve()

                mask0_path = (lstai_root / pid / "t0" / "lesion_mask.nii.gz").resolve()
                mask1_path = (lstai_root / pid / "t1" / "lesion_mask.nii.gz").resolve()
                if not mask0_path.exists() or not mask1_path.exists():
                    raise FileNotFoundError(f"Missing LST-AI masks for {pid}: {mask0_path} or {mask1_path}")

                _, flair0 = read_array(flair0_path)
                _, flair1 = read_array(flair1_path)
                _, bm = read_array(bm_path)
                _, mask0 = read_array(mask0_path)
                _, mask1 = read_array(mask1_path)

                brain = bm > 0
                m0 = mask0 > 0
                m1 = mask1 > 0
                union = m0 | m1

                per_z = union.sum(axis=(0, 1))
                z = int(np.argmax(per_z)) if int(np.max(per_z)) > 0 else int(flair0.shape[2] // 2)

                b = brain[:, :, z]
                f0 = flair0[:, :, z].astype(np.float32, copy=False).copy()
                f1 = flair1[:, :, z].astype(np.float32, copy=False).copy()
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

                ax0, ax1 = axes[i, 0], axes[i, 1]
                for ax in (ax0, ax1):
                    ax.set_axis_off()

                m0z = m0[:, :, z]
                m1z = m1[:, :, z]

                ax0.imshow(f0.T, cmap=gray, vmin=vmin0, vmax=vmax0, origin="lower", interpolation="nearest")
                ax0.imshow(np.ma.masked_where(m0z.T == 0, m0z.T), cmap="Reds", alpha=0.45, origin="lower")

                ax1.imshow(f1.T, cmap=gray, vmin=vmin1, vmax=vmax1, origin="lower", interpolation="nearest")
                ax1.imshow(np.ma.masked_where(m1z.T == 0, m1z.T), cmap="Reds", alpha=0.45, origin="lower")

                # volumes
                hdr0 = read_header(flair0_path)
                hdr1 = read_header(flair1_path)
                # Prefer each timepoint's own voxel volume (they should match, but keep it robust).
                m0_mm3 = float(int(m0.sum()) * hdr0.voxel_volume_mm3)
                m1_mm3 = float(int(m1.sum()) * hdr1.voxel_volume_mm3)

                t0_median = float(np.nanmedian(f0))
                t1_median = float(np.nanmedian(f1))
                t0_p99 = float(np.nanpercentile(f0, 99))
                t1_p99 = float(np.nanpercentile(f1, 99))
                ratio = (t1_median / t0_median) if t0_median != 0 else float("nan")

                ax0.set_title(f"{pid} t0 (z={z})  vox={int(m0.sum())}  mm³={m0_mm3:.0f}", fontsize=11)
                ax1.set_title(f"{pid} t1 (z={z})  vox={int(m1.sum())}  mm³={m1_mm3:.0f}", fontsize=11)

                writer.writerow(
                    {
                        "patient_id": pid,
                        "t0_mask_voxels": str(int(m0.sum())),
                        "t0_mask_volume_mm3": f"{m0_mm3:.3f}",
                        "t0_flair_median": f"{t0_median:.6f}",
                        "t0_flair_p99": f"{t0_p99:.6f}",
                        "t1_mask_voxels": str(int(m1.sum())),
                        "t1_mask_volume_mm3": f"{m1_mm3:.3f}",
                        "t1_flair_median": f"{t1_median:.6f}",
                        "t1_flair_p99": f"{t1_p99:.6f}",
                        "t1_over_t0_median_ratio": f"{ratio:.6f}",
                        "delta_mask_volume_mm3": f"{(m1_mm3 - m0_mm3):.3f}",
                        "t0_z_used": "",
                        "t1_z_used": "",
                        "qc_z_used": str(z),
                        "qc_union_voxels_on_z": str(int(per_z[z])),
                        "t0_voxels_on_z": str(int(m0z.sum())),
                        "t1_voxels_on_z": str(int(m1z.sum())),
                    }
                )

            fig.suptitle(
                f"Phase 1 QC: LST-AI overlays (FLAIR)  —  per-row: patient, per-col: t0/t1  —  scale={args.scale}",
                fontsize=14,
            )
            pdf.savefig(fig, dpi=args.dpi)
            plt.close(fig)

    print(f"Wrote {out_pdf.relative_to(repo_root).as_posix()}")
    print(f"Wrote {out_csv.relative_to(repo_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
