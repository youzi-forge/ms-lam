from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_patients_arg(patients: str) -> set[str]:
    return {p.strip() for p in patients.split(",") if p.strip()}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _bool(arr: np.ndarray) -> np.ndarray:
    return (arr > 0)


def _safe_float(x: float) -> float:
    return float(x) if math.isfinite(float(x)) else float("nan")


def _make_patient_change_figure(
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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = plt.get_cmap("gray").copy()
    gray.set_bad(color="black")

    bm = brain > 0
    m0 = _bool(mask0)
    m1 = _bool(mask1)
    gt = _bool(gt)
    new_mask = _bool(new_mask)

    union = gt | new_mask | m0 | m1
    if bool(union.any()):
        per_z = union.sum(axis=(0, 1))
        z = int(np.argmax(per_z))
    else:
        per_z = bm.sum(axis=(0, 1))
        z = int(np.argmax(per_z)) if per_z.size else int(flair0.shape[2] // 2)

    f0 = flair0[:, :, z].astype(np.float32, copy=False).copy()
    f1 = flair1[:, :, z].astype(np.float32, copy=False).copy()
    b2 = bm[:, :, z]
    f0[~b2] = np.nan
    f1[~b2] = np.nan

    def robust_vmin_vmax(vals: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
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

    v0 = f0[b2].reshape(-1)
    v1 = f1[b2].reshape(-1)
    if scale == "shared":
        vmin, vmax = robust_vmin_vmax(np.concatenate([v0, v1]), 1, 99)
        vmin0, vmax0 = vmin, vmax
        vmin1, vmax1 = vmin, vmax
    else:
        vmin0, vmax0 = robust_vmin_vmax(v0, 1, 99)
        vmin1, vmax1 = robust_vmin_vmax(v1, 1, 99)

    d = diff_norm[:, :, z].astype(np.float32, copy=False).copy()
    d[~b2] = np.nan
    dvals = d[b2].reshape(-1)
    _, dvmax = robust_vmin_vmax(dvals, 1, 99.5)

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    axes = fig.subplots(2, 2)
    for ax in axes.reshape(-1):
        ax.set_axis_off()

    axes[0, 0].imshow(f0.T, cmap=gray, vmin=vmin0, vmax=vmax0, origin="lower", interpolation="nearest")
    if bool(m0[:, :, z].any()):
        axes[0, 0].contour(m0[:, :, z].T.astype(float), levels=[0.5], colors="red", linewidths=1)
    axes[0, 0].set_title("t0 FLAIR + lesion(mask)", fontsize=12)

    axes[0, 1].imshow(f1.T, cmap=gray, vmin=vmin1, vmax=vmax1, origin="lower", interpolation="nearest")
    if bool(m1[:, :, z].any()):
        axes[0, 1].contour(m1[:, :, z].T.astype(float), levels=[0.5], colors="red", linewidths=1)
    axes[0, 1].set_title("t1 FLAIR + lesion(mask)", fontsize=12)

    axes[1, 0].imshow(f1.T, cmap=gray, vmin=vmin1, vmax=vmax1, origin="lower", interpolation="nearest")
    if bool(new_mask[:, :, z].any()):
        axes[1, 0].contour(new_mask[:, :, z].T.astype(float), levels=[0.5], colors="yellow", linewidths=1)
    if bool(gt[:, :, z].any()):
        axes[1, 0].contour(gt[:, :, z].T.astype(float), levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 0].set_title("t1 + new_proxy(yellow) + GT_change(cyan)", fontsize=12)

    im = axes[1, 1].imshow(d.T, cmap="magma", vmin=0.0, vmax=dvmax, origin="lower", interpolation="nearest")
    if bool(gt[:, :, z].any()):
        axes[1, 1].contour(gt[:, :, z].T.astype(float), levels=[0.5], colors="cyan", linewidths=1)
    axes[1, 1].set_title("|norm(t1)-norm(t0)| + GT_change(cyan)", fontsize=12)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.045, pad=0.02)

    fig.suptitle(f"{patient_id} (z={z}){title_suffix}", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: longitudinal metrics + GT validation + patient reports")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/tables/phase0_manifest_checked.csv"),
        help="Manifest with t0/t1 FLAIR, brainmask, gt_change.",
    )
    parser.add_argument(
        "--lstai-root",
        type=Path,
        default=Path("data/processed/lstai_outputs"),
        help="LST-AI outputs root: patientXX/t0/lesion_mask.nii.gz",
    )
    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient ids to run (default: all ok)")

    parser.add_argument("--out-csv", type=Path, default=Path("results/tables/phase2_longitudinal_metrics.csv"))
    parser.add_argument("--out-reports-dir", type=Path, default=Path("results/reports/phase2"))
    parser.add_argument("--out-fig-dir", type=Path, default=Path("results/figures"))

    parser.add_argument("--new-dilation-radius-mm", type=float, default=2.0)
    parser.add_argument("--min-component-volume-mm3", type=float, default=10.0)
    parser.add_argument("--connectivity", choices=["6", "26"], default="26")

    parser.add_argument("--intensity-p-lo", type=float, default=1.0)
    parser.add_argument("--intensity-p-hi", type=float, default=99.0)
    parser.add_argument("--diff-threshold", type=float, default=0.2)

    parser.add_argument("--example-patients", type=str, default="", help="Comma-separated patient ids for phase2_examples.png")
    parser.add_argument("--scale", choices=["per-timepoint", "shared"], default="per-timepoint")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    lstai_root = (repo_root / args.lstai_root).resolve() if not args.lstai_root.is_absolute() else args.lstai_root.resolve()
    out_csv = (repo_root / args.out_csv).resolve() if not args.out_csv.is_absolute() else args.out_csv.resolve()
    out_reports = (repo_root / args.out_reports_dir).resolve() if not args.out_reports_dir.is_absolute() else args.out_reports_dir.resolve()
    out_fig = (repo_root / args.out_fig_dir).resolve() if not args.out_fig_dir.is_absolute() else args.out_fig_dir.resolve()
    _ensure_dir(out_reports)
    _ensure_dir(out_fig)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.io.nifti import allclose_affine, allclose_zooms, read_array, read_header
    from mslam.metrics.longitudinal_metrics import (
        dice_coefficient,
        dilate_binary_mm,
        intensity_change_stats,
        remove_small_components,
        volume_mm3,
    )

    rows = _read_manifest(manifest_path)
    ok_rows = [r for r in rows if r.get("ok", "True") == "True"]
    if args.patients.strip():
        wanted = _parse_patients_arg(args.patients)
        ok_rows = [r for r in ok_rows if r.get("patient_id") in wanted]

    if not ok_rows:
        print("No patients selected.")
        return 1

    conn = int(args.connectivity)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    out_fieldnames = [
        "patient_id",
        "ok",
        "warnings",
        "voxel_spacing_x",
        "voxel_spacing_y",
        "voxel_spacing_z",
        "voxel_vol_mm3",
        "lesion_vol_t0_mm3",
        "lesion_vol_t1_mm3",
        "delta_lesion_vol_mm3",
        "change_gt_vol_mm3",
        "gt_covered_by_t1_frac",
        "gt_outside_brain_frac",
        "new_raw_vol_mm3",
        "new_cons_vol_mm3",
        "chg_sym_raw_vol_mm3",
        "chg_sym_cons_vol_mm3",
        "new_raw_vol_err_mm3",
        "new_cons_vol_err_mm3",
        "chg_sym_raw_vol_err_mm3",
        "chg_sym_cons_vol_err_mm3",
        "dice_new_raw",
        "dice_new_cons",
        "dice_chg_sym_raw",
        "dice_chg_sym_cons",
        "intensity_diff_mean",
        "intensity_diff_p95",
        "intensity_diff_frac_gt",
        "t0_flair_median",
        "t1_flair_median",
        "t1_over_t0_median_ratio",
        "report_json",
    ]

    patient_metrics: list[dict[str, object]] = []

    for r in ok_rows:
        pid = r["patient_id"]
        warnings: list[str] = []

        flair0_path = (repo_root / r["t0_flair"]).resolve()
        flair1_path = (repo_root / r["t1_flair"]).resolve()
        brainmask_path = (repo_root / r["brainmask"]).resolve()
        gt_path = (repo_root / r["gt_change"]).resolve()

        mask0_path = (lstai_root / pid / "t0" / "lesion_mask.nii.gz").resolve()
        mask1_path = (lstai_root / pid / "t1" / "lesion_mask.nii.gz").resolve()
        prob0_path = (lstai_root / pid / "t0" / "lesion_prob.nii.gz").resolve()
        prob1_path = (lstai_root / pid / "t1" / "lesion_prob.nii.gz").resolve()

        required = [flair0_path, flair1_path, brainmask_path, gt_path, mask0_path, mask1_path]
        missing = [p for p in required if not p.exists()]
        if missing:
            warnings.append("missing_files:" + ",".join(p.name for p in missing))
            report_path = out_reports / f"{pid}.json"
            report = {"patient_id": pid, "ok": False, "warnings": warnings, "errors": ["missing required files"]}
            _save_json(report_path, report)
            patient_metrics.append(
                {
                    "patient_id": pid,
                    "ok": "False",
                    "warnings": ";".join(warnings),
                    "report_json": str(report_path.relative_to(repo_root)),
                }
            )
            continue

        # Grid sanity: compare to t0 FLAIR grid (shape+affine+zooms)
        hdr_ref = read_header(flair0_path)
        spacing = hdr_ref.zooms
        voxel_vol = hdr_ref.voxel_volume_mm3

        grid_ok = True
        for p in [flair1_path, brainmask_path, gt_path, mask0_path, mask1_path]:
            hdr = read_header(p)
            if hdr.shape != hdr_ref.shape:
                warnings.append(f"shape_mismatch:{p.name}")
                grid_ok = False
            if not allclose_affine(hdr.affine, hdr_ref.affine, atol=1e-3):
                warnings.append(f"affine_mismatch:{p.name}")
                grid_ok = False
            if not allclose_zooms(hdr.zooms, hdr_ref.zooms, atol=1e-5):
                warnings.append(f"spacing_diff:{p.name}")

        if not grid_ok:
            report_path = out_reports / f"{pid}.json"
            report = {
                "patient_id": pid,
                "ok": False,
                "warnings": warnings,
                "errors": ["grid mismatch; resampling is not enabled in v1"],
            }
            _save_json(report_path, report)
            patient_metrics.append(
                {
                    "patient_id": pid,
                    "ok": "False",
                    "warnings": ";".join(warnings),
                    "voxel_spacing_x": spacing[0],
                    "voxel_spacing_y": spacing[1],
                    "voxel_spacing_z": spacing[2],
                    "voxel_vol_mm3": voxel_vol,
                    "report_json": str(report_path.relative_to(repo_root)),
                }
            )
            continue

        # Load arrays
        _, flair0 = read_array(flair0_path)
        _, flair1 = read_array(flair1_path)
        _, brainmask = read_array(brainmask_path)
        _, gt = read_array(gt_path)
        _, mask0 = read_array(mask0_path)
        _, mask1 = read_array(mask1_path)

        bm = _bool(brainmask)
        gt_b = _bool(gt)
        m0 = _bool(mask0)
        m1 = _bool(mask1)

        # Volumes
        v0 = volume_mm3(m0, voxel_vol)
        v1 = volume_mm3(m1, voxel_vol)
        dv = v1 - v0

        gt_vol = volume_mm3(gt_b, voxel_vol)

        gt_vox = int(gt_b.sum())
        gt_intersection_t1 = int(np.logical_and(gt_b, m1).sum())
        gt_outside_brain = int(np.logical_and(gt_b, ~bm).sum())
        if gt_vox > 0:
            gt_covered_by_t1_frac: float | None = gt_intersection_t1 / gt_vox
            gt_outside_brain_frac: float | None = gt_outside_brain / gt_vox
        else:
            gt_covered_by_t1_frac = None
            gt_outside_brain_frac = None

        # Proxies
        new_raw = m1 & (~m0)

        m0_dil = dilate_binary_mm(m0, spacing_xyz=spacing, radius_mm=float(args.new_dilation_radius_mm))
        new_cons_pre = m1 & (~m0_dil)

        min_voxels = int(math.ceil(float(args.min_component_volume_mm3) / float(voxel_vol))) if voxel_vol > 0 else 1
        new_cons = remove_small_components(new_cons_pre, min_voxels=min_voxels, connectivity=conn)

        chg_sym_raw = np.logical_xor(m1, m0)
        chg_sym_cons_pre = np.logical_xor(m1, m0_dil)
        chg_sym_cons = remove_small_components(chg_sym_cons_pre, min_voxels=min_voxels, connectivity=conn)

        # Dice vs GT
        dice_new_raw = dice_coefficient(new_raw, gt_b)
        dice_new_cons = dice_coefficient(new_cons, gt_b)
        dice_sym_raw = dice_coefficient(chg_sym_raw, gt_b)
        dice_sym_cons = dice_coefficient(chg_sym_cons, gt_b)

        # Proxy volumes + volume errors to GT (mm³)
        new_raw_vol = volume_mm3(new_raw, voxel_vol)
        new_cons_vol = volume_mm3(new_cons, voxel_vol)
        sym_raw_vol = volume_mm3(chg_sym_raw, voxel_vol)
        sym_cons_vol = volume_mm3(chg_sym_cons, voxel_vol)

        # Intensity change stats (normalized)
        t0_stats, t1_stats, diff_stats = intensity_change_stats(
            flair0,
            flair1,
            bm,
            p_lo=float(args.intensity_p_lo),
            p_hi=float(args.intensity_p_hi),
            diff_threshold=float(args.diff_threshold),
        )

        # Normalized diff image for visualization
        # (recompute norms using returned v_lo/v_hi for reproducibility)
        denom0 = (t0_stats["v_hi"] - t0_stats["v_lo"]) if (t0_stats["v_hi"] > t0_stats["v_lo"]) else 1.0
        denom1 = (t1_stats["v_hi"] - t1_stats["v_lo"]) if (t1_stats["v_hi"] > t1_stats["v_lo"]) else 1.0
        n0 = np.clip((flair0.astype(np.float32, copy=False) - float(t0_stats["v_lo"])) / float(denom0), 0.0, 1.0)
        n1 = np.clip((flair1.astype(np.float32, copy=False) - float(t1_stats["v_lo"])) / float(denom1), 0.0, 1.0)
        diff_norm = np.abs(n1 - n0)

        report_path = out_reports / f"{pid}.json"
        report = {
            "patient_id": pid,
            "ok": True,
            "meta": {
                "generated_at": started_at,
                "reference_space": "open_ms_data longitudinal/coregistered FLAIR(t0) grid",
                "voxel_spacing_xyz": [float(spacing[0]), float(spacing[1]), float(spacing[2])],
                "voxel_vol_mm3": float(voxel_vol),
                "dice_empty_policy": "both_empty=1.0; one_empty=0.0",
                "new_dilation_radius_mm": float(args.new_dilation_radius_mm),
                "min_component_volume_mm3": float(args.min_component_volume_mm3),
                "connectivity": conn,
                "intensity_norm": {
                    "method": "brainmask_percentile_scaling",
                    "p_lo": float(args.intensity_p_lo),
                    "p_hi": float(args.intensity_p_hi),
                    "diff_threshold": float(args.diff_threshold),
                },
            },
            "inputs": {
                "flair_t0": str(flair0_path.relative_to(repo_root)),
                "flair_t1": str(flair1_path.relative_to(repo_root)),
                "brainmask": str(brainmask_path.relative_to(repo_root)),
                "change_gt": str(gt_path.relative_to(repo_root)),
                "lesion_mask_t0": str(mask0_path.relative_to(repo_root)),
                "lesion_mask_t1": str(mask1_path.relative_to(repo_root)),
                "lesion_prob_t0": str(prob0_path.relative_to(repo_root)) if prob0_path.exists() else "",
                "lesion_prob_t1": str(prob1_path.relative_to(repo_root)) if prob1_path.exists() else "",
            },
            "warnings": warnings,
            "lesion_volumes_mm3": {"t0": _safe_float(v0), "t1": _safe_float(v1), "delta": _safe_float(dv)},
            "change_gt_mm3": {"volume": _safe_float(gt_vol), "voxels": int(gt_b.sum())},
            "gt_diagnostics": {
                "covered_by_t1_lesion_mask_frac": gt_covered_by_t1_frac,
                "outside_brainmask_frac": gt_outside_brain_frac,
                "covered_by_t1_lesion_mask_voxels": gt_intersection_t1,
                "outside_brainmask_voxels": gt_outside_brain,
            },
            "proxies": {
                "new_raw": {
                    "voxels": int(new_raw.sum()),
                    "volume_mm3": _safe_float(new_raw_vol),
                    "volume_error_mm3": _safe_float(new_raw_vol - gt_vol),
                    "abs_volume_error_mm3": _safe_float(abs(new_raw_vol - gt_vol)),
                    "dice_to_gt": _safe_float(dice_new_raw),
                },
                "new_conservative": {
                    "voxels": int(new_cons.sum()),
                    "volume_mm3": _safe_float(new_cons_vol),
                    "volume_error_mm3": _safe_float(new_cons_vol - gt_vol),
                    "abs_volume_error_mm3": _safe_float(abs(new_cons_vol - gt_vol)),
                    "dice_to_gt": _safe_float(dice_new_cons),
                },
                "chg_sym_raw": {
                    "voxels": int(chg_sym_raw.sum()),
                    "volume_mm3": _safe_float(sym_raw_vol),
                    "volume_error_mm3": _safe_float(sym_raw_vol - gt_vol),
                    "abs_volume_error_mm3": _safe_float(abs(sym_raw_vol - gt_vol)),
                    "dice_to_gt": _safe_float(dice_sym_raw),
                },
                "chg_sym_cons": {
                    "voxels": int(chg_sym_cons.sum()),
                    "volume_mm3": _safe_float(sym_cons_vol),
                    "volume_error_mm3": _safe_float(sym_cons_vol - gt_vol),
                    "abs_volume_error_mm3": _safe_float(abs(sym_cons_vol - gt_vol)),
                    "dice_to_gt": _safe_float(dice_sym_cons),
                },
            },
            "intensity_change": {"t0": t0_stats, "t1": t1_stats, "diff": diff_stats},
        }
        _save_json(report_path, report)

        patient_metrics.append(
            {
                "patient_id": pid,
                "ok": "True",
                "warnings": ";".join(warnings),
                "voxel_spacing_x": spacing[0],
                "voxel_spacing_y": spacing[1],
                "voxel_spacing_z": spacing[2],
                "voxel_vol_mm3": voxel_vol,
                "lesion_vol_t0_mm3": v0,
                "lesion_vol_t1_mm3": v1,
                "delta_lesion_vol_mm3": dv,
                "change_gt_vol_mm3": gt_vol,
                "gt_covered_by_t1_frac": "" if gt_covered_by_t1_frac is None else gt_covered_by_t1_frac,
                "gt_outside_brain_frac": "" if gt_outside_brain_frac is None else gt_outside_brain_frac,
                "new_raw_vol_mm3": new_raw_vol,
                "new_cons_vol_mm3": new_cons_vol,
                "chg_sym_raw_vol_mm3": sym_raw_vol,
                "chg_sym_cons_vol_mm3": sym_cons_vol,
                "new_raw_vol_err_mm3": (new_raw_vol - gt_vol),
                "new_cons_vol_err_mm3": (new_cons_vol - gt_vol),
                "chg_sym_raw_vol_err_mm3": (sym_raw_vol - gt_vol),
                "chg_sym_cons_vol_err_mm3": (sym_cons_vol - gt_vol),
                "dice_new_raw": dice_new_raw,
                "dice_new_cons": dice_new_cons,
                "dice_chg_sym_raw": dice_sym_raw,
                "dice_chg_sym_cons": dice_sym_cons,
                "intensity_diff_mean": diff_stats["mean"],
                "intensity_diff_p95": diff_stats["p95"],
                "intensity_diff_frac_gt": diff_stats["frac_gt"],
                "t0_flair_median": t0_stats["median"],
                "t1_flair_median": t1_stats["median"],
                "t1_over_t0_median_ratio": (t1_stats["median"] / t0_stats["median"]) if t0_stats["median"] != 0 else float("nan"),
                "report_json": str(report_path.relative_to(repo_root)),
                # keep file paths for figure generation (avoid holding large arrays in memory)
                "_paths": {
                    "flair0": str(flair0_path),
                    "flair1": str(flair1_path),
                    "brainmask": str(brainmask_path),
                    "gt": str(gt_path),
                    "mask0": str(mask0_path),
                    "mask1": str(mask1_path),
                },
            }
        )

    # Write aggregate CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for m in patient_metrics:
            row = {k: m.get(k, "") for k in out_fieldnames}
            writer.writerow(row)

    # Figures: deltaV distribution + examples + worst-case
    ok_metrics = [m for m in patient_metrics if m.get("ok") == "True"]
    if ok_metrics:
        def load_for_fig(m: dict[str, object]) -> dict[str, np.ndarray]:
            p = m["_paths"]  # type: ignore[index]
            flair0_p = Path(p["flair0"])  # type: ignore[index]
            flair1_p = Path(p["flair1"])  # type: ignore[index]
            brain_p = Path(p["brainmask"])  # type: ignore[index]
            gt_p = Path(p["gt"])  # type: ignore[index]
            mask0_p = Path(p["mask0"])  # type: ignore[index]
            mask1_p = Path(p["mask1"])  # type: ignore[index]

            _, flair0 = read_array(flair0_p)
            _, flair1 = read_array(flair1_p)
            _, brainmask = read_array(brain_p)
            _, gt_arr = read_array(gt_p)
            _, mask0_arr = read_array(mask0_p)
            _, mask1_arr = read_array(mask1_p)

            bm_arr = _bool(brainmask)
            m0_arr = _bool(mask0_arr)
            m1_arr = _bool(mask1_arr)
            gt_bool = _bool(gt_arr)

            # Proxies for visualization (use conservative new)
            hdr_ref = read_header(flair0_p)
            spacing_xyz = hdr_ref.zooms
            voxel_vol_mm3 = hdr_ref.voxel_volume_mm3
            m0_dil_v = dilate_binary_mm(m0_arr, spacing_xyz=spacing_xyz, radius_mm=float(args.new_dilation_radius_mm))
            new_pre = m1_arr & (~m0_dil_v)
            min_vox = int(math.ceil(float(args.min_component_volume_mm3) / float(voxel_vol_mm3))) if voxel_vol_mm3 > 0 else 1
            new_v = remove_small_components(new_pre, min_voxels=min_vox, connectivity=conn)

            # Intensity normalized diff
            t0_s, t1_s, _ = intensity_change_stats(
                flair0,
                flair1,
                bm_arr,
                p_lo=float(args.intensity_p_lo),
                p_hi=float(args.intensity_p_hi),
                diff_threshold=float(args.diff_threshold),
            )
            denom0 = (t0_s["v_hi"] - t0_s["v_lo"]) if (t0_s["v_hi"] > t0_s["v_lo"]) else 1.0
            denom1 = (t1_s["v_hi"] - t1_s["v_lo"]) if (t1_s["v_hi"] > t1_s["v_lo"]) else 1.0
            n0 = np.clip((flair0.astype(np.float32, copy=False) - float(t0_s["v_lo"])) / float(denom0), 0.0, 1.0)
            n1 = np.clip((flair1.astype(np.float32, copy=False) - float(t1_s["v_lo"])) / float(denom1), 0.0, 1.0)
            diff_n = np.abs(n1 - n0)

            return {
                "flair0": flair0,
                "flair1": flair1,
                "brain": bm_arr,
                "mask0": m0_arr,
                "mask1": m1_arr,
                "gt": gt_bool,
                "new": new_v,
                "diff_norm": diff_n,
            }

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        deltas = np.array([float(m["delta_lesion_vol_mm3"]) for m in ok_metrics], dtype=np.float64)
        fig = plt.figure(figsize=(7, 4.5), layout="constrained")
        ax = fig.subplots()
        ax.hist(deltas, bins=10, color="#4C72B0", alpha=0.9)
        ax.set_title("Phase 2: ΔLesion volume (t1 - t0) [mm³]")
        ax.set_xlabel("ΔV (mm³)")
        ax.set_ylabel("Count")
        fig.savefig(out_fig / "phase2_deltaV_hist.png", dpi=200)
        plt.close(fig)

        # Worst-case by dice_chg_sym_cons (lower is worse)
        worst = min(ok_metrics, key=lambda x: float(x.get("dice_chg_sym_cons", 1.0)))
        wa = load_for_fig(worst)
        _make_patient_change_figure(
            patient_id=worst["patient_id"],
            flair0=wa["flair0"],
            flair1=wa["flair1"],
            brain=wa["brain"],
            mask0=wa["mask0"],
            mask1=wa["mask1"],
            gt=wa["gt"],
            new_mask=wa["new"],
            diff_norm=wa["diff_norm"],
            out_path=out_fig / "phase2_worst_case.png",
            title_suffix=(
                f"  dice_new_cons={float(worst['dice_new_cons']):.3f}"
                f"  dice_sym_cons={float(worst['dice_chg_sym_cons']):.3f}"
            ),
            scale=args.scale,
        )

        # Example figure: either user-specified patients, or [patient01 + worst (if different)]
        example_ids: list[str]
        if args.example_patients.strip():
            example_ids = [p.strip() for p in args.example_patients.split(",") if p.strip()]
        else:
            example_ids = ["patient01"]
            if worst["patient_id"] != "patient01":
                example_ids.append(worst["patient_id"])

        ex_metrics = [m for m in ok_metrics if m["patient_id"] in set(example_ids)]
        if ex_metrics:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            n = len(ex_metrics)
            fig = plt.figure(figsize=(12, 4.6 * n), layout="constrained")
            gs = fig.add_gridspec(n, 1)
            for i, m in enumerate(ex_metrics):
                ax = fig.add_subplot(gs[i, 0])
                ax.set_axis_off()
                # Render to a temporary png per patient, then embed.
                tmp = out_fig / f"_tmp_phase2_{m['patient_id']}.png"
                a = load_for_fig(m)
                _make_patient_change_figure(
                    patient_id=m["patient_id"],
                    flair0=a["flair0"],
                    flair1=a["flair1"],
                    brain=a["brain"],
                    mask0=a["mask0"],
                    mask1=a["mask1"],
                    gt=a["gt"],
                    new_mask=a["new"],
                    diff_norm=a["diff_norm"],
                    out_path=tmp,
                    title_suffix=(
                        f"  dice_new_cons={float(m['dice_new_cons']):.3f}"
                        f"  dice_sym_cons={float(m['dice_chg_sym_cons']):.3f}"
                    ),
                    scale=args.scale,
                )
                img = plt.imread(tmp)
                ax.imshow(img)
                tmp.unlink(missing_ok=True)
            fig.suptitle("Phase 2 examples", fontsize=14)
            fig.savefig(out_fig / "phase2_examples.png", dpi=200)
            plt.close(fig)

    print(f"Wrote {out_csv.relative_to(repo_root).as_posix()}")
    print(f"Wrote reports -> {out_reports.relative_to(repo_root).as_posix()}/")
    print(f"Wrote figures -> {out_fig.relative_to(repo_root).as_posix()}/ (phase2_*.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
