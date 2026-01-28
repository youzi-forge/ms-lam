from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_patients_arg(patients: str) -> set[str]:
    return {p.strip() for p in patients.split(",") if p.strip()}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _safe_float(x: float) -> float:
    return float(x) if math.isfinite(float(x)) else float("nan")


def _write_nifti_like(ref_path: Path, out_path: Path, data) -> None:
    import nibabel as nib
    import numpy as np

    ref_img = nib.load(str(ref_path))
    arr = np.asarray(data)
    if tuple(arr.shape) != tuple(ref_img.shape[:3]):
        raise ValueError(f"Shape mismatch for write: ref={ref_img.shape} vs data={arr.shape}")
    out_img = nib.Nifti1Image(arr.astype(np.float32, copy=False), ref_img.affine, ref_img.header)
    out_img.header.set_data_dtype(np.float32)
    try:
        out_img.header.set_zooms(ref_img.header.get_zooms()[:3])
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


def _robust_vmin_vmax(vals, p_lo: float = 1.0, p_hi: float = 99.0) -> tuple[float, float]:
    import numpy as np

    vals = np.asarray(vals)
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


def _make_uncertainty_overlay_figure(
    *,
    patient_ids: list[str],
    rows_by_patient: dict[str, dict[str, object]],
    out_path: Path,
    title: str,
    scale: str = "per-timepoint",
) -> None:
    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = plt.get_cmap("gray").copy()
    gray.set_bad(color="black")

    n = len(patient_ids)
    fig = plt.figure(figsize=(11, 4.4 * n), layout="constrained")
    gs = fig.add_gridspec(n, 2)

    # Use a shared uncertainty scale across displayed patients for comparability.
    umax_global = 0.0
    for pid in patient_ids:
        d = rows_by_patient[pid]
        brain = d["brainmask"]
        lesion = d["lesion_mask_t1"]
        boundary = d["boundary_mask_t1"]
        unc = d["unc_map_t1"]
        z = int(d["z"])
        bm = brain[:, :, z] > 0
        u = unc[:, :, z].astype(np.float32, copy=False)
        region = (lesion[:, :, z] > 0) | (boundary[:, :, z] > 0)
        region = region & bm
        if bool(region.any()):
            umax = _robust_vmin_vmax(u[region].reshape(-1), 1, 99.5)[1]
        else:
            umax = _robust_vmin_vmax(u[bm].reshape(-1), 1, 99.5)[1]
        umax_global = max(float(umax_global), float(umax))
    if not math.isfinite(float(umax_global)) or float(umax_global) <= 0:
        umax_global = 1.0

    for i, pid in enumerate(patient_ids):
        d = rows_by_patient[pid]
        flair = d["flair_t1"]
        brain = d["brainmask"]
        lesion = d["lesion_mask_t1"]
        boundary = d["boundary_mask_t1"]
        unc = d["unc_map_t1"]
        z = int(d["z"])
        needs_review = bool(d.get("qc_needs_review", False))
        unc_mean_les = float(d.get("unc_mean_lesion", float("nan")))
        unc_p95_br = float(d.get("unc_p95_brain", float("nan")))

        bm = brain[:, :, z] > 0
        f = flair[:, :, z].astype(np.float32, copy=False).copy()
        f[~bm] = np.nan
        u = unc[:, :, z].astype(np.float32, copy=False).copy()
        u[~bm] = np.nan

        if scale == "shared":
            vmin, vmax = _robust_vmin_vmax(f[bm].reshape(-1), 1, 99)
        else:
            vmin, vmax = _robust_vmin_vmax(f[bm].reshape(-1), 1, 99)

        umin, umax = 0.0, float(umax_global)

        ax0 = fig.add_subplot(gs[i, 0])
        ax1 = fig.add_subplot(gs[i, 1])
        for ax in (ax0, ax1):
            ax.set_axis_off()

        ax0.imshow(f.T, cmap=gray, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        if bool((lesion[:, :, z] > 0).any()):
            ax0.contour((lesion[:, :, z] > 0).T.astype(float), levels=[0.5], colors="red", linewidths=1)
        ax0.set_title(f"{pid}  needs_review={needs_review}", fontsize=11)

        ax1.imshow(f.T, cmap=gray, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
        im = ax1.imshow(u.T, cmap="magma", vmin=umin, vmax=umax, origin="lower", interpolation="nearest", alpha=0.65)
        if bool((lesion[:, :, z] > 0).any()):
            ax1.contour((lesion[:, :, z] > 0).T.astype(float), levels=[0.5], colors="red", linewidths=1)
        ax1.set_title(
            f"{pid}  unc_mean_lesion={unc_mean_les:.3f}  unc_p95_brain={unc_p95_br:.2e}",
            fontsize=11,
        )
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)

    fig.suptitle(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4: voxel→patient uncertainty summaries + QC flags")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/tables/phase0_manifest_checked.csv"),
        help="Manifest with ok, t0/t1 FLAIR, brainmask, gt_change.",
    )
    parser.add_argument(
        "--lstai-root",
        type=Path,
        default=Path("data/processed/lstai_outputs"),
        help="LST-AI outputs root: patientXX/{t0,t1}/lesion_prob*.nii.gz",
    )
    parser.add_argument(
        "--phase2-csv",
        type=Path,
        default=Path("results/tables/phase2_longitudinal_metrics.csv"),
        help="Phase 2 aggregate CSV (for error/ΔV context).",
    )
    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient ids (default: all ok).")
    parser.add_argument("--example-patients", type=str, default="patient01", help="Comma-separated examples for overlay figure.")

    parser.add_argument("--boundary-radius-mm", type=float, default=1.0, help="Outer boundary band radius (mm).")
    parser.add_argument("--unc-types", type=str, default="ens_var,prob_entropy", help="Comma-separated: ens_var,prob_entropy,prob_p1mp")

    parser.add_argument("--qc-quantile", type=float, default=0.9, help="Quantile for uncertainty thresholds (e.g. 0.9).")
    parser.add_argument("--change-quantile", type=float, default=0.9, help="Quantile for |ΔV| high-change threshold.")
    parser.add_argument("--primary-unc", type=str, default="ens_var", help="Which uncertainty type drives QC by default.")

    parser.add_argument("--out-csv", type=Path, default=Path("results/tables/phase4_uncertainty_metrics.csv"))
    parser.add_argument("--out-thresholds", type=Path, default=Path("results/tables/phase4_qc_thresholds.json"))
    parser.add_argument("--out-reports-dir", type=Path, default=Path("results/reports/phase4"))
    parser.add_argument("--out-fig-dir", type=Path, default=Path("results/figures"))

    parser.add_argument("--save-maps", action="store_true", help="Write voxel-level uncertainty maps to disk (can be large).")
    parser.add_argument("--maps-root", type=Path, default=Path("data/processed/phase4_uncertainty_maps"))
    parser.add_argument("--scale", choices=["per-timepoint", "shared"], default="per-timepoint")
    parser.add_argument(
        "--scatter-annotate",
        choices=["none", "extremes", "qc", "all"],
        default="qc",
        help="Annotate points in phase4_unc_vs_error.png (qc=QC-flagged + extremes).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    lstai_root = (repo_root / args.lstai_root).resolve() if not args.lstai_root.is_absolute() else args.lstai_root.resolve()
    phase2_csv = (repo_root / args.phase2_csv).resolve() if not args.phase2_csv.is_absolute() else args.phase2_csv.resolve()

    out_csv = (repo_root / args.out_csv).resolve() if not args.out_csv.is_absolute() else args.out_csv.resolve()
    out_thresholds = (repo_root / args.out_thresholds).resolve() if not args.out_thresholds.is_absolute() else args.out_thresholds.resolve()
    out_reports = (repo_root / args.out_reports_dir).resolve() if not args.out_reports_dir.is_absolute() else args.out_reports_dir.resolve()
    out_fig = (repo_root / args.out_fig_dir).resolve() if not args.out_fig_dir.is_absolute() else args.out_fig_dir.resolve()
    maps_root = (repo_root / args.maps_root).resolve() if not args.maps_root.is_absolute() else args.maps_root.resolve()

    _ensure_parent(out_csv)
    _ensure_parent(out_thresholds)
    _ensure_dir(out_reports)
    _ensure_dir(out_fig)
    if args.save_maps:
        _ensure_dir(maps_root)

    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.io.nifti import allclose_affine, allclose_zooms, read_array, read_header
    from mslam.metrics.uncertainty_metrics import (
        UncertaintySummary,
        finite_quantile,
        make_boundary_band,
        prob_entropy,
        prob_p1mp,
        summarize_in_mask,
        variance_across_probs,
    )

    import pandas as pd

    p2 = pd.read_csv(phase2_csv)
    p2 = p2.set_index("patient_id")

    rows = _read_manifest(manifest_path)
    ok_rows = [r for r in rows if r.get("ok", "False") == "True"]
    if args.patients.strip():
        wanted = _parse_patients_arg(args.patients)
        ok_rows = [r for r in ok_rows if r.get("patient_id") in wanted]
    if not ok_rows:
        print("No patients selected.")
        return 1

    unc_types = [u.strip() for u in args.unc_types.split(",") if u.strip()]
    allowed = {"ens_var", "prob_entropy", "prob_p1mp"}
    for u in unc_types:
        if u not in allowed:
            raise SystemExit(f"Unknown unc type '{u}'. Allowed: {', '.join(sorted(allowed))}")
    primary = args.primary_unc.strip()
    if primary not in allowed:
        raise SystemExit(f"--primary-unc must be one of: {', '.join(sorted(allowed))}")

    # Collect per-patient outputs.
    patient_rows: list[dict[str, object]] = []
    per_patient_reports: dict[str, dict[str, object]] = {}

    # For figures (examples + worst uncertainty).
    fig_cache: dict[str, dict[str, object]] = {}

    def load_prob_paths(pid: str, tp: str) -> dict[str, Path]:
        d = lstai_root / pid / tp
        paths = {
            "mask": d / "lesion_mask.nii.gz",
            "prob": d / "lesion_prob.nii.gz",
            "prob_m1": d / "lesion_prob_model1.nii.gz",
            "prob_m2": d / "lesion_prob_model2.nii.gz",
            "prob_m3": d / "lesion_prob_model3.nii.gz",
        }
        return paths

    # Build header fields for the aggregate CSV.
    fieldnames: list[str] = [
        "patient_id",
        "ok",
        "warnings",
        "qc_needs_review",
        "qc_change_not_confident",
        # Phase 2 context (for scatter + interpretability)
        "phase2_delta_lesion_vol_mm3",
        "phase2_dice_chg_sym_cons",
        "phase2_chg_sym_cons_abs_vol_err_mm3",
    ]

    # Uncertainty columns (timepoint t0/t1)
    # For each uncertainty type: mean_brain, p95_brain, mean_lesion, mean_boundary.
    for tp in ["t0", "t1"]:
        for ut in unc_types:
            fieldnames += [
                f"unc_{ut}_mean_brain_{tp}",
                f"unc_{ut}_p95_brain_{tp}",
                f"unc_{ut}_mean_lesion_{tp}",
                f"unc_{ut}_mean_boundary_{tp}",
            ]
        fieldnames += [
            f"lesion_voxels_{tp}",
            f"boundary_voxels_{tp}",
            f"brain_voxels_{tp}",
        ]

    # Main loop: compute uncertainty summaries (no QC thresholds yet).
    for r in ok_rows:
        pid = r["patient_id"]
        warnings: list[str] = []

        # Phase 2 context (may be missing if phase2 not run for this pid).
        p2_row = p2.loc[pid] if pid in p2.index else None
        deltaV = float(p2_row["delta_lesion_vol_mm3"]) if p2_row is not None else float("nan")
        dice_sym = float(p2_row["dice_chg_sym_cons"]) if p2_row is not None else float("nan")
        abs_vol_err = float(abs(p2_row["chg_sym_cons_vol_err_mm3"])) if p2_row is not None else float("nan")

        patient_out: dict[str, object] = {k: "" for k in fieldnames}
        patient_out.update(
            {
                "patient_id": pid,
                "ok": "True",
                "warnings": "",
                "qc_needs_review": "",
                "qc_change_not_confident": "",
                "phase2_delta_lesion_vol_mm3": _safe_float(deltaV),
                "phase2_dice_chg_sym_cons": _safe_float(dice_sym),
                "phase2_chg_sym_cons_abs_vol_err_mm3": _safe_float(abs_vol_err),
            }
        )

        # Per-patient report skeleton
        rep: dict[str, object] = {
            "patient_id": pid,
            "ok": True,
            "meta": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "primary_uncertainty": primary,
                "uncertainty_types": unc_types,
                "boundary_radius_mm": float(args.boundary_radius_mm),
                "qc_quantile": float(args.qc_quantile),
                "change_quantile": float(args.change_quantile),
                "thresholds_path": str(out_thresholds.relative_to(repo_root)),
            },
            "inputs": {},
            "warnings": warnings,
            "context_phase2": {
                "delta_lesion_vol_mm3": _safe_float(deltaV),
                "dice_chg_sym_cons": _safe_float(dice_sym),
                "abs_vol_err_chg_sym_cons_mm3": _safe_float(abs_vol_err),
            },
            "uncertainty": {"maps": {"t0": {}, "t1": {}}, "summaries": {"t0": {}, "t1": {}}},
            "qc": {},
        }

        # Reference grid: use t0 FLAIR header for consistency checks.
        flair0_path = (repo_root / r["t0_flair"]).resolve()
        hdr_ref = read_header(flair0_path)
        spacing = hdr_ref.zooms

        # Load shared brainmask once.
        brainmask_path = (repo_root / r["brainmask"]).resolve()
        hdr_bm = read_header(brainmask_path)
        if hdr_bm.shape != hdr_ref.shape or not allclose_affine(hdr_bm.affine, hdr_ref.affine, atol=1e-3):
            warnings.append("grid_mismatch:brainmask")
        _, brainmask = read_array(brainmask_path)

        for tp in ["t0", "t1"]:
            flair_path = (repo_root / r["t0_flair"]).resolve() if tp == "t0" else (repo_root / r["t1_flair"]).resolve()
            paths = load_prob_paths(pid, tp)
            rep["inputs"].setdefault(tp, {})
            rep["inputs"][tp] = {
                "flair": str(flair_path.relative_to(repo_root)),
                "lesion_mask": str(paths["mask"].relative_to(repo_root)),
                "lesion_prob": str(paths["prob"].relative_to(repo_root)),
                "lesion_prob_model1": str(paths["prob_m1"].relative_to(repo_root)),
                "lesion_prob_model2": str(paths["prob_m2"].relative_to(repo_root)),
                "lesion_prob_model3": str(paths["prob_m3"].relative_to(repo_root)),
            }

            # Grid checks
            for p in [flair_path, paths["mask"], paths["prob"]]:
                if not p.exists():
                    warnings.append(f"missing:{p.name}:{tp}")
                    continue
                hdr = read_header(p)
                if hdr.shape != hdr_ref.shape:
                    warnings.append(f"shape_mismatch:{p.name}:{tp}")
                if not allclose_affine(hdr.affine, hdr_ref.affine, atol=1e-3):
                    warnings.append(f"affine_mismatch:{p.name}:{tp}")
                if not allclose_zooms(hdr.zooms, hdr_ref.zooms, atol=1e-5):
                    warnings.append(f"spacing_diff:{p.name}:{tp}")

            if not (paths["mask"].exists() and paths["prob"].exists() and flair_path.exists()):
                patient_out["ok"] = "False"
                rep["ok"] = False
                continue

            _, flair = read_array(flair_path)
            _, lesion_mask = read_array(paths["mask"])
            _, prob = read_array(paths["prob"])

            lesion_b = (lesion_mask > 0)
            brain_b = (brainmask > 0)

            # Boundary band (outer) within brain.
            boundary = make_boundary_band(
                lesion_b,
                brainmask=brain_b,
                spacing_xyz=spacing,
                radius_mm=float(args.boundary_radius_mm),
            )

            # Uncertainty maps
            unc_maps: dict[str, object] = {}
            if "prob_p1mp" in unc_types:
                unc_maps["prob_p1mp"] = prob_p1mp(prob)
            if "prob_entropy" in unc_types:
                unc_maps["prob_entropy"] = prob_entropy(prob)
            if "ens_var" in unc_types:
                sub_probs: list[object] = []
                for k in ["prob_m1", "prob_m2", "prob_m3"]:
                    pth = paths[k]
                    if pth.exists():
                        _, arr = read_array(pth)
                        sub_probs.append(arr)
                    else:
                        warnings.append(f"missing:{pth.name}:{tp}")
                if len(sub_probs) >= 2:
                    unc_maps["ens_var"] = variance_across_probs([p for p in sub_probs])  # type: ignore[arg-type]
                else:
                    warnings.append(f"insufficient_submodels:{tp}")
                    unc_maps["ens_var"] = None

            # Summaries
            brain_n = int((brain_b > 0).sum())
            lesion_n = int(lesion_b.sum())
            boundary_n = int(boundary.sum())
            patient_out[f"brain_voxels_{tp}"] = brain_n
            patient_out[f"lesion_voxels_{tp}"] = lesion_n
            patient_out[f"boundary_voxels_{tp}"] = boundary_n

            rep["uncertainty"]["summaries"][tp] = {}
            rep["uncertainty"]["maps"][tp] = {}

            for ut, umap in unc_maps.items():
                if umap is None:
                    # Fill with NaNs
                    patient_out[f"unc_{ut}_mean_brain_{tp}"] = float("nan")
                    patient_out[f"unc_{ut}_p95_brain_{tp}"] = float("nan")
                    patient_out[f"unc_{ut}_mean_lesion_{tp}"] = float("nan")
                    patient_out[f"unc_{ut}_mean_boundary_{tp}"] = float("nan")
                    rep["uncertainty"]["summaries"][tp][ut] = {
                        "brain": {"mean": float("nan"), "p95": float("nan"), "n_voxels": brain_n},
                        "lesion": {"mean": float("nan"), "p95": float("nan"), "n_voxels": lesion_n},
                        "boundary": {"mean": float("nan"), "p95": float("nan"), "n_voxels": boundary_n},
                    }
                    continue

                # Ensure float map
                um = umap  # array

                s_brain: UncertaintySummary = summarize_in_mask(um, brain_b, p=95.0)
                s_lesion: UncertaintySummary = summarize_in_mask(um, lesion_b & brain_b, p=95.0)
                s_bnd: UncertaintySummary = summarize_in_mask(um, boundary, p=95.0)

                patient_out[f"unc_{ut}_mean_brain_{tp}"] = _safe_float(s_brain.mean)
                patient_out[f"unc_{ut}_p95_brain_{tp}"] = _safe_float(s_brain.p95)
                patient_out[f"unc_{ut}_mean_lesion_{tp}"] = _safe_float(s_lesion.mean)
                patient_out[f"unc_{ut}_mean_boundary_{tp}"] = _safe_float(s_bnd.mean)

                rep["uncertainty"]["summaries"][tp][ut] = {
                    "brain": {"mean": _safe_float(s_brain.mean), "p95": _safe_float(s_brain.p95), "n_voxels": int(s_brain.n_voxels)},
                    "lesion": {"mean": _safe_float(s_lesion.mean), "p95": _safe_float(s_lesion.p95), "n_voxels": int(s_lesion.n_voxels)},
                    "boundary": {"mean": _safe_float(s_bnd.mean), "p95": _safe_float(s_bnd.p95), "n_voxels": int(s_bnd.n_voxels)},
                }

                if args.save_maps:
                    out_map_dir = maps_root / pid / tp
                    _ensure_dir(out_map_dir)
                    out_map_path = out_map_dir / f"unc_{ut}.nii.gz"
                    _write_nifti_like(paths["prob"], out_map_path, um)
                    rep["uncertainty"]["maps"][tp][ut] = str(out_map_path.relative_to(repo_root))

            # Cache arrays for example overlays (t1 only).
            if tp == "t1":
                # Choose z slice: maximize lesion voxels, else maximize brain voxels.
                import numpy as np

                if bool(lesion_b.any()):
                    per_z = lesion_b.sum(axis=(0, 1))
                    z = int(np.argmax(per_z))
                else:
                    per_z = brain_b.sum(axis=(0, 1))
                    z = int(np.argmax(per_z)) if per_z.size else int(flair.shape[2] // 2)

                # Primary map for overlay
                umap = unc_maps.get(primary)
                if umap is None:
                    umap = unc_maps.get("prob_entropy", prob_entropy(prob))
                fig_cache[pid] = {
                    "flair_t1": flair,
                    "brainmask": brain_b,
                    "lesion_mask_t1": lesion_b,
                    "boundary_mask_t1": boundary,
                    "unc_map_t1": umap,
                    "z": z,
                }

        patient_out["warnings"] = ";".join(warnings)
        patient_rows.append(patient_out)
        per_patient_reports[pid] = rep

    # Compute QC thresholds (cohort quantiles)
    # Use primary uncertainty type at t1.
    primary_col_mean_les = f"unc_{primary}_mean_lesion_t1"
    primary_col_p95_brain = f"unc_{primary}_p95_brain_t1"

    values_mean_les = [float(r.get(primary_col_mean_les, float("nan"))) for r in patient_rows]
    values_p95_brain = [float(r.get(primary_col_p95_brain, float("nan"))) for r in patient_rows]
    thr_mean_les = finite_quantile(values_mean_les, float(args.qc_quantile))
    thr_p95_brain = finite_quantile(values_p95_brain, float(args.qc_quantile))

    abs_deltaV = [abs(float(r.get("phase2_delta_lesion_vol_mm3", float("nan")))) for r in patient_rows]
    thr_abs_dv = finite_quantile(abs_deltaV, float(args.change_quantile))

    thresholds = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cohort_n": int(len(patient_rows)),
        "primary_uncertainty": primary,
        "qc_quantile": float(args.qc_quantile),
        "change_quantile": float(args.change_quantile),
        "thresholds": {
            "unc_mean_lesion_t1": {"column": primary_col_mean_les, "value": _safe_float(thr_mean_les)},
            "unc_p95_brain_t1": {"column": primary_col_p95_brain, "value": _safe_float(thr_p95_brain)},
            "abs_delta_lesion_vol_mm3": {"column": "phase2_delta_lesion_vol_mm3", "value": _safe_float(thr_abs_dv)},
        },
    }
    _save_json(out_thresholds, thresholds)

    # Apply QC flags + write per-patient reports.
    for row in patient_rows:
        pid = str(row["patient_id"])
        unc_mean_les = float(row.get(primary_col_mean_les, float("nan")))
        unc_p95_br = float(row.get(primary_col_p95_brain, float("nan")))
        dv = float(row.get("phase2_delta_lesion_vol_mm3", float("nan")))

        triggers: list[str] = []
        needs_review = False
        if math.isfinite(unc_mean_les) and math.isfinite(thr_mean_les) and unc_mean_les > thr_mean_les:
            needs_review = True
            triggers.append("unc_mean_lesion_t1_high")
        if math.isfinite(unc_p95_br) and math.isfinite(thr_p95_brain) and unc_p95_br > thr_p95_brain:
            needs_review = True
            triggers.append("unc_p95_brain_t1_high")

        change_not_confident = False
        if math.isfinite(dv) and math.isfinite(thr_abs_dv) and abs(dv) > thr_abs_dv and needs_review:
            change_not_confident = True
            triggers.append("large_change_and_high_uncertainty")

        row["qc_needs_review"] = str(needs_review)
        row["qc_change_not_confident"] = str(change_not_confident)

        rep = per_patient_reports[pid]
        rep["qc"] = {
            "thresholds_used": thresholds["thresholds"],
            "flags": {"needs_review": bool(needs_review), "change_not_confident": bool(change_not_confident)},
            "triggers": triggers,
        }
        report_path = out_reports / f"{pid}.json"
        _save_json(report_path, rep)

    # Write aggregate CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in patient_rows:
            writer.writerow(row)

    # Figures: uncertainty overlay + scatter
    example_ids = [p.strip() for p in args.example_patients.split(",") if p.strip()]
    ex_ids = [p for p in example_ids if p in fig_cache]
    # Add one "high-uncertainty" example by lesion mean and one by brain p95 (t1).
    best_pid_mean = ""
    best_val_mean = -1.0
    best_pid_p95 = ""
    best_val_p95 = -1.0
    for row in patient_rows:
        pid = str(row["patient_id"])
        v_mean = float(row.get(primary_col_mean_les, float("nan")))
        if math.isfinite(v_mean) and v_mean > best_val_mean and pid in fig_cache:
            best_pid_mean = pid
            best_val_mean = v_mean
        v_p95 = float(row.get(primary_col_p95_brain, float("nan")))
        if math.isfinite(v_p95) and v_p95 > best_val_p95 and pid in fig_cache:
            best_pid_p95 = pid
            best_val_p95 = v_p95
    for pid in [best_pid_mean, best_pid_p95]:
        if pid and pid not in ex_ids:
            ex_ids.append(pid)
    if ex_ids:
        # Attach QC + scalar summaries to each cache entry (used in titles).
        by_pid = {str(r["patient_id"]): r for r in patient_rows}
        rows_by_patient = {}
        for pid in ex_ids:
            d = dict(fig_cache[pid])
            r = by_pid.get(pid, {})
            d["qc_needs_review"] = (str(r.get("qc_needs_review", "False")) == "True")
            d["unc_mean_lesion"] = float(r.get(primary_col_mean_les, float("nan")))
            d["unc_p95_brain"] = float(r.get(primary_col_p95_brain, float("nan")))
            rows_by_patient[pid] = d
        _make_uncertainty_overlay_figure(
            patient_ids=ex_ids,
            rows_by_patient=rows_by_patient,
            out_path=out_fig / "phase4_unc_overlay.png",
            title=f"Phase 4: uncertainty overlay (primary={primary})",
            scale=args.scale,
        )

    # Scatter: uncertainty vs Phase 2 error
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs: list[float] = []
    ys: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    for row in patient_rows:
        pid = str(row["patient_id"])
        x = float(row.get(primary_col_mean_les, float("nan")))
        dice = float(row.get("phase2_dice_chg_sym_cons", float("nan")))
        if not (math.isfinite(x) and math.isfinite(dice)):
            continue
        y = 1.0 - float(dice)
        xs.append(x)
        ys.append(y)
        labels.append(pid)
        colors.append("#D55E00" if str(row.get("qc_needs_review", "False")) == "True" else "#4C72B0")

    if xs and ys:
        fig = plt.figure(figsize=(6.5, 5.0), layout="constrained")
        ax = fig.subplots()
        ax.scatter(xs, ys, c=colors, alpha=0.9, edgecolors="black", linewidths=0.4)
        ax.set_title("Phase 4: uncertainty vs error (Phase 2)")
        ax.set_xlabel(f"{primary_col_mean_les} (mean lesion uncertainty @ t1)")
        ax.set_ylabel("1 - dice_chg_sym_cons")

        # Annotate points for interpretability.
        xs_a = np.array(xs)
        ys_a = np.array(ys)
        idxs: set[int] = set()
        if args.scatter_annotate != "none":
            idxs.add(int(np.argmax(xs_a)))
            idxs.add(int(np.argmax(ys_a)))
        if args.scatter_annotate == "all":
            idxs.update(range(len(labels)))
        elif args.scatter_annotate == "qc":
            for i, c in enumerate(colors):
                if c == "#D55E00":
                    idxs.add(i)
        # extremes-only is handled by the default idxs above.

        fs = 7 if args.scatter_annotate == "all" else 9
        for i in sorted(idxs):
            ax.annotate(labels[i], (xs[i], ys[i]), textcoords="offset points", xytext=(5, 5), fontsize=fs)

        fig.savefig(out_fig / "phase4_unc_vs_error.png", dpi=200)
        plt.close(fig)

    print(f"Wrote {out_csv.relative_to(repo_root).as_posix()}")
    print(f"Wrote {out_thresholds.relative_to(repo_root).as_posix()}")
    print(f"Wrote reports -> {out_reports.relative_to(repo_root).as_posix()}/")
    print(f"Wrote figures -> {out_fig.relative_to(repo_root).as_posix()}/ (phase4_*.png)")
    if args.save_maps:
        print(f"Wrote maps -> {maps_root.relative_to(repo_root).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
