from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for x in _parse_csv_list(s):
        out.append(int(x))
    return out


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_run_complete(run_dir: Path, probability_map: bool) -> bool:
    meta_path = run_dir / "lstai_run.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        if int(meta.get("returncode", 1)) != 0:
            return False
    except Exception:
        return False

    required = [run_dir / "lesion_mask.nii.gz"]
    if probability_map:
        required += [
            run_dir / "lesion_prob.nii.gz",
            run_dir / "lesion_prob_model1.nii.gz",
            run_dir / "lesion_prob_model2.nii.gz",
            run_dir / "lesion_prob_model3.nii.gz",
        ]
    return all(p.exists() for p in required)


def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2))


def _save_nifti_like(ref_path: Path, out_path: Path, data) -> None:
    import nibabel as nib
    import numpy as np

    ref_img = nib.load(str(ref_path))
    arr = np.asarray(data)
    if arr.shape != tuple(ref_img.shape[:3]):
        raise ValueError(f"Shape mismatch for write: ref={ref_img.shape} vs data={arr.shape}")
    out_img = nib.Nifti1Image(arr.astype(np.float32, copy=False), ref_img.affine, ref_img.header)
    out_img.header.set_data_dtype(np.float32)
    # Preserve zooms if possible.
    try:
        out_img.header.set_zooms(ref_img.header.get_zooms()[:3])
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(out_path))


def _bool(a) -> "np.ndarray":
    import numpy as np

    return (np.asarray(a) > 0)


def _compute_metrics(
    *,
    patient_id: str,
    flair0_path: Path,
    flair1_path: Path,
    brainmask_path: Path,
    gt_path: Path,
    mask0_path: Path,
    mask1_path: Path,
    new_dilation_radius_mm: float,
    min_component_volume_mm3: float,
    connectivity: int,
    intensity_p_lo: float,
    intensity_p_hi: float,
    diff_threshold: float,
) -> tuple[dict[str, float | str | bool], list[str]]:
    import numpy as np

    from mslam.io.nifti import allclose_affine, allclose_zooms, read_array, read_header
    from mslam.metrics.longitudinal_metrics import (
        dice_coefficient,
        dilate_binary_mm,
        intensity_change_stats,
        remove_small_components,
        volume_mm3,
    )

    warnings: list[str] = []
    hdr_ref = read_header(flair0_path)
    spacing = hdr_ref.zooms
    voxel_vol = hdr_ref.voxel_volume_mm3

    for p in [flair1_path, brainmask_path, gt_path, mask0_path, mask1_path]:
        hdr = read_header(p)
        if hdr.shape != hdr_ref.shape:
            warnings.append(f"shape_mismatch:{p.name}")
        if not allclose_affine(hdr.affine, hdr_ref.affine, atol=1e-3):
            warnings.append(f"affine_mismatch:{p.name}")
        if not allclose_zooms(hdr.zooms, hdr_ref.zooms, atol=1e-5):
            warnings.append(f"spacing_diff:{p.name}")

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

    # Base volumes
    v0 = volume_mm3(m0, voxel_vol)
    v1 = volume_mm3(m1, voxel_vol)
    dv = v1 - v0
    gt_vol = volume_mm3(gt_b, voxel_vol)

    # Conservative proxies (monitoring-friendly)
    m0_dil = dilate_binary_mm(m0, spacing_xyz=spacing, radius_mm=float(new_dilation_radius_mm))
    new_cons_pre = m1 & (~m0_dil)
    min_voxels = int(math.ceil(float(min_component_volume_mm3) / float(voxel_vol))) if voxel_vol > 0 else 1
    new_cons = remove_small_components(new_cons_pre, min_voxels=min_voxels, connectivity=int(connectivity))
    chg_sym_cons_pre = np.logical_xor(m1, m0_dil)
    chg_sym_cons = remove_small_components(chg_sym_cons_pre, min_voxels=min_voxels, connectivity=int(connectivity))

    # Dice to change-GT
    dice_new_cons = dice_coefficient(new_cons, gt_b)
    dice_sym_cons = dice_coefficient(chg_sym_cons, gt_b)

    # Intensity change stats (normalized, within brainmask)
    t0_stats, t1_stats, diff_stats = intensity_change_stats(
        flair0,
        flair1,
        bm,
        p_lo=float(intensity_p_lo),
        p_hi=float(intensity_p_hi),
        diff_threshold=float(diff_threshold),
    )

    out: dict[str, float | str | bool] = {
        "patient_id": patient_id,
        "voxel_spacing_x": float(spacing[0]),
        "voxel_spacing_y": float(spacing[1]),
        "voxel_spacing_z": float(spacing[2]),
        "voxel_vol_mm3": float(voxel_vol),
        "lesion_vol_t0_mm3": float(v0),
        "lesion_vol_t1_mm3": float(v1),
        "delta_lesion_vol_mm3": float(dv),
        "change_gt_vol_mm3": float(gt_vol),
        "dice_new_cons": float(dice_new_cons),
        "dice_chg_sym_cons": float(dice_sym_cons),
        "intensity_diff_mean": float(diff_stats["mean"]),
        "intensity_diff_p95": float(diff_stats["p95"]),
        "t0_flair_median": float(t0_stats["median"]),
        "t1_flair_median": float(t1_stats["median"]),
    }
    return out, warnings


def _make_patient_change_figure(
    *,
    patient_id: str,
    flair0,
    flair1,
    brain,
    mask0,
    mask1,
    gt,
    new_mask,
    diff_norm,
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

    bm = (brain > 0)
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

    f0 = np.array(flair0[:, :, z], dtype=np.float32, copy=False).copy()
    f1 = np.array(flair1[:, :, z], dtype=np.float32, copy=False).copy()
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

    d = np.array(diff_norm[:, :, z], dtype=np.float32, copy=False).copy()
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

    fig.suptitle(f"{patient_id} (z={z})  {title}", fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: robustness shift suite (shift_v1) + LST-AI + longitudinal metrics")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/tables/phase0_manifest_checked.csv"),
        help="Manifest with t0/t1 (T1W/FLAIR), brainmask, gt_change, ok.",
    )
    parser.add_argument(
        "--baseline-lstai-root",
        type=Path,
        default=Path("data/processed/lstai_outputs"),
        help="Baseline LST-AI outputs root (Phase 1).",
    )
    parser.add_argument(
        "--shift-input-root",
        type=Path,
        default=Path("data/processed/shift_v1"),
        help="Where to cache shifted images (not tracked).",
    )
    parser.add_argument(
        "--shift-lstai-root",
        type=Path,
        default=Path("data/processed/lstai_outputs_shift_v1"),
        help="Where to write LST-AI outputs for shifted inputs (not tracked).",
    )
    parser.add_argument("--mode", choices=["t1_only", "both"], default="t1_only")
    parser.add_argument(
        "--shifts",
        type=str,
        default="gamma,noise,resolution",
        help="Comma-separated shifts (subset of: gamma,noise,blur,resolution).",
    )
    parser.add_argument("--levels", type=str, default="0,1,2", help="Comma-separated levels to run (default: 0,1,2).")
    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient ids (default: all ok).")
    parser.add_argument("--limit", type=int, default=0, help="If >0, limit number of patients (after filtering).")

    # LST-AI docker settings (match Phase 1 defaults but allow speed-oriented overrides).
    parser.add_argument("--image", type=str, default="jqmcginnis/lst-ai:v1.2.0")
    parser.add_argument("--platform", type=str, default="linux/amd64")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fast-mode", action="store_true", help="Use LST-AI --fast-mode (faster hd-bet).")
    parser.add_argument("--probability-map", action="store_true", help="Store probability maps (slower, bigger).")
    parser.add_argument(
        "--no-segment-only",
        action="store_true",
        help="Disable LST-AI --segment_only (will also run lesion annotation; slower).",
    )

    # Metrics parameters (match Phase 2 defaults).
    parser.add_argument("--new-dilation-radius-mm", type=float, default=2.0)
    parser.add_argument("--min-component-volume-mm3", type=float, default=10.0)
    parser.add_argument("--connectivity", choices=["6", "26"], default="26")
    parser.add_argument("--intensity-p-lo", type=float, default=1.0)
    parser.add_argument("--intensity-p-hi", type=float, default=99.0)
    parser.add_argument("--diff-threshold", type=float, default=0.2)

    # Outputs
    parser.add_argument("--out-csv", type=Path, default=Path("results/tables/phase3_robustness.csv"))
    parser.add_argument("--out-summary-csv", type=Path, default=Path("results/tables/phase3_robustness_summary.csv"))
    parser.add_argument("--runlog", type=Path, default=Path("results/tables/phase3_runlog.csv"))
    parser.add_argument("--out-fig-dir", type=Path, default=Path("results/figures"))

    parser.add_argument("--overwrite-shifts", action="store_true", help="Overwrite shifted inputs on disk.")
    parser.add_argument("--overwrite-results", action="store_true", help="Overwrite phase3_robustness.csv (recommended).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing LST-AI run dirs.")
    parser.add_argument("--rerun", action="store_true", help="Rerun inference even if output exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--scale", choices=["per-timepoint", "shared"], default="per-timepoint")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    baseline_root = (repo_root / args.baseline_lstai_root).resolve() if not args.baseline_lstai_root.is_absolute() else args.baseline_lstai_root.resolve()
    shift_input_root = (repo_root / args.shift_input_root).resolve() if not args.shift_input_root.is_absolute() else args.shift_input_root.resolve()
    shift_lstai_root = (repo_root / args.shift_lstai_root).resolve() if not args.shift_lstai_root.is_absolute() else args.shift_lstai_root.resolve()
    out_csv = (repo_root / args.out_csv).resolve() if not args.out_csv.is_absolute() else args.out_csv.resolve()
    out_summary_csv = (repo_root / args.out_summary_csv).resolve() if not args.out_summary_csv.is_absolute() else args.out_summary_csv.resolve()
    runlog_path = (repo_root / args.runlog).resolve() if not args.runlog.is_absolute() else args.runlog.resolve()
    out_fig = (repo_root / args.out_fig_dir).resolve() if not args.out_fig_dir.is_absolute() else args.out_fig_dir.resolve()
    _ensure_parent(out_csv)
    _ensure_parent(out_summary_csv)
    _ensure_parent(runlog_path)
    _ensure_dir(out_fig)

    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.engines.lstai import LstAiDockerConfig, run_lstai_docker
    from mslam.io.nifti import read_array, read_header
    from mslam.preprocessing.shift_suite import SHIFT_V1, apply_shift, stable_seed

    rows = _read_manifest(manifest_path)
    ok_rows = [r for r in rows if r.get("ok", "False") == "True"]
    if args.patients.strip():
        wanted = set(_parse_csv_list(args.patients))
        ok_rows = [r for r in ok_rows if r.get("patient_id") in wanted]
    if args.limit and args.limit > 0:
        ok_rows = ok_rows[: int(args.limit)]
    if not ok_rows:
        print("No patients selected.")
        return 1

    shifts = _parse_csv_list(args.shifts)
    for s in shifts:
        if s not in SHIFT_V1:
            raise SystemExit(f"Unknown shift '{s}'. Available: {', '.join(sorted(SHIFT_V1))}")
    levels = _parse_int_list(args.levels)

    cfg = LstAiDockerConfig(
        image=args.image,
        platform=args.platform,
        device=args.device,
        threads=(None if int(args.threads) == 0 else int(args.threads)),
        segment_only=not bool(args.no_segment_only),
        fast_mode=bool(args.fast_mode),
        probability_map=bool(args.probability_map),
    )

    # Prepare baseline metrics (computed from baseline LST-AI outputs + original images).
    baseline_metrics: dict[str, dict[str, float | str | bool]] = {}
    baseline_warnings: dict[str, list[str]] = {}

    for r in ok_rows:
        pid = r["patient_id"]
        flair0 = (repo_root / r["t0_flair"]).resolve()
        flair1 = (repo_root / r["t1_flair"]).resolve()
        brainmask = (repo_root / r["brainmask"]).resolve()
        gt = (repo_root / r["gt_change"]).resolve()
        mask0 = (baseline_root / pid / "t0" / "lesion_mask.nii.gz").resolve()
        mask1 = (baseline_root / pid / "t1" / "lesion_mask.nii.gz").resolve()
        if not (mask0.exists() and mask1.exists()):
            raise FileNotFoundError(f"Missing baseline LST-AI outputs for {pid}: {mask0} / {mask1}")
        m, warns = _compute_metrics(
            patient_id=pid,
            flair0_path=flair0,
            flair1_path=flair1,
            brainmask_path=brainmask,
            gt_path=gt,
            mask0_path=mask0,
            mask1_path=mask1,
            new_dilation_radius_mm=float(args.new_dilation_radius_mm),
            min_component_volume_mm3=float(args.min_component_volume_mm3),
            connectivity=int(args.connectivity),
            intensity_p_lo=float(args.intensity_p_lo),
            intensity_p_hi=float(args.intensity_p_hi),
            diff_threshold=float(args.diff_threshold),
        )
        baseline_metrics[pid] = m
        baseline_warnings[pid] = warns

    # Runlog CSV for LST-AI calls under shift suite.
    runlog_fields = [
        "timestamp",
        "patient_id",
        "mode",
        "shift",
        "level",
        "timepoint",
        "ok",
        "returncode",
        "runtime_sec",
        "t1_path",
        "flair_path",
        "run_dir",
        "metadata_json",
        "stderr_log",
        "stdout_log",
        "cmd_json",
        "skipped",
        "error",
    ]
    write_runlog_header = not runlog_path.exists()
    runlog_f = runlog_path.open("a", newline="")
    runlog_writer = csv.DictWriter(runlog_f, fieldnames=runlog_fields)
    if write_runlog_header:
        runlog_writer.writeheader()

    # Robustness output CSV.
    out_fields = [
        "patient_id",
        "ok",
        "warnings",
        "mode",
        "shift",
        "level",
        "shifted_timepoints",
        "delta_lesion_vol_mm3_base",
        "delta_lesion_vol_mm3",
        "deltaV_minus_base_mm3",
        "abs_deltaV_minus_base_mm3",
        "dice_chg_sym_cons_base",
        "dice_chg_sym_cons",
        "dice_sym_minus_base",
        "dice_new_cons_base",
        "dice_new_cons",
        "dice_new_minus_base",
        "lesion_vol_t0_mm3",
        "lesion_vol_t1_mm3",
        "change_gt_vol_mm3",
        "intensity_diff_mean",
        "intensity_diff_p95",
        "t0_flair_median",
        "t1_flair_median",
        "inference_runtime_sec",
        "shift_meta_json",
    ]

    _ensure_parent(out_csv)
    completed: set[tuple[str, str, str, int]] = set()
    if out_csv.exists() and not args.overwrite_results:
        try:
            with out_csv.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("ok", "False") == "True":
                        completed.add(
                            (
                                str(row.get("patient_id", "")),
                                str(row.get("mode", "")),
                                str(row.get("shift", "")),
                                int(row.get("level", "0")),
                            )
                        )
        except Exception:
            completed = set()

    write_out_header = (not out_csv.exists()) or bool(args.overwrite_results)
    out_f = out_csv.open(("w" if args.overwrite_results else "a"), newline="")
    out_writer = csv.DictWriter(out_f, fieldnames=out_fields)
    if write_out_header:
        out_writer.writeheader()

    def infer_one(
        *,
        pid: str,
        shift: str,
        level: int,
        tp: str,
        t1_path: Path,
        flair_path: Path,
        run_dir: Path,
    ) -> tuple[bool, float, str]:
        if _is_run_complete(run_dir, cfg.probability_map) and not (args.rerun or args.overwrite):
            return True, 0.0, "skipped"
        if args.dry_run:
            return True, 0.0, "dry-run"
        started = time.time()
        try:
            res = run_lstai_docker(
                t1_path=t1_path,
                flair_path=flair_path,
                run_dir=run_dir,
                cfg=cfg,
                keep_temp=False,
                overwrite=args.overwrite,
                timeout_sec=None,
            )
            ok = (res.returncode == 0)
            runtime = time.time() - started
            row = {k: "" for k in runlog_fields}
            row.update(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_id": pid,
                    "mode": args.mode,
                    "shift": shift,
                    "level": str(level),
                    "timepoint": tp,
                    "ok": str(ok),
                    "returncode": str(res.returncode),
                    "runtime_sec": f"{res.runtime_sec:.3f}",
                    "t1_path": str(t1_path),
                    "flair_path": str(flair_path),
                    "run_dir": str(run_dir),
                    "metadata_json": str(res.metadata_json),
                    "stderr_log": str(res.stderr_log),
                    "stdout_log": str(res.stdout_log),
                    "cmd_json": json.dumps(res.cmd),
                    "skipped": "False",
                }
            )
            runlog_writer.writerow(row)
            runlog_f.flush()
            return ok, float(runtime), ""
        except Exception as e:
            runtime = time.time() - started
            row = {k: "" for k in runlog_fields}
            row.update(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_id": pid,
                    "mode": args.mode,
                    "shift": shift,
                    "level": str(level),
                    "timepoint": tp,
                    "ok": "False",
                    "returncode": "",
                    "runtime_sec": f"{runtime:.3f}",
                    "t1_path": str(t1_path),
                    "flair_path": str(flair_path),
                    "run_dir": str(run_dir),
                    "skipped": "False",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            runlog_writer.writerow(row)
            runlog_f.flush()
            if args.fail_fast:
                raise
            return False, float(runtime), row["error"]

    # Main loop.
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Phase 3 started at {started_at}. patients={len(ok_rows)} mode={args.mode} shifts={shifts} levels={levels}")

    for shift_name in shifts:
        for level in levels:
            for r in ok_rows:
                pid = r["patient_id"]
                if (
                    (pid, args.mode, shift_name, int(level)) in completed
                    and not (args.rerun or args.overwrite or args.overwrite_results)
                ):
                    continue
                base = baseline_metrics[pid]
                warns = list(baseline_warnings.get(pid, []))

                shifted_tps: list[str] = []
                shift_meta_path = ""
                infer_runtime = 0.0

                flair0_path = (repo_root / r["t0_flair"]).resolve()
                flair1_path = (repo_root / r["t1_flair"]).resolve()
                t10_path = (repo_root / r["t0_t1w"]).resolve()
                t11_path = (repo_root / r["t1_t1w"]).resolve()
                brainmask_path = (repo_root / r["brainmask"]).resolve()
                gt_path = (repo_root / r["gt_change"]).resolve()

                # Determine which timepoints are shifted
                if int(level) > 0:
                    if args.mode == "t1_only":
                        shifted_tps = ["t1"]
                    else:
                        shifted_tps = ["t0", "t1"]

                # Level 0: just copy baseline metrics into this shift/level row.
                if int(level) == 0:
                    row = {
                        "patient_id": pid,
                        "ok": "True",
                        "warnings": ";".join(warns),
                        "mode": args.mode,
                        "shift": shift_name,
                        "level": str(level),
                        "shifted_timepoints": "",
                        "delta_lesion_vol_mm3_base": base["delta_lesion_vol_mm3"],
                        "delta_lesion_vol_mm3": base["delta_lesion_vol_mm3"],
                        "deltaV_minus_base_mm3": 0.0,
                        "abs_deltaV_minus_base_mm3": 0.0,
                        "dice_chg_sym_cons_base": base["dice_chg_sym_cons"],
                        "dice_chg_sym_cons": base["dice_chg_sym_cons"],
                        "dice_sym_minus_base": 0.0,
                        "dice_new_cons_base": base["dice_new_cons"],
                        "dice_new_cons": base["dice_new_cons"],
                        "dice_new_minus_base": 0.0,
                        "lesion_vol_t0_mm3": base["lesion_vol_t0_mm3"],
                        "lesion_vol_t1_mm3": base["lesion_vol_t1_mm3"],
                        "change_gt_vol_mm3": base["change_gt_vol_mm3"],
                        "intensity_diff_mean": base["intensity_diff_mean"],
                        "intensity_diff_p95": base["intensity_diff_p95"],
                        "t0_flair_median": base["t0_flair_median"],
                        "t1_flair_median": base["t1_flair_median"],
                        "inference_runtime_sec": 0.0,
                        "shift_meta_json": "",
                    }
                    out_writer.writerow(row)
                    out_f.flush()
                    continue

                # Generate shifted inputs (cached on disk).
                hdr = read_header(flair0_path)
                spacing = hdr.zooms
                _, brainmask = read_array(brainmask_path)
                bm = (brainmask > 0)

                def prepare_inputs(tp: str) -> tuple[Path, Path, str]:
                    src_t1 = t10_path if tp == "t0" else t11_path
                    src_flair = flair0_path if tp == "t0" else flair1_path
                    out_dir = shift_input_root / pid / args.mode / shift_name / f"level{level}" / tp
                    out_t1 = out_dir / "t1w.nii.gz"
                    out_flair = out_dir / "flair.nii.gz"
                    meta_path = out_dir / "shift_meta.json"

                    if (not args.overwrite_shifts) and out_t1.exists() and out_flair.exists() and meta_path.exists():
                        return out_t1, out_flair, str(meta_path)

                    if args.dry_run:
                        return out_t1, out_flair, str(meta_path)

                    _, t1_arr = read_array(src_t1)
                    _, fl_arr = read_array(src_flair)
                    seed_t1 = stable_seed(pid, args.mode, shift_name, str(level), tp, "t1w")
                    seed_fl = stable_seed(pid, args.mode, shift_name, str(level), tp, "flair")
                    t1_shifted, meta_t1 = apply_shift(
                        t1_arr,
                        brainmask=bm,
                        spacing_xyz=spacing,
                        shift=shift_name,
                        level=int(level),
                        seed=int(seed_t1),
                        clip_p_lo=1.0,
                        clip_p_hi=99.0,
                    )
                    fl_shifted, meta_fl = apply_shift(
                        fl_arr,
                        brainmask=bm,
                        spacing_xyz=spacing,
                        shift=shift_name,
                        level=int(level),
                        seed=int(seed_fl),
                        clip_p_lo=1.0,
                        clip_p_hi=99.0,
                    )
                    _save_nifti_like(src_t1, out_t1, t1_shifted)
                    _save_nifti_like(src_flair, out_flair, fl_shifted)
                    _save_json(meta_path, {"t1w": meta_t1, "flair": meta_fl})
                    return out_t1, out_flair, str(meta_path)

                # Pick which images/masks to use for evaluation.
                # For t1_only: only shift t1 images and rerun t1 segmentation.
                # For both: shift both timepoints and rerun both segmentations.
                t0_t1_in = t10_path
                t0_fl_in = flair0_path
                t1_t1_in = t11_path
                t1_fl_in = flair1_path

                if "t0" in shifted_tps:
                    t0_t1_in, t0_fl_in, meta0 = prepare_inputs("t0")
                    shift_meta_path = meta0
                if "t1" in shifted_tps:
                    t1_t1_in, t1_fl_in, meta1 = prepare_inputs("t1")
                    shift_meta_path = meta1 if not shift_meta_path else shift_meta_path

                # LST-AI inference (only for shifted timepoints; t0 may reuse baseline).
                mask0_path = (baseline_root / pid / "t0" / "lesion_mask.nii.gz").resolve()
                mask1_path = (baseline_root / pid / "t1" / "lesion_mask.nii.gz").resolve()

                if "t0" in shifted_tps:
                    run_dir0 = shift_lstai_root / pid / args.mode / shift_name / f"level{level}" / "t0"
                    ok0, rt0, err0 = infer_one(
                        pid=pid,
                        shift=shift_name,
                        level=int(level),
                        tp="t0",
                        t1_path=t0_t1_in,
                        flair_path=t0_fl_in,
                        run_dir=run_dir0,
                    )
                    infer_runtime += rt0
                    if not ok0:
                        warns.append(f"lstai_failed:t0:{err0}")
                    mask0_path = run_dir0 / "lesion_mask.nii.gz"

                if "t1" in shifted_tps:
                    run_dir1 = shift_lstai_root / pid / args.mode / shift_name / f"level{level}" / "t1"
                    ok1, rt1, err1 = infer_one(
                        pid=pid,
                        shift=shift_name,
                        level=int(level),
                        tp="t1",
                        t1_path=t1_t1_in,
                        flair_path=t1_fl_in,
                        run_dir=run_dir1,
                    )
                    infer_runtime += rt1
                    if not ok1:
                        warns.append(f"lstai_failed:t1:{err1}")
                    mask1_path = run_dir1 / "lesion_mask.nii.gz"

                if not (mask0_path.exists() and mask1_path.exists()):
                    row = {
                        "patient_id": pid,
                        "ok": "False",
                        "warnings": ";".join(warns + ["missing_shift_masks"]),
                        "mode": args.mode,
                        "shift": shift_name,
                        "level": str(level),
                        "shifted_timepoints": ",".join(shifted_tps),
                        "delta_lesion_vol_mm3_base": base["delta_lesion_vol_mm3"],
                        "delta_lesion_vol_mm3": "",
                        "deltaV_minus_base_mm3": "",
                        "abs_deltaV_minus_base_mm3": "",
                        "dice_chg_sym_cons_base": base["dice_chg_sym_cons"],
                        "dice_chg_sym_cons": "",
                        "dice_sym_minus_base": "",
                        "dice_new_cons_base": base["dice_new_cons"],
                        "dice_new_cons": "",
                        "dice_new_minus_base": "",
                        "lesion_vol_t0_mm3": "",
                        "lesion_vol_t1_mm3": "",
                        "change_gt_vol_mm3": base["change_gt_vol_mm3"],
                        "intensity_diff_mean": "",
                        "intensity_diff_p95": "",
                        "t0_flair_median": "",
                        "t1_flair_median": "",
                        "inference_runtime_sec": f"{infer_runtime:.3f}",
                        "shift_meta_json": shift_meta_path,
                    }
                    out_writer.writerow(row)
                    out_f.flush()
                    continue

                # Evaluate metrics on the shifted image pair actually observed by the pipeline.
                m, w2 = _compute_metrics(
                    patient_id=pid,
                    flair0_path=t0_fl_in,
                    flair1_path=t1_fl_in,
                    brainmask_path=brainmask_path,
                    gt_path=gt_path,
                    mask0_path=mask0_path,
                    mask1_path=mask1_path,
                    new_dilation_radius_mm=float(args.new_dilation_radius_mm),
                    min_component_volume_mm3=float(args.min_component_volume_mm3),
                    connectivity=int(args.connectivity),
                    intensity_p_lo=float(args.intensity_p_lo),
                    intensity_p_hi=float(args.intensity_p_hi),
                    diff_threshold=float(args.diff_threshold),
                )
                warns.extend(w2)

                dv_base = float(base["delta_lesion_vol_mm3"])
                dv_shift = float(m["delta_lesion_vol_mm3"])
                dv_diff = dv_shift - dv_base
                dice_sym_base = float(base["dice_chg_sym_cons"])
                dice_sym_shift = float(m["dice_chg_sym_cons"])
                dice_new_base = float(base["dice_new_cons"])
                dice_new_shift = float(m["dice_new_cons"])

                row = {
                    "patient_id": pid,
                    "ok": "True",
                    "warnings": ";".join(warns),
                    "mode": args.mode,
                    "shift": shift_name,
                    "level": str(level),
                    "shifted_timepoints": ",".join(shifted_tps),
                    "delta_lesion_vol_mm3_base": dv_base,
                    "delta_lesion_vol_mm3": dv_shift,
                    "deltaV_minus_base_mm3": dv_diff,
                    "abs_deltaV_minus_base_mm3": abs(dv_diff),
                    "dice_chg_sym_cons_base": dice_sym_base,
                    "dice_chg_sym_cons": dice_sym_shift,
                    "dice_sym_minus_base": dice_sym_shift - dice_sym_base,
                    "dice_new_cons_base": dice_new_base,
                    "dice_new_cons": dice_new_shift,
                    "dice_new_minus_base": dice_new_shift - dice_new_base,
                    "lesion_vol_t0_mm3": m["lesion_vol_t0_mm3"],
                    "lesion_vol_t1_mm3": m["lesion_vol_t1_mm3"],
                    "change_gt_vol_mm3": m["change_gt_vol_mm3"],
                    "intensity_diff_mean": m["intensity_diff_mean"],
                    "intensity_diff_p95": m["intensity_diff_p95"],
                    "t0_flair_median": m["t0_flair_median"],
                    "t1_flair_median": m["t1_flair_median"],
                    "inference_runtime_sec": f"{infer_runtime:.3f}",
                    "shift_meta_json": shift_meta_path,
                }
                out_writer.writerow(row)
                out_f.flush()

    out_f.close()
    runlog_f.close()

    # Summary + figures
    import pandas as pd

    df = pd.read_csv(out_csv)
    df_ok_all = df[df["ok"].astype(str) == "True"].copy()
    df_ok = df_ok_all[(df_ok_all["mode"].astype(str) == str(args.mode)) & (df_ok_all["shift"].isin(shifts))].copy()
    if df_ok.empty:
        print(f"Wrote {out_csv.relative_to(repo_root)} but no ok rows to summarize.")
        return 0

    gb = df_ok.groupby(["mode", "shift", "level"], as_index=False)
    summary = gb.agg(
        abs_deltaV_minus_base_mm3_mean=("abs_deltaV_minus_base_mm3", "mean"),
        abs_deltaV_minus_base_mm3_std=("abs_deltaV_minus_base_mm3", "std"),
        dice_sym_minus_base_mean=("dice_sym_minus_base", "mean"),
        dice_sym_minus_base_std=("dice_sym_minus_base", "std"),
        dice_new_minus_base_mean=("dice_new_minus_base", "mean"),
        dice_new_minus_base_std=("dice_new_minus_base", "std"),
        n=("patient_id", "count"),
    )
    summary.to_csv(out_summary_csv, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels_present = sorted({int(x) for x in summary["level"].tolist()})

    # ΔV sensitivity curves
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    for shift_name in shifts:
        sub = summary[(summary["shift"] == shift_name)].sort_values("level")
        if sub.empty:
            continue
        xs = sub["level"].astype(int).to_numpy()
        ys = sub["abs_deltaV_minus_base_mm3_mean"].to_numpy(dtype=float)
        yerr = sub["abs_deltaV_minus_base_mm3_std"].fillna(0.0).to_numpy(dtype=float)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=3, label=shift_name)
    ax.set_title(f"Phase 3: |ΔV_shift - ΔV_base| vs severity  (mode={args.mode})")
    ax.set_xlabel("shift severity level")
    ax.set_ylabel("|ΔΔV| (mm³)")
    ax.set_xticks(levels_present)
    ax.legend()
    fig.savefig(out_fig / "phase3_robustness_curve_deltaV.png", dpi=200)
    plt.close(fig)

    # Dice sensitivity curves (difference to baseline)
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    for shift_name in shifts:
        sub = summary[(summary["shift"] == shift_name)].sort_values("level")
        if sub.empty:
            continue
        xs = sub["level"].astype(int).to_numpy()
        ys = sub["dice_sym_minus_base_mean"].to_numpy(dtype=float)
        yerr = sub["dice_sym_minus_base_std"].fillna(0.0).to_numpy(dtype=float)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=2, capsize=3, label=shift_name)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"Phase 3: Dice_sym_cons(shift) - Dice_sym_cons(base)  (mode={args.mode})")
    ax.set_xlabel("shift severity level")
    ax.set_ylabel("ΔDice (to change GT)")
    ax.set_xticks(levels_present)
    ax.legend()
    fig.savefig(out_fig / "phase3_robustness_curve_dice.png", dpi=200)
    plt.close(fig)

    # Sensitive-case visualization: pick row with largest |ΔΔV| among shifted levels (>0).
    df_shifted = df_ok[df_ok["level"].astype(int) > 0].copy()
    if not df_shifted.empty:
        worst_row = df_shifted.sort_values("abs_deltaV_minus_base_mm3", ascending=False).iloc[0]
        pid = str(worst_row["patient_id"])
        shift_name = str(worst_row["shift"])

        # Collect per-level panels for this patient and shift.
        import numpy as np

        from mslam.metrics.longitudinal_metrics import (
            dilate_binary_mm,
            intensity_change_stats,
            remove_small_components,
        )

        # Load base reference arrays
        r = next(rr for rr in ok_rows if rr["patient_id"] == pid)
        brainmask_path = (repo_root / r["brainmask"]).resolve()
        gt_path = (repo_root / r["gt_change"]).resolve()
        _, bm_arr = read_array(brainmask_path)
        _, gt_arr = read_array(gt_path)
        bm = (bm_arr > 0)
        gt_b = (gt_arr > 0)

        def load_case(level: int):
            # Decide which inputs/masks are used in this (pid,shift,level).
            flair0_path = (repo_root / r["t0_flair"]).resolve()
            flair1_path = (repo_root / r["t1_flair"]).resolve()
            if args.mode == "both" and level > 0:
                flair0_path = (shift_input_root / pid / args.mode / shift_name / f"level{level}" / "t0" / "flair.nii.gz").resolve()
            if level > 0:
                flair1_path = (shift_input_root / pid / args.mode / shift_name / f"level{level}" / "t1" / "flair.nii.gz").resolve()

            mask0_path = (baseline_root / pid / "t0" / "lesion_mask.nii.gz").resolve()
            mask1_path = (baseline_root / pid / "t1" / "lesion_mask.nii.gz").resolve()
            if args.mode == "both" and level > 0:
                mask0_path = (shift_lstai_root / pid / args.mode / shift_name / f"level{level}" / "t0" / "lesion_mask.nii.gz").resolve()
            if level > 0:
                mask1_path = (shift_lstai_root / pid / args.mode / shift_name / f"level{level}" / "t1" / "lesion_mask.nii.gz").resolve()

            _, f0 = read_array(flair0_path)
            _, f1 = read_array(flair1_path)
            hdr = read_header(flair0_path)
            spacing = hdr.zooms
            _, m0 = read_array(mask0_path)
            _, m1 = read_array(mask1_path)
            m0b = (m0 > 0)
            m1b = (m1 > 0)

            m0_dil = dilate_binary_mm(m0b, spacing_xyz=spacing, radius_mm=float(args.new_dilation_radius_mm))
            new_cons_pre = m1b & (~m0_dil)
            min_voxels = int(math.ceil(float(args.min_component_volume_mm3) / float(hdr.voxel_volume_mm3))) if hdr.voxel_volume_mm3 > 0 else 1
            new_cons = remove_small_components(new_cons_pre, min_voxels=min_voxels, connectivity=int(args.connectivity))

            t0_s, t1_s, _ = intensity_change_stats(
                f0,
                f1,
                bm,
                p_lo=float(args.intensity_p_lo),
                p_hi=float(args.intensity_p_hi),
                diff_threshold=float(args.diff_threshold),
            )
            denom0 = (t0_s["v_hi"] - t0_s["v_lo"]) if (t0_s["v_hi"] > t0_s["v_lo"]) else 1.0
            denom1 = (t1_s["v_hi"] - t1_s["v_lo"]) if (t1_s["v_hi"] > t1_s["v_lo"]) else 1.0
            n0 = np.clip((f0.astype(np.float32, copy=False) - float(t0_s["v_lo"])) / float(denom0), 0.0, 1.0)
            n1 = np.clip((f1.astype(np.float32, copy=False) - float(t1_s["v_lo"])) / float(denom1), 0.0, 1.0)
            diff_n = np.abs(n1 - n0)
            return f0, f1, m0b, m1b, new_cons, diff_n

        levels_sorted = sorted(set(int(l) for l in levels_present))
        # Ensure level0 included for comparison.
        if 0 not in levels_sorted:
            levels_sorted = [0] + levels_sorted

        fig = plt.figure(figsize=(12, 4.6 * len(levels_sorted)), layout="constrained")
        gs = fig.add_gridspec(len(levels_sorted), 1)
        for i, lv in enumerate(levels_sorted):
            ax = fig.add_subplot(gs[i, 0])
            ax.set_axis_off()
            tmp = out_fig / f"_tmp_phase3_{pid}_{shift_name}_level{lv}.png"
            f0, f1, m0b, m1b, new_cons, diff_n = load_case(lv)
            title = f"{shift_name} level{lv}"
            _make_patient_change_figure(
                patient_id=pid,
                flair0=f0,
                flair1=f1,
                brain=bm,
                mask0=m0b,
                mask1=m1b,
                gt=gt_b,
                new_mask=new_cons,
                diff_norm=diff_n,
                out_path=tmp,
                title=title,
                scale=args.scale,
            )
            img = plt.imread(tmp)
            ax.imshow(img)
            tmp.unlink(missing_ok=True)
        fig.suptitle(f"Phase 3 sensitive case: {pid}  (mode={args.mode}, shift={shift_name})", fontsize=14)
        fig.savefig(out_fig / "phase3_sensitive_case.png", dpi=200)
        plt.close(fig)

    print(f"Wrote {out_csv.relative_to(repo_root).as_posix()}")
    print(f"Wrote {out_summary_csv.relative_to(repo_root).as_posix()}")
    print(f"Wrote {runlog_path.relative_to(repo_root).as_posix()}")
    print(f"Wrote figures -> {out_fig.relative_to(repo_root).as_posix()}/ (phase3_*.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
