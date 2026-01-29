from __future__ import annotations

import argparse
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 5: export patient-level features (Phase 2 + Phase 4 [+ optional Phase 3 sensitivity]).")
    ap.add_argument(
        "--phase2-csv",
        type=str,
        default="results/tables/phase2_longitudinal_metrics.csv",
        help="Phase 2 patient-level metrics table.",
    )
    ap.add_argument(
        "--phase4-csv",
        type=str,
        default="results/tables/phase4_uncertainty_metrics.csv",
        help="Phase 4 uncertainty/QC table.",
    )
    ap.add_argument(
        "--phase3-sensitivity-csv",
        type=str,
        default="auto",
        help='Optional patient-level sensitivity table (default: "auto" uses results/tables/phase4_uncertainty_vs_shift_sensitivity.csv if present). Use "" to disable.',
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="results/tables/features_v1.csv",
        help="Output feature table (one row per patient).",
    )
    args = ap.parse_args()

    repo = _repo_root()
    p2 = (repo / args.phase2_csv).resolve()
    p4 = (repo / args.phase4_csv).resolve()
    out_csv = (repo / args.out_csv).resolve()

    import numpy as np
    import pandas as pd

    df2 = pd.read_csv(p2)
    df4 = pd.read_csv(p4)

    df2 = df2[df2["ok"].astype(str) == "True"].copy()
    df4 = df4[df4["ok"].astype(str) == "True"].copy()
    if df2.empty or df4.empty:
        raise SystemExit("No ok rows in Phase 2 and/or Phase 4 tables.")

    merged = pd.merge(
        df2,
        df4,
        on="patient_id",
        how="inner",
        suffixes=("_phase2", "_phase4"),
    )
    if merged.empty:
        raise SystemExit("Phase 2 and Phase 4 tables had no overlapping patient_id rows.")

    # Standardize QC flags to bool.
    for c in ["qc_needs_review", "qc_change_not_confident"]:
        if c in merged.columns:
            merged[c] = merged[c].map(_to_bool)

    # Derived metrics
    merged["abs_delta_lesion_vol_mm3"] = merged["delta_lesion_vol_mm3"].astype(float).abs()

    # log1p transforms (heavy-tailed, non-negative)
    log1p_cols = [
        "lesion_vol_t0_mm3",
        "lesion_vol_t1_mm3",
        "abs_delta_lesion_vol_mm3",
        "new_cons_vol_mm3",
        "chg_sym_cons_vol_mm3",
    ]
    for c in log1p_cols:
        if c in merged.columns:
            merged[f"log1p_{c}"] = np.log1p(np.maximum(merged[c].astype(float), 0.0))

    # Optional: Phase 3 sensitivity (typically produced by script 11; may be subset-only).
    sens_path: Path | None
    if str(args.phase3_sensitivity_csv).strip() == "":
        sens_path = None
    elif str(args.phase3_sensitivity_csv).strip().lower() == "auto":
        cand = repo / "results/tables/phase4_uncertainty_vs_shift_sensitivity.csv"
        sens_path = cand.resolve() if cand.exists() else None
    else:
        cand = repo / args.phase3_sensitivity_csv
        sens_path = cand.resolve() if cand.exists() else None

    if sens_path is not None:
        sens = pd.read_csv(sens_path)
        if "patient_id" in sens.columns:
            keep = {
                "patient_id",
                "n_phase3",
                "deltav_abs_mean",
                "deltav_abs_p95",
                "deltav_abs_max",
                "dice_sym_drop_p05",
                "dice_sym_drop_max",
            }
            sens = sens[[c for c in sens.columns if c in keep]].copy()
            rename = {c: f"phase3_{c}" for c in sens.columns if c != "patient_id"}
            sens = sens.rename(columns=rename)
            merged = pd.merge(merged, sens, on="patient_id", how="left")

            for c in ["phase3_deltav_abs_mean", "phase3_deltav_abs_p95", "phase3_deltav_abs_max", "phase3_dice_sym_drop_p05", "phase3_dice_sym_drop_max"]:
                if c in merged.columns:
                    merged[f"log1p_{c}"] = np.log1p(np.maximum(merged[c].astype(float), 0.0))

    # Construct a compact, auditable feature table.
    # We keep both "monitoring" features and "eval" features (to change-GT) with explicit names.
    cols_order: list[str] = []

    def add(cols: list[str]) -> None:
        for c in cols:
            if c in merged.columns and c not in cols_order:
                cols_order.append(c)

    add(["patient_id"])

    # QC/meta
    add(["qc_needs_review", "qc_change_not_confident"])

    # Monitoring features (model-based + segmentation-independent evidence)
    add(
        [
            "lesion_vol_t0_mm3",
            "lesion_vol_t1_mm3",
            "delta_lesion_vol_mm3",
            "abs_delta_lesion_vol_mm3",
            "new_cons_vol_mm3",
            "chg_sym_cons_vol_mm3",
            "intensity_diff_mean",
            "intensity_diff_p95",
            "t1_over_t0_median_ratio",
            "unc_ens_var_mean_lesion_t1",
            "unc_ens_var_p95_brain_t1",
        ]
    )

    # Transformed versions (recommended for clustering)
    add([f"log1p_{c}" for c in log1p_cols])

    # Optional robustness sensitivity (if available)
    add(
        [
            "phase3_n_phase3",
            "phase3_deltav_abs_p95",
            "phase3_dice_sym_drop_p05",
            "log1p_phase3_deltav_abs_p95",
            "log1p_phase3_dice_sym_drop_p05",
        ]
    )

    # Eval features (explicitly to change-GT; dataset-dependent)
    eval_cols = [
        "change_gt_vol_mm3",
        "dice_new_cons",
        "dice_chg_sym_cons",
        "chg_sym_cons_vol_err_mm3",
        "gt_covered_by_t1_frac",
        "gt_outside_brain_frac",
    ]
    add(eval_cols)

    # Keep warnings for audit (not for clustering).
    add(["warnings_phase2", "warnings_phase4", "warnings"])

    out = merged[[c for c in cols_order if c in merged.columns]].copy()

    _ensure_parent(out_csv)
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv.relative_to(repo)}  (rows={len(out)}, cols={len(out.columns)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
