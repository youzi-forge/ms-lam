from __future__ import annotations

import argparse
import os
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _spearmanr(x, y) -> float | None:
    try:
        from scipy.stats import spearmanr
    except Exception:
        return None
    try:
        r, _ = spearmanr(x, y)
        return float(r)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 4 add-on: relate Phase 4 uncertainty to Phase 3 shift sensitivity (patient-level merge)."
    )
    ap.add_argument(
        "--phase3-csv",
        type=str,
        default="results/tables/phase3_robustness.csv",
        help="Phase 3 robustness long table (patient×shift×level).",
    )
    ap.add_argument(
        "--phase4-csv",
        type=str,
        default="results/tables/phase4_uncertainty_metrics.csv",
        help="Phase 4 patient-level uncertainty table.",
    )
    ap.add_argument("--mode", type=str, default="t1_only", choices=["t1_only", "both"])
    ap.add_argument(
        "--shifts",
        type=str,
        default="gamma,noise,resolution",
        help="Comma-separated shifts to include from Phase 3.",
    )
    ap.add_argument(
        "--levels",
        type=str,
        default="1,2",
        help="Comma-separated severity levels to summarize (exclude baseline 0 by default).",
    )
    ap.add_argument(
        "--out-csv",
        type=str,
        default="results/tables/phase4_uncertainty_vs_shift_sensitivity.csv",
        help="Output merged patient-level table.",
    )
    ap.add_argument(
        "--out-fig-deltav",
        type=str,
        default="results/figures/phase4_unc_vs_shift_sens_deltaV.png",
        help="Output scatter figure (uncertainty vs |ΔΔV| sensitivity).",
    )
    ap.add_argument(
        "--out-fig-dice",
        type=str,
        default="results/figures/phase4_unc_vs_shift_sens_dice.png",
        help="Output scatter figure (uncertainty vs Dice-drop sensitivity).",
    )
    ap.add_argument(
        "--annotate",
        type=str,
        default="all",
        choices=["none", "qc", "extremes", "all"],
        help="Which points to annotate with patient_id in the scatter plots.",
    )
    ap.add_argument(
        "--primary-unc",
        type=str,
        default="ens_var",
        choices=["ens_var", "prob_entropy"],
        help="Which uncertainty family to use for x-axis plots.",
    )
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    repo = _repo_root()
    p3 = (repo / args.phase3_csv).resolve()
    p4 = (repo / args.phase4_csv).resolve()
    out_csv = (repo / args.out_csv).resolve()
    out_fig_dv = (repo / args.out_fig_deltav).resolve()
    out_fig_dc = (repo / args.out_fig_dice).resolve()

    import numpy as np
    import pandas as pd

    df3 = pd.read_csv(p3)
    df4 = pd.read_csv(p4)

    # Phase 3 filtering
    shifts = set(_parse_csv_list(args.shifts))
    levels = set(_parse_int_list(args.levels))
    df3 = df3[df3["ok"].astype(str) == "True"].copy()
    df3 = df3[(df3["mode"].astype(str) == str(args.mode)) & (df3["shift"].isin(sorted(shifts)))].copy()
    df3["level"] = df3["level"].astype(int)
    df3 = df3[df3["level"].isin(sorted(levels))].copy()
    if df3.empty:
        raise SystemExit(f"No Phase 3 rows after filtering (mode={args.mode}, shifts={sorted(shifts)}, levels={sorted(levels)}).")

    # Patient-level sensitivity summaries from Phase 3 (across shift×level).
    def q(p: float):
        def f(x):
            return x.quantile(p)

        f.__name__ = f"q{int(round(100 * p))}"
        return f

    gb = df3.groupby("patient_id", as_index=False)
    sens = gb.agg(
        n_phase3=("shift", "count"),
        deltav_abs_mean=("abs_deltaV_minus_base_mm3", "mean"),
        deltav_abs_p95=("abs_deltaV_minus_base_mm3", q(0.95)),
        deltav_abs_max=("abs_deltaV_minus_base_mm3", "max"),
        dice_sym_minus_base_mean=("dice_sym_minus_base", "mean"),
        dice_sym_minus_base_p05=("dice_sym_minus_base", q(0.05)),
        dice_sym_minus_base_min=("dice_sym_minus_base", "min"),
    )
    # Convert to "drop magnitude" (positive = worse) for ease of plotting.
    sens["dice_sym_drop_p05"] = -sens["dice_sym_minus_base_p05"].astype(float)
    sens["dice_sym_drop_max"] = -sens["dice_sym_minus_base_min"].astype(float)

    # Phase 4 filtering (keep ok rows)
    df4 = df4[df4["ok"].astype(str) == "True"].copy()
    if df4.empty:
        raise SystemExit("No ok Phase 4 rows.")

    merged = pd.merge(df4, sens, on="patient_id", how="inner")
    merged = merged.sort_values("patient_id").reset_index(drop=True)

    _ensure_parent(out_csv)
    merged.to_csv(out_csv, index=False)

    # Plot settings
    # Make matplotlib cache local+writable (avoids slow font-cache rebuilds on some systems).
    mpl_cache = repo / "results" / ".mplconfig"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unc_prefix = "unc_ens_var" if args.primary_unc == "ens_var" else "unc_prob_entropy"
    x_cols = [
        f"{unc_prefix}_mean_lesion_t1",
        f"{unc_prefix}_p95_brain_t1",
    ]
    for c in x_cols:
        if c not in merged.columns:
            raise SystemExit(f"Missing uncertainty column in merged table: {c}")

    qc_col = "qc_needs_review" if "qc_needs_review" in merged.columns else None
    is_qc = merged[qc_col].map(_to_bool) if qc_col else np.zeros(len(merged), dtype=bool)

    def annotate_mask(xvals, yvals) -> np.ndarray:
        if args.annotate == "none":
            return np.zeros(len(merged), dtype=bool)
        if args.annotate == "all":
            return np.ones(len(merged), dtype=bool)
        if args.annotate == "qc":
            return np.asarray(is_qc, dtype=bool)
        # extremes: top-3 x or top-3 y
        k = min(3, len(merged))
        top_x = set(np.argsort(xvals)[-k:].tolist())
        top_y = set(np.argsort(yvals)[-k:].tolist())
        mask = np.zeros(len(merged), dtype=bool)
        for i in top_x.union(top_y):
            mask[int(i)] = True
        return mask

    def scatter_two_panel(*, y_col: str, y_label: str, out_path: Path, title: str) -> None:
        fig = plt.figure(figsize=(9, 4.2), layout="constrained")
        axs = fig.subplots(1, 2, sharey=True)
        for ax, x_col in zip(axs, x_cols, strict=True):
            x = merged[x_col].astype(float).to_numpy()
            y = merged[y_col].astype(float).to_numpy()
            colors = np.where(is_qc, "#d97706", "#1f77b4")  # orange vs blue
            ax.scatter(x, y, s=45, c=colors, alpha=0.90, edgecolors="none")

            r = _spearmanr(x, y)
            if r is not None and np.isfinite(r):
                ax.set_title(f"{x_col}  (Spearman ρ={r:.2f})", fontsize=10)
            else:
                ax.set_title(x_col, fontsize=10)

            ax.set_xlabel(x_col)
            mask = annotate_mask(x, y)
            offsets = [(4, 2), (4, -8), (-44, 2), (-44, -8)]
            for j, i in enumerate(np.where(mask)[0]):
                pid = str(merged.loc[int(i), "patient_id"])
                dx, dy = offsets[j % len(offsets)]
                ax.annotate(
                    pid,
                    (x[int(i)], y[int(i)]),
                    fontsize=8,
                    xytext=(dx, dy),
                    textcoords="offset points",
                )

        axs[0].set_ylabel(y_label)
        fig.suptitle(title, fontsize=12)
        _ensure_parent(out_path)
        fig.savefig(out_path, dpi=int(args.dpi))
        plt.close(fig)

    scatter_two_panel(
        y_col="deltav_abs_p95",
        y_label="shift sensitivity: p95(|ΔV_shift − ΔV_base|) [mm³]",
        out_path=out_fig_dv,
        title=f"Uncertainty vs shift sensitivity (ΔV)  (Phase3 mode={args.mode}, shifts={','.join(sorted(shifts))}, levels={','.join(str(x) for x in sorted(levels))})",
    )

    scatter_two_panel(
        y_col="dice_sym_drop_p05",
        y_label="shift sensitivity: p05 Dice drop (−ΔDice_sym_cons) [higher=worse]",
        out_path=out_fig_dc,
        title=f"Uncertainty vs shift sensitivity (Dice)  (Phase3 mode={args.mode}, shifts={','.join(sorted(shifts))}, levels={','.join(str(x) for x in sorted(levels))})",
    )

    print(f"Wrote merged table: {out_csv.relative_to(repo)}")
    print(f"Wrote figure: {out_fig_dv.relative_to(repo)}")
    print(f"Wrote figure: {out_fig_dc.relative_to(repo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
