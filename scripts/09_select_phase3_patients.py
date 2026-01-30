from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a representative Phase 3 subset from Phase 2 metrics (deterministic, rule-based)."
    )
    parser.add_argument(
        "--phase2-csv",
        type=Path,
        default=Path("results/tables/phase2_longitudinal_metrics.csv"),
        help="Phase 2 aggregate metrics CSV.",
    )
    parser.add_argument("--gt-col", type=str, default="change_gt_vol_mm3", help="Column used as GT change magnitude.")
    parser.add_argument(
        "--dice-col",
        type=str,
        default="dice_chg_sym_cons",
        help="Column used as baseline segmentation/change proxy quality.",
    )
    parser.add_argument("--k-gt-top", type=int, default=2, help="How many patients to take from the top of GT magnitude.")
    parser.add_argument("--k-gt-bottom", type=int, default=2, help="How many patients to take from the bottom of GT magnitude.")
    parser.add_argument("--k-dice-top", type=int, default=2, help="How many patients to take from the top of dice.")
    parser.add_argument("--k-dice-bottom", type=int, default=2, help="How many patients to take from the bottom of dice.")
    parser.add_argument("--target-n", type=int, default=8, help="Target number of patients after de-duplication.")
    parser.add_argument("--min-n", type=int, default=6, help="If fewer after de-duplication, fill with median-ish cases.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/tables/phase3_selected_patients.txt"),
        help="Write selected patient ids (one per line).",
    )
    args = parser.parse_args()

    import pandas as pd

    csv_path = args.phase2_csv
    if not csv_path.exists():
        raise SystemExit(f"Missing Phase 2 CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ["patient_id", args.gt_col, args.dice_col]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column '{col}' in {csv_path}")

    df = df.copy()
    df[args.gt_col] = pd.to_numeric(df[args.gt_col], errors="coerce")
    df[args.dice_col] = pd.to_numeric(df[args.dice_col], errors="coerce")
    df = df.dropna(subset=[args.gt_col, args.dice_col, "patient_id"])
    df = df.sort_values("patient_id")

    gt_top = df.nlargest(int(args.k_gt_top), args.gt_col)["patient_id"].tolist()
    gt_bottom = df.nsmallest(int(args.k_gt_bottom), args.gt_col)["patient_id"].tolist()
    dice_top = df.nlargest(int(args.k_dice_top), args.dice_col)["patient_id"].tolist()
    dice_bottom = df.nsmallest(int(args.k_dice_bottom), args.dice_col)["patient_id"].tolist()

    # Priority order to avoid cherry-picking accusations:
    # 1) worst dice (stress the pipeline)
    # 2) highest GT change (large-change regime)
    # 3) lowest GT change (near-zero regime; checks false positives)
    # 4) best dice (best-case regime)
    ordered: list[str] = []
    for group in [dice_bottom, gt_top, gt_bottom, dice_top]:
        for pid in group:
            if pid not in ordered:
                ordered.append(pid)

    # If too many, keep first target_n by the priority order above.
    if int(args.target_n) > 0:
        ordered = ordered[: int(args.target_n)]

    # If too few, fill with median-ish cases (GT and dice near median), deterministic.
    if len(ordered) < int(args.min_n):
        gt_med = float(df[args.gt_col].median())
        dice_med = float(df[args.dice_col].median())
        mid_gt = (
            df.assign(_d=(df[args.gt_col] - gt_med).abs())
            .sort_values(["_d", "patient_id"])
            .head(2)["patient_id"]
            .tolist()
        )
        mid_dice = (
            df.assign(_d=(df[args.dice_col] - dice_med).abs())
            .sort_values(["_d", "patient_id"])
            .head(2)["patient_id"]
            .tolist()
        )
        for pid in mid_gt + mid_dice:
            if pid not in ordered:
                ordered.append(pid)

    if int(args.target_n) > 0:
        ordered = ordered[: int(args.target_n)]

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ordered) + ("\n" if ordered else ""))

    print("Selection rule:")
    print(f"- GT magnitude: '{args.gt_col}' (top {args.k_gt_top}, bottom {args.k_gt_bottom})")
    print(f"- Baseline quality: '{args.dice_col}' (top {args.k_dice_top}, bottom {args.k_dice_bottom})")
    print(f"- Priority: worst dice -> highest GT -> lowest GT -> best dice; de-dup then fill with median-ish if needed")
    print("")
    print(f"Wrote: {out_path.as_posix()}")
    print("Patients (comma-separated):")
    print(",".join(ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

