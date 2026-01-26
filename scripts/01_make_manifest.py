from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _relpath(repo_root: Path, p: Path) -> str:
    return p.relative_to(repo_root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0: build manifest.csv for open_ms_data longitudinal/coregistered")
    parser.add_argument("--data-root", type=Path, required=True, help="e.g. data/raw/open_ms_data/longitudinal/coregistered")
    parser.add_argument("--out", type=Path, required=True, help="e.g. results/tables/phase0_manifest.csv")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()

    patients = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("patient")])
    out_path = (repo_root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "patient_id",
        "brainmask",
        "gt_change",
        "t0_t1w",
        "t0_t2w",
        "t0_flair",
        "t1_t1w",
        "t1_t2w",
        "t1_flair",
        "t0_flair_to_common_txt",
        "t1_flair_to_common_txt",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in patients:
            row = {
                "patient_id": p.name,
                "brainmask": _relpath(repo_root, p / "brainmask.nii.gz"),
                "gt_change": _relpath(repo_root, p / "gt.nii.gz"),
                "t0_t1w": _relpath(repo_root, p / "study1_T1W.nii.gz"),
                "t0_t2w": _relpath(repo_root, p / "study1_T2W.nii.gz"),
                "t0_flair": _relpath(repo_root, p / "study1_FLAIR.nii.gz"),
                "t1_t1w": _relpath(repo_root, p / "study2_T1W.nii.gz"),
                "t1_t2w": _relpath(repo_root, p / "study2_T2W.nii.gz"),
                "t1_flair": _relpath(repo_root, p / "study2_FLAIR.nii.gz"),
                "t0_flair_to_common_txt": _relpath(repo_root, p / "study1_FLAIR_to_common_space.txt"),
                "t1_flair_to_common_txt": _relpath(repo_root, p / "study2_FLAIR_to_common_space.txt"),
            }
            writer.writerow(row)

    print(f"Wrote manifest for {len(patients)} patients -> {out_path.relative_to(repo_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

