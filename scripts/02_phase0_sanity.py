from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 0: sanity checks for longitudinal/coregistered manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    out_report = (repo_root / args.out_report).resolve() if not args.out_report.is_absolute() else args.out_report.resolve()
    out_manifest = (repo_root / args.out_manifest).resolve() if not args.out_manifest.is_absolute() else args.out_manifest.resolve()
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    # local import (no need to install package)
    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.preprocessing.sanity_checks import check_patient_from_paths

    rows = _read_manifest(manifest_path)

    report_fieldnames = [
        "patient_id",
        "ok",
        "missing_files",
        "shape_ok",
        "affine_ok",
        "zooms_ok",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "gt_voxels",
        "gt_volume_mm3",
        "notes",
    ]

    checked_fieldnames = list(rows[0].keys()) + ["ok"]

    n_ok = 0
    with out_report.open("w", newline="") as fr, out_manifest.open("w", newline="") as fm:
        report_writer = csv.DictWriter(fr, fieldnames=report_fieldnames)
        report_writer.writeheader()
        manifest_writer = csv.DictWriter(fm, fieldnames=checked_fieldnames)
        manifest_writer.writeheader()

        for r in rows:
            patient_id = r["patient_id"]
            abs_paths = {k: (repo_root / Path(v)).resolve() if v else "" for k, v in r.items() if k != "patient_id"}
            sanity = check_patient_from_paths(patient_id=patient_id, paths=abs_paths)
            report_writer.writerow(sanity.to_row())
            r2 = dict(r)
            r2["ok"] = str(sanity.ok)
            manifest_writer.writerow(r2)
            n_ok += int(sanity.ok)

    print(f"Wrote sanity report -> {out_report.relative_to(repo_root).as_posix()}")
    print(f"Wrote checked manifest -> {out_manifest.relative_to(repo_root).as_posix()} (ok={n_ok}/{len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

