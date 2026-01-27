from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _parse_patients_arg(patients: str) -> set[str]:
    return {p.strip() for p in patients.split(",") if p.strip()}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_run_complete(run_dir: Path, probability_map: bool) -> bool:
    """
    Treat a run as complete only if:
    - lstai_run.json exists and indicates returncode==0
    - lesion_mask.nii.gz exists
    - if probability_map: lesion_prob(_model1/2/3).nii.gz exist
    """
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: run LST-AI (Docker) for t0/t1 from Phase 0 manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/tables/phase0_manifest_checked.csv"),
        help="Input manifest (must include ok column).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/processed/lstai_outputs"),
        help="Output root, e.g. data/processed/lstai_outputs/",
    )
    parser.add_argument(
        "--runlog",
        type=Path,
        default=Path("results/tables/phase1_lstai_runlog.csv"),
        help="CSV runlog output.",
    )
    parser.add_argument("--image", type=str, default="jqmcginnis/lst-ai:v1.2.0")
    parser.add_argument("--platform", type=str, default="linux/amd64")
    parser.add_argument("--device", type=str, default="cpu", help='Either "cpu" or GPU id like "0"')
    parser.add_argument("--threads", type=int, default=0, help="0 => use LST-AI default (all available)")

    parser.add_argument("--patients", type=str, default="", help="Comma-separated patient ids (e.g. patient01,patient02)")
    parser.add_argument("--num-patients", type=int, default=2, help="Used when --patients is empty and --all is not set")
    parser.add_argument("--all", action="store_true", help="Run all ok patients in the manifest")
    parser.add_argument("--timepoints", choices=["t0", "t1", "both"], default="both")

    parser.add_argument("--no-probability-map", action="store_true", help="Disable --probability_map")
    parser.add_argument("--no-segment-only", action="store_true", help="Disable --segment_only (will also run annotation)")
    parser.add_argument("--stripped", action="store_true", help="Pass --stripped (only if inputs are skull-stripped)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--lesion-threshold", type=int, default=0)
    parser.add_argument("--clipping", nargs=2, type=float, default=(0.5, 99.5))
    parser.add_argument("--fast-mode", action="store_true")

    parser.add_argument("--keep-temp", action="store_true", help="Keep LST-AI temp dir (can be large)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run dirs")
    parser.add_argument("--rerun", action="store_true", help="Rerun even if lesion_mask.nii.gz already exists")
    parser.add_argument("--timeout-sec", type=float, default=0.0, help="0 => no timeout")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs, do not execute")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest.resolve()
    out_root = (repo_root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root.resolve()
    runlog_path = (repo_root / args.runlog).resolve() if not args.runlog.is_absolute() else args.runlog.resolve()
    _ensure_parent(runlog_path)

    import sys

    sys.path.insert(0, str(repo_root / "src"))
    from mslam.engines.lstai import LstAiDockerConfig, run_lstai_docker

    rows = _read_manifest(manifest_path)
    ok_rows = [r for r in rows if r.get("ok", "False") == "True"]

    selected: list[dict[str, str]]
    if args.patients.strip():
        wanted = _parse_patients_arg(args.patients)
        selected = [r for r in ok_rows if r.get("patient_id") in wanted]
    elif args.all:
        selected = ok_rows
    else:
        if args.num_patients <= 0:
            selected = []
        else:
            selected = ok_rows[: args.num_patients]

    if not selected:
        print("No patients selected. Use --patients, --all, or adjust --num-patients.")
        return 1

    tps = ["t0", "t1"] if args.timepoints == "both" else [args.timepoints]

    cfg = LstAiDockerConfig(
        image=args.image,
        platform=args.platform,
        device=args.device,
        threads=(None if args.threads == 0 else args.threads),
        segment_only=not args.no_segment_only,
        stripped=args.stripped,
        threshold=args.threshold,
        lesion_threshold=args.lesion_threshold,
        clipping=(float(args.clipping[0]), float(args.clipping[1])),
        fast_mode=args.fast_mode,
        probability_map=not args.no_probability_map,
    )

    fieldnames = [
        "timestamp",
        "patient_id",
        "timepoint",
        "ok",
        "returncode",
        "runtime_sec",
        "docker_image",
        "docker_platform",
        "device",
        "threads",
        "segment_only",
        "stripped",
        "fast_mode",
        "probability_map",
        "threshold",
        "lesion_threshold",
        "clipping_min",
        "clipping_max",
        "t1_path",
        "flair_path",
        "run_dir",
        "lesion_mask",
        "lesion_prob",
        "lesion_prob_model1",
        "lesion_prob_model2",
        "lesion_prob_model3",
        "stdout_log",
        "stderr_log",
        "metadata_json",
        "cmd_json",
        "error",
    ]

    write_header = not runlog_path.exists()
    with runlog_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for r in selected:
            patient_id = r["patient_id"]
            for tp in tps:
                t1_key = "t0_t1w" if tp == "t0" else "t1_t1w"
                flair_key = "t0_flair" if tp == "t0" else "t1_flair"
                t1_path = (repo_root / r[t1_key]).resolve()
                flair_path = (repo_root / r[flair_key]).resolve()
                run_dir = out_root / patient_id / tp

                if _is_run_complete(run_dir, cfg.probability_map) and not (args.rerun or args.overwrite):
                    print(f"Skip complete: {patient_id} {tp} ({run_dir.relative_to(repo_root)})")
                    continue

                if args.dry_run:
                    print(f"[dry-run] {patient_id} {tp} -> {run_dir}")
                    continue

                started = time.time()
                row_out = {k: "" for k in fieldnames}
                row_out.update(
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "patient_id": patient_id,
                        "timepoint": tp,
                        "docker_image": cfg.image,
                        "docker_platform": cfg.platform or "",
                        "device": cfg.device,
                        "threads": "" if cfg.threads is None else str(cfg.threads),
                        "segment_only": str(cfg.segment_only),
                        "stripped": str(cfg.stripped),
                        "fast_mode": str(cfg.fast_mode),
                        "probability_map": str(cfg.probability_map),
                        "threshold": str(cfg.threshold),
                        "lesion_threshold": str(cfg.lesion_threshold),
                        "clipping_min": str(cfg.clipping[0]),
                        "clipping_max": str(cfg.clipping[1]),
                        "t1_path": str(t1_path),
                        "flair_path": str(flair_path),
                        "run_dir": str(run_dir),
                    }
                )

                try:
                    result = run_lstai_docker(
                        t1_path=t1_path,
                        flair_path=flair_path,
                        run_dir=run_dir,
                        cfg=cfg,
                        keep_temp=args.keep_temp,
                        overwrite=args.overwrite,
                        timeout_sec=(None if args.timeout_sec == 0 else args.timeout_sec),
                    )
                    row_out["ok"] = str(result.returncode == 0)
                    row_out["returncode"] = str(result.returncode)
                    row_out["runtime_sec"] = f"{result.runtime_sec:.3f}"
                    row_out["lesion_mask"] = str(result.lesion_mask)
                    row_out["lesion_prob"] = "" if result.lesion_prob is None else str(result.lesion_prob)
                    row_out["lesion_prob_model1"] = "" if result.lesion_prob_model1 is None else str(result.lesion_prob_model1)
                    row_out["lesion_prob_model2"] = "" if result.lesion_prob_model2 is None else str(result.lesion_prob_model2)
                    row_out["lesion_prob_model3"] = "" if result.lesion_prob_model3 is None else str(result.lesion_prob_model3)
                    row_out["stdout_log"] = str(result.stdout_log)
                    row_out["stderr_log"] = str(result.stderr_log)
                    row_out["metadata_json"] = str(result.metadata_json)
                    row_out["cmd_json"] = json.dumps(result.cmd)
                    print(f"Done: {patient_id} {tp} (rc={result.returncode}, {result.runtime_sec:.1f}s)")
                except Exception as e:
                    runtime = time.time() - started
                    row_out["ok"] = "False"
                    row_out["returncode"] = ""
                    row_out["runtime_sec"] = f"{runtime:.3f}"
                    row_out["error"] = f"{type(e).__name__}: {e}"
                    print(f"FAILED: {patient_id} {tp}: {row_out['error']}")
                    if args.fail_fast:
                        writer.writerow(row_out)
                        return 1
                writer.writerow(row_out)
                f.flush()

    print(f"Wrote runlog -> {runlog_path.relative_to(repo_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
