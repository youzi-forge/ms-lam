from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_nifti(path: Path, data, affine) -> None:
    import nibabel as nib

    arr = np.asarray(data)
    img = nib.Nifti1Image(arr.astype(np.float32, copy=False), affine)
    # Preserve voxel sizes (derived from affine diagonal if present).
    try:
        zooms = tuple(float(abs(affine[i, i])) for i in range(3))
        if all(z > 0 for z in zooms):
            img.header.set_zooms(zooms)
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))


def _make_sphere(shape: tuple[int, int, int], center_xyz: tuple[float, float, float], radius: float) -> np.ndarray:
    x = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    z = np.arange(shape[2])[None, None, :]
    cx, cy, cz = center_xyz
    d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return (d2 <= radius**2)


def _generate_toy_patient(
    *,
    patient_dir: Path,
    lstai_dir: Path,
    rng_seed: int,
    shape: tuple[int, int, int] = (32, 32, 16),
) -> None:
    rng = np.random.default_rng(int(rng_seed))
    affine = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float64)

    brain = _make_sphere(shape, center_xyz=(shape[0] / 2, shape[1] / 2, shape[2] / 2), radius=min(shape) * 0.42)

    # Base intensities (within brain).
    base0 = 80.0 + 8.0 * rng.standard_normal(shape)
    base1 = 82.0 + 8.0 * rng.standard_normal(shape)
    flair0 = np.where(brain, base0, 0.0).astype(np.float32)
    flair1 = np.where(brain, base1, 0.0).astype(np.float32)
    t1w0 = np.where(brain, 110.0 + 6.0 * rng.standard_normal(shape), 0.0).astype(np.float32)
    t1w1 = np.where(brain, 112.0 + 6.0 * rng.standard_normal(shape), 0.0).astype(np.float32)
    t2w0 = np.where(brain, 90.0 + 6.0 * rng.standard_normal(shape), 0.0).astype(np.float32)
    t2w1 = np.where(brain, 92.0 + 6.0 * rng.standard_normal(shape), 0.0).astype(np.float32)

    # Lesions (toy). We vary change patterns by patient_id to exercise edge-cases.
    pid = patient_dir.name
    if pid.endswith("01"):
        lesion0 = _make_sphere(shape, (12, 18, 8), 2.2)
        lesion1 = lesion0 | _make_sphere(shape, (16, 16, 8), 2.8)  # growth/new
    elif pid.endswith("02"):
        lesion0 = _make_sphere(shape, (18, 14, 8), 2.5)
        lesion1 = lesion0.copy()  # stable
    else:
        lesion0 = _make_sphere(shape, (14, 14, 7), 2.8)
        lesion1 = lesion0 & (~_make_sphere(shape, (14, 14, 7), 2.2))  # partial regression

    # Restrict lesions to brain and inject into FLAIR (hyperintense).
    lesion0 = lesion0 & brain
    lesion1 = lesion1 & brain
    flair0 = flair0 + lesion0.astype(np.float32) * 60.0
    flair1 = flair1 + lesion1.astype(np.float32) * 60.0

    # Change-region GT: symmetric difference in lesion regions.
    gt_change = np.logical_xor(lesion0, lesion1) & brain

    # Write dataset files (open_ms_data-like naming).
    _ensure_dir(patient_dir)
    _write_nifti(patient_dir / "brainmask.nii.gz", brain.astype(np.uint8), affine)
    _write_nifti(patient_dir / "gt.nii.gz", gt_change.astype(np.uint8), affine)

    _write_nifti(patient_dir / "study1_T1W.nii.gz", t1w0, affine)
    _write_nifti(patient_dir / "study1_T2W.nii.gz", t2w0, affine)
    _write_nifti(patient_dir / "study1_FLAIR.nii.gz", flair0, affine)

    _write_nifti(patient_dir / "study2_T1W.nii.gz", t1w1, affine)
    _write_nifti(patient_dir / "study2_T2W.nii.gz", t2w1, affine)
    _write_nifti(patient_dir / "study2_FLAIR.nii.gz", flair1, affine)

    _write_text(patient_dir / "study1_FLAIR_to_common_space.txt", "toy_transform\n")
    _write_text(patient_dir / "study2_FLAIR_to_common_space.txt", "toy_transform\n")

    # Fake "LST-AI outputs": use lesion masks with mild corruption, plus probmaps.
    def make_pred(mask: np.ndarray, p_keep: float, p_flip: float) -> np.ndarray:
        m = mask.astype(bool)
        keep = rng.random(shape) < float(p_keep)
        flip = rng.random(shape) < float(p_flip)
        out = (m & keep) | (~m & flip)
        return out & brain

    pred0 = make_pred(lesion0, p_keep=0.98, p_flip=0.002)
    pred1 = make_pred(lesion1, p_keep=0.98, p_flip=0.002)

    def make_prob(mask: np.ndarray, noise: float) -> np.ndarray:
        base = np.where(mask, 0.90, 0.05).astype(np.float32)
        base = base + noise * rng.standard_normal(shape).astype(np.float32)
        base = np.clip(base, 0.0, 1.0)
        base[~brain] = 0.0
        return base

    def write_lstai(tp: str, pred: np.ndarray) -> None:
        out_tp = lstai_dir / pid / tp
        _ensure_dir(out_tp)
        _write_nifti(out_tp / "lesion_mask.nii.gz", pred.astype(np.uint8), affine)

        p1 = make_prob(pred, noise=0.03)
        p2 = make_prob(pred, noise=0.04)
        p3 = make_prob(pred, noise=0.05)
        ens = (p1 + p2 + p3) / 3.0

        _write_nifti(out_tp / "lesion_prob_model1.nii.gz", p1, affine)
        _write_nifti(out_tp / "lesion_prob_model2.nii.gz", p2, affine)
        _write_nifti(out_tp / "lesion_prob_model3.nii.gz", p3, affine)
        _write_nifti(out_tp / "lesion_prob.nii.gz", ens, affine)

    write_lstai("t0", pred0)
    write_lstai("t1", pred1)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Toy end-to-end smoke test (no Docker, no external data): "
            "generates a tiny synthetic cohort + fake LST-AI outputs, then runs Phase 0/2/4/5."
        )
    )
    ap.add_argument("--out-root", type=str, default="results/_toy", help="Output root under results/ (tables/figures/reports).")
    ap.add_argument("--toy-data-root", type=str, default="data/processed/_toy_open_ms_data/longitudinal/coregistered")
    ap.add_argument("--toy-lstai-root", type=str, default="data/processed/_toy_lstai_outputs")
    ap.add_argument("--patients", type=str, default="patient01,patient02,patient03")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing toy outputs.")
    ap.add_argument("--phenotyping-seeds", type=int, default=10, help="Number of seeds for Phase 5 stability (smaller=faster).")
    args = ap.parse_args()

    repo = _repo_root()
    out_root = (repo / args.out_root).resolve()
    data_root = (repo / args.toy_data_root).resolve()
    lstai_root = (repo / args.toy_lstai_root).resolve()

    if args.overwrite:
        for p in [out_root, data_root, lstai_root]:
            if p.exists():
                shutil.rmtree(p)

    # 1) Generate toy cohort
    patient_ids = [p.strip() for p in str(args.patients).split(",") if p.strip()]
    if not patient_ids:
        raise SystemExit("No patients provided.")
    for i, pid in enumerate(patient_ids):
        _generate_toy_patient(
            patient_dir=data_root / pid,
            lstai_dir=lstai_root,
            rng_seed=int(args.seed) + i,
        )

    tables = out_root / "tables"
    figs = out_root / "figures"
    reports = out_root / "reports"
    _ensure_dir(tables)
    _ensure_dir(figs)
    _ensure_dir(reports)

    # 2) Phase 0: manifest + sanity checks
    manifest = tables / "phase0_manifest.csv"
    sanity = tables / "phase0_sanity_report.csv"
    checked = tables / "phase0_manifest_checked.csv"
    _run(
        [
            sys.executable,
            "scripts/01_make_manifest.py",
            "--data-root",
            str(data_root.relative_to(repo)),
            "--out",
            str(manifest.relative_to(repo)),
        ],
        cwd=repo,
    )
    _run(
        [
            sys.executable,
            "scripts/02_phase0_sanity.py",
            "--manifest",
            str(manifest.relative_to(repo)),
            "--out-report",
            str(sanity.relative_to(repo)),
            "--out-manifest",
            str(checked.relative_to(repo)),
        ],
        cwd=repo,
    )

    # 3) Phase 2: monitoring metrics + change-GT validation
    phase2_csv = tables / "phase2_longitudinal_metrics.csv"
    phase2_reports = reports / "phase2"
    _run(
        [
            sys.executable,
            "scripts/07_eval_longitudinal.py",
            "--manifest",
            str(checked.relative_to(repo)),
            "--lstai-root",
            str(lstai_root.relative_to(repo)),
            "--out-csv",
            str(phase2_csv.relative_to(repo)),
            "--out-reports-dir",
            str(phase2_reports.relative_to(repo)),
            "--out-fig-dir",
            str(figs.relative_to(repo)),
            "--example-patients",
            patient_ids[0],
        ],
        cwd=repo,
    )

    # 4) Phase 4: uncertainty + QC flags
    phase4_csv = tables / "phase4_uncertainty_metrics.csv"
    phase4_thr = tables / "phase4_qc_thresholds.json"
    phase4_reports = reports / "phase4"
    _run(
        [
            sys.executable,
            "scripts/10_phase4_uncertainty_qc.py",
            "--manifest",
            str(checked.relative_to(repo)),
            "--lstai-root",
            str(lstai_root.relative_to(repo)),
            "--phase2-csv",
            str(phase2_csv.relative_to(repo)),
            "--out-csv",
            str(phase4_csv.relative_to(repo)),
            "--out-thresholds",
            str(phase4_thr.relative_to(repo)),
            "--out-reports-dir",
            str(phase4_reports.relative_to(repo)),
            "--out-fig-dir",
            str(figs.relative_to(repo)),
            "--example-patients",
            patient_ids[0],
            "--scatter-annotate",
            "all",
        ],
        cwd=repo,
    )

    # 5) Phase 5: features + phenotyping
    features_csv = tables / "features_v1.csv"
    _run(
        [
            sys.executable,
            "scripts/12_phase5_export_features.py",
            "--phase2-csv",
            str(phase2_csv.relative_to(repo)),
            "--phase4-csv",
            str(phase4_csv.relative_to(repo)),
            "--phase3-sensitivity-csv",
            "",
            "--out-csv",
            str(features_csv.relative_to(repo)),
        ],
        cwd=repo,
    )

    _run(
        [
            sys.executable,
            "scripts/13_phase5_phenotyping.py",
            "--features-csv",
            str(features_csv.relative_to(repo)),
            "--feature-set",
            "mode_a_pheno",
            "--missing",
            "drop",
            "--k",
            "2",
            "--n-seeds",
            str(int(args.phenotyping_seeds)),
            "--out-assignments",
            str((tables / "phenotype_assignments.csv").relative_to(repo)),
            "--out-profiles",
            str((tables / "phase5_cluster_profiles.csv").relative_to(repo)),
            "--out-k-metrics",
            str((tables / "phase5_k_selection.csv").relative_to(repo)),
            "--out-fig-latent",
            str((figs / "phase5_latent_space_pca.png").relative_to(repo)),
            "--out-fig-stability",
            str((figs / "phase5_coassignment_heatmap.png").relative_to(repo)),
            "--qc-encoding",
            "marker",
            "--annotate",
            "all",
        ],
        cwd=repo,
    )

    print("")
    print("Toy smoke test finished.")
    print(f"- Outputs: {out_root.relative_to(repo).as_posix()}/")
    print(f"- Toy data: {data_root.relative_to(repo).as_posix()}/")
    print(f"- Toy LST-AI: {lstai_root.relative_to(repo).as_posix()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
