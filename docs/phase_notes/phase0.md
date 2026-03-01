# Phase 0: Data inventory + sanity checks

Phase 0 enumerates the dataset, validates geometric consistency across timepoints, and records basic statistics so that downstream phases can trust their assumptions (same grid, same affine, all files present).

## Why this matters

Longitudinal pipelines silently break when timepoints have mismatched grids, missing files, or corrupted headers. Phase 0 catches these issues before any inference runs, so failures in later phases are never caused by data-level surprises.

## Inputs

From `open_ms_data/longitudinal/coregistered/` (per patient):
- `study1_T1W.nii.gz`, `study1_FLAIR.nii.gz` (t0)
- `study2_T1W.nii.gz`, `study2_FLAIR.nii.gz` (t1)
- `brainmask.nii.gz`
- `gt.nii.gz` (change-region ground truth)

## What it checks

Per patient:
- File existence for all required inputs
- Shape consistency across t0, t1, brainmask, and GT
- Affine/spacing consistency (tolerant `allclose` comparisons to avoid false failures from tiny float diffs in NIfTI headers)
- Extracts voxel spacing and GT change volume for downstream use

## Outputs

- `results/tables/phase0_manifest.csv` — raw file listing
- `results/tables/phase0_manifest_checked.csv` — filtered manifest with `ok` column
- `results/tables/phase0_sanity_report.csv` — per-patient validation report
- `results/figures/phase0_patientXX_zZZ.png` — optional visual sanity (t0, t1, |diff|, GT overlay)

## Observations on open\_ms\_data

All 20 patients pass: shapes, affines, and voxel spacings are consistent across timepoints and files. The dataset's geometric integrity is solid.

Change-GT volume (`gt.nii.gz`) ranges from 416 mm³ (patient20) to 24,309 mm³ (patient19), with a median around 4,200 mm³. No patient has zero GT change, so the cohort covers a useful spread of change magnitudes.

A minor data-level note: in 4 patients (patient03, 04, 06, 12) a small fraction of GT voxels fall outside the provided brainmask (`gt_outside_brain_frac` up to ~0.035). This is negligible for most purposes but can interact with brainmask-restricted computations in later phases (e.g. intensity-change statistics, uncertainty aggregation).

