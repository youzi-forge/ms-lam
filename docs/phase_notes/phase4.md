# Phase 4: Voxel→patient uncertainty summaries + minimal QC flags

This phase turns voxel-wise uncertainty signals into **patient-level QC outputs** that help interpret longitudinal change more safely.

Phase 4 is designed to be **engine-agnostic** (works whenever a model produces probability maps), while also allowing
LST-AI-specific enhancements when available.

## Inputs
- Phase 1 LST-AI outputs:
  - `lesion_prob.nii.gz` (ensemble probability map)
  - `lesion_prob_model{1,2,3}.nii.gz` (sub-model probabilities; if available)
  - `lesion_mask.nii.gz`
- `brainmask.nii.gz` (restrict computations to brain)
- Phase 2 metrics table (for context/error plots): `results/tables/phase2_longitudinal_metrics.csv`

## Uncertainty maps

We support multiple voxel-wise uncertainty definitions:

### 1) Probability-map proxies (no extra inference)
- `unc_prob_entropy`: binary entropy of `p(x)`
- `unc_prob_p1mp`: `p(x) * (1 - p(x))`

These are fast and always available when a probability map exists.

### 2) Ensemble disagreement (recommended when available)
If sub-model probability maps are present, compute:
- `unc_ens_var(x) = Var(p1(x), p2(x), p3(x))`

This captures model disagreement and is often more useful for QC than a single-map proxy.

### What drives QC?
The runner chooses a **primary uncertainty** (default: `ens_var`) and uses its **t1** summaries to set thresholds and QC flags.
You can switch to a proxy-based primary uncertainty if sub-model maps are missing:
- `--primary-unc prob_entropy`

## Voxel→patient aggregation

For each timepoint, we summarize uncertainty over interpretable regions:
- `brain`: within `brainmask`
- `lesion`: within predicted lesion mask
- `boundary`: a conservative **outer boundary band** around lesions:
  - `boundary = dilate(lesion, r_mm) & ~lesion` (restricted to brain)

We report simple statistics:
- mean
- p95 (brain-only; outlier-focused)

## QC flags (minimal, cohort-quantile thresholds)

Instead of hand-tuned constants, we derive thresholds from cohort quantiles and save them to:
- `results/tables/phase4_qc_thresholds.json`

Minimal flags:
- `needs_review`: triggered if follow-up uncertainty is high:
  - `unc_mean_lesion_t1` above cohort `p90`, and/or
  - `unc_p95_brain_t1` above cohort `p90`
- `change_not_confident`: triggered if **large** |ΔV| co-occurs with `needs_review`:
  - `|ΔV|` above cohort `p90` **and** `needs_review=True`

These are meant as initial QC rules; thresholds and rules can evolve.

## Grid consistency
We explicitly check that `lesion_prob`, `lesion_mask`, `brainmask`, and the reference FLAIR are on the same grid
(shape + affine; spacing differences are recorded). Resampling is **not** performed by default; mismatches are surfaced in
the per-patient `warnings`.

## Outputs
- `results/tables/phase4_uncertainty_metrics.csv`
- `results/tables/phase4_qc_thresholds.json`
- per-patient reports: `results/reports/phase4/patientXX.json`
- optional voxel maps (if enabled): `data/processed/phase4_uncertainty_maps/patientXX/{t0,t1}/unc_{type}.nii.gz`
- figures:
  - `results/figures/phase4_unc_overlay.png`
  - `results/figures/phase4_unc_vs_error.png`

## Figures (how to interpret)

### Uncertainty overlay
The overlay figure shows a small set of patients:
- any `--example-patients` you provide (defaults to `patient01`),
- plus the cohort maxima (by `unc_mean_lesion_t1` and by `unc_p95_brain_t1`) for the primary uncertainty.

For comparability, the uncertainty color scale is shared across the displayed patients and is chosen from the
lesion/boundary region when available (otherwise brainmask-wide).

### Uncertainty vs error scatter
We plot:
- x: `unc_{primary}_mean_lesion_t1`
- y: `1 - dice_chg_sym_cons` (from Phase 2)

Point annotations are configurable via `--scatter-annotate` (default: QC-flagged points + extremes).

## How to run
Minimal (no voxel maps):
- `python3 scripts/10_phase4_uncertainty_qc.py`

Write voxel maps (can be large):
- `python3 scripts/10_phase4_uncertainty_qc.py --save-maps`

## Example cases (this run)
These brief interpretations refer to the outputs produced by `scripts/10_phase4_uncertainty_qc.py` on `open_ms_data`
with LST-AI as the baseline engine. Exact flagged patients can change if you rerun with different settings.

- `patient04`: `needs_review=True` triggered by `unc_mean_lesion_t1_high` (high lesion-mean ensemble variance). In this run it also shows very high Phase 2 error (`1 - dice_chg_sym_cons`), making it a clear QC-positive example.
- `patient07`: `needs_review=True` triggered by `unc_p95_brain_t1_high` (brain-wide high tail uncertainty), even though lesion-mean uncertainty is not among the highest. This pattern is consistent with more global instability rather than only lesion-boundary ambiguity.
- `patient01`: `needs_review=False` and serves as a convenient non-flagged reference in the overlay.

For reproducible details, consult:
- per-patient QC triggers: `results/reports/phase4/patientXX.json`
- cohort table: `results/tables/phase4_uncertainty_metrics.csv`
