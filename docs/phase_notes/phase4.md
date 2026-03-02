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

Multiple voxel-wise uncertainty definitions are supported:

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

For each timepoint, uncertainty is summarized over interpretable regions:
- `brain`: within `brainmask`
- `lesion`: within predicted lesion mask
- `boundary`: a conservative **outer boundary band** around lesions:
  - `boundary = dilate(lesion, r_mm) & ~lesion` (restricted to brain)

Reported statistics:
- mean
- p95 (brain-only; outlier-focused)

## Design rationale

**Why ensemble variance as the primary uncertainty?**
LST-AI is a 3-model ensemble. When sub-model probability maps are available, their voxel-wise variance directly measures model disagreement — the most informative uncertainty signal extractable without extra inference. Single-map proxies (entropy, p(1−p)) only measure "how far from 0 or 1 is the ensemble probability," which conflates genuine boundary ambiguity with calibration effects. Ensemble variance is zero when all models agree (regardless of the probability value) and high only when models disagree, making it a cleaner signal for QC.

**Why p90 quantile for QC thresholds?**
The goal is to flag outliers within the cohort, not define absolute "safe/unsafe" thresholds (which would require external calibration data). The 90th percentile flags the top ~10% — aggressive enough to catch meaningful outliers on a 20-patient cohort (flagging ~2 patients per metric), but not so aggressive that the majority of patients are flagged. This is a pragmatic starting point; on larger cohorts, a higher quantile (e.g. p95) might be more appropriate.

**Why does `needs_review` use OR (high lesion-mean OR high brain-p95)?**
The two metrics capture different failure modes. High lesion-mean uncertainty indicates disagreement specifically at the lesion boundaries (the model is unsure about the segmentation it produced). High brain-wide p95 indicates instability far from lesions (the model's predictions are noisy across the brain). Either condition alone is cause for concern, so the QC flag triggers on either.

**Why does `change_not_confident` require BOTH high |ΔV| AND `needs_review`?**
A large volume change is only worrying if there is reason to doubt the segmentation that produced it. If |ΔV| is large but uncertainty is low, the model is at least internally confident about the change (whether or not it is correct). Conversely, if uncertainty is high but |ΔV| is small, the segmentation may be unreliable but the monitoring conclusion ("not much changed") is unlikely to be wrong by a clinically meaningful amount. The compound flag targets the intersection: cases where a large monitoring signal might be driven by unreliable segmentation.

## QC flags (minimal, cohort-quantile thresholds)

Instead of hand-tuned constants, thresholds are derived from cohort quantiles and saved to:
- `results/tables/phase4_qc_thresholds.json`

Minimal flags:
- `needs_review`: triggered if follow-up uncertainty is high:
  - `unc_mean_lesion_t1` above cohort `p90`, and/or
  - `unc_p95_brain_t1` above cohort `p90`
- `change_not_confident`: triggered if **large** |ΔV| co-occurs with `needs_review`:
  - `|ΔV|` above cohort `p90` **and** `needs_review=True`

These are meant as initial QC rules; thresholds and rules can evolve.

## Grid consistency
The runner explicitly checks that `lesion_prob`, `lesion_mask`, `brainmask`, and the reference FLAIR are on the same grid
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
  - `results/figures/phase4_unc_vs_shift_sens_deltaV.png` *(optional; requires Phase 3 outputs)*
  - `results/figures/phase4_unc_vs_shift_sens_dice.png` *(optional; requires Phase 3 outputs)*

## Figures (how to interpret)

### Uncertainty overlay
The overlay figure shows a small set of patients:
- any `--example-patients` you provide (defaults to `patient01`),
- plus the cohort maxima (by `unc_mean_lesion_t1` and by `unc_p95_brain_t1`) for the primary uncertainty.

For comparability, the uncertainty color scale is shared across the displayed patients and is chosen from the
lesion/boundary region when available (otherwise brainmask-wide).

### Uncertainty vs error scatter
The scatter shows:
- x: `unc_{primary}_mean_lesion_t1`
- y: `1 - dice_chg_sym_cons` (from Phase 2)

Point annotations are configurable via `--scatter-annotate` (default: QC-flagged points + extremes).

## How to run
Minimal (no voxel maps):
- `python3 scripts/10_phase4_uncertainty_qc.py`

Write voxel maps (can be large):
- `python3 scripts/10_phase4_uncertainty_qc.py --save-maps`

### Optional: relate uncertainty to robustness sensitivity (Phase 3)
If you have Phase 3 outputs (`results/tables/phase3_robustness.csv`) for a subset or the full cohort, you can merge:

- Phase 4 uncertainty summaries (x-axis), and
- Phase 3 shift sensitivity (y-axis; e.g. `p95(|ΔV_shift − ΔV_base|)` across shift×level)

and generate scatter plots:

- `python3 scripts/11_phase4_uncertainty_vs_shift_sensitivity.py`

This produces:
- merged table: `results/tables/phase4_uncertainty_vs_shift_sensitivity.csv`
- figures:
  - `results/figures/phase4_unc_vs_shift_sens_deltaV.png`
  - `results/figures/phase4_unc_vs_shift_sens_dice.png`

Interpretation goal: check whether **high uncertainty** cases also tend to be **more shift-sensitive** (higher false-change risk).

Tip: for small cohorts (e.g., the representative Phase 3 subset), annotating all points can be helpful:
- `python3 scripts/11_phase4_uncertainty_vs_shift_sensitivity.py --annotate all`

## Observations on open\_ms\_data

Settings for this run: primary=`ens_var`, `qc_quantile=0.9`, `change_quantile=0.9`, `example_patients=patient01`. Exact flagged patients may change if you rerun with different settings.

### QC flags and their two failure modes

4 of 20 patients are flagged `needs_review=True`: patient04, patient07, patient09, patient17. They separate into two distinct patterns:

- **Focal lesion-boundary disagreement** (patient04, patient09): triggered by `unc_mean_lesion_t1_high`. The ensemble sub-models disagree specifically around lesion boundaries. Patient04 is the worst segmentation failure in Phase 2 (Dice = 0, GT completely missed), and also has the highest lesion-mean uncertainty in the cohort (~0.164). Patient09 shows a similar pattern with high lesion-mean uncertainty (~0.128) and very low GT coverage (~0.001).
- **Global prediction instability** (patient07, patient17): triggered by `unc_p95_brain_t1_high`. The disagreement is not confined to lesions — the brain-wide p95 of ensemble variance is elevated (patient07: ~6.3e-4, highest in cohort). Patient07 also has the largest symmetric-change volume error in Phase 2 (~58,000 mm³), consistent with widespread segmentation instability producing massive false change.

### What QC misses

5 patients have `dice_chg_sym_cons ≤ 0.1`, but only 2 are flagged (patient04, patient07). The unflagged cases (patient13, patient15, patient20) have unremarkable ensemble variance — the sub-models agree with each other, they are just collectively wrong. This "confident but wrong" failure mode is a known limitation of ensemble-based uncertainty and cannot be caught by variance alone.

### Uncertainty vs Phase 2 error

There is a moderate positive trend between `unc_ens_var_mean_lesion_t1` and `1 − dice_chg_sym_cons` (the scatter in `phase4_unc_vs_error.png`), but the correlation is not strong (r ≈ 0.45). Uncertainty is directionally useful for QC but not a reliable error predictor.

### Uncertainty vs shift sensitivity (Phase 3 subset)

On the 8-patient Phase 3 subset, lesion-mean uncertainty does **not** predict shift sensitivity. Patient19 — by far the most shift-vulnerable case (|ΔΔV| up to ~20,800 mm³) — has one of the lowest lesion-mean uncertainties (~0.019). This means uncertainty-based QC and robustness testing provide complementary, not redundant, information: a patient can pass the uncertainty check but still be highly vulnerable to scanner/protocol changes.

### Example overlay interpretation

- `patient04`: high lesion-mean uncertainty, focal hotspots around predicted lesions. Corresponds to the worst Phase 2 failure.
- `patient07`: elevated uncertainty broadly across the brain, not just at lesion boundaries. Corresponds to the largest false-change volume in Phase 2.
- `patient01`: low uncertainty, non-flagged reference.

For per-patient QC triggers, see `results/reports/phase4/patientXX.json`.
