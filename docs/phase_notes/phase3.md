# Phase 3: Robustness sensitivity under scanner/protocol shift (shift_v1)

This phase stress-tests the **monitoring + validation harness** (Phases 1–2) under controlled input perturbations
that mimic scanner/protocol heterogeneity.

We treat LST-AI as a fixed pretrained engine and measure how longitudinal metrics change under shift.

## Experimental design (v1)

### Who gets shifted?
Default mode is **`t1_only`**: only the follow-up (t1) inputs are shifted, while t0 is kept unchanged.
This matches the practical monitoring scenario where scanner/protocol differs at follow-up and may create *spurious change*.

We also support a control mode **`both`** (shift t0 and t1 with the same parameters). In a healthy pipeline,
same-source shifts should generally perturb longitudinal **Δ** metrics less than shifting only one timepoint.

## Suggested patient subset (representative mini-batch)
When iterating locally (or when compute is limited), it is reasonable to run Phase 3 on a representative subset
instead of the full cohort. A simple, reproducible selection rule is:
- Take patients with **largest** and **smallest** `change_gt_vol_mm3` (large-change vs near-zero regime),
- Take patients with **highest** and **lowest** baseline `dice_chg_sym_cons` (best-case vs worst-case regime),
- De-duplicate with the priority: worst-dice → highest-GT → lowest-GT → best-dice, and fill with median-ish cases if needed.

This rule is implemented in `scripts/09_select_phase3_patients.py`.

## shift_v1 suite (deterministic, levelled)

Each shift is defined as `shift_name × severity_level`:
- `level0`: identity (baseline)
- `level1/2`: increasing perturbation strength

Shifts implemented (v1):
- `gamma`: intensity gamma change with a small brightness offset in normalized space (levels: 1.0 / 0.8(+0.02) / 1.2(−0.02))
- `noise`: Gaussian noise with sigma relative to brainmask intensity range (levels: 0 / 0.01 / 0.03 of `(p99-p1)`)
- `resolution`: downsample→upsample back to original grid (in-plane XY) (levels: factor 1.0 / 1.5 / 2.0)
- `blur`: Gaussian blur in mm (levels: 0 / 0.5 / 1.0 mm) *(optional; not enabled by default)*

### Synchronization rules
- All geometric/resolution-related shifts are applied **synchronously** to T1 and FLAIR (same parameters).
- All outputs are written back on the **original reference grid** (same shape/affine), so `brainmask`/`gt_change` can be reused.
- After intensity-affecting shifts, values are clipped back to a brainmask-based percentile range (p1–p99) to avoid invalid inputs.

## Outputs

The Phase 3 runner produces:
- `results/tables/phase3_robustness.csv`: per-patient, per-shift, per-level metrics + sensitivity vs baseline
- `results/tables/phase3_robustness_summary.csv`: aggregated mean/std per shift/level
- `results/tables/phase3_runlog.csv`: LST-AI docker run log for shifted inference
- `results/figures/phase3_robustness_curve_deltaV.png`: sensitivity curve for |ΔV_shift − ΔV_base|
- `results/figures/phase3_robustness_curve_dice.png`: sensitivity curve for Dice_sym_cons shift-induced change
- `results/figures/phase3_robustness_curve_deltaV_robust.png`: median (IQR) sensitivity curve for |ΔV_shift − ΔV_base|
- `results/figures/phase3_robustness_curve_dice_robust.png`: median (IQR) sensitivity curve for Dice_sym_cons shift-induced change
- `results/figures/phase3_sensitive_case.png`: a worst-case example across severity levels

## Key interpretation (what this phase means)
- Large |ΔΔV| under `t1_only` indicates that follow-up heterogeneity can create **false change** in monitoring signals.
- Dice drops (vs change-GT) under shift indicate degraded agreement with observed change regions, helping diagnose “unsafe” conditions.

### Mean±std vs median(IQR)
We save both:
- **mean±std** curves: emphasize *tail risk* (but can be dominated by outliers on small cohorts),
- **median(IQR)** curves: emphasize *typical* sensitivity and are more robust to a single extreme case.

## Figures (recommended reading order)

### Typical sensitivity (robust)
These summarize the *typical* impact of shift on monitoring/validation signals (median with IQR band):

![Phase 3 robust ΔV curve](../../results/figures/phase3_robustness_curve_deltaV_robust.png)

- y-axis is `|ΔV_shift − ΔV_base|` (mm³). If this grows with severity, follow-up heterogeneity can create *false change*.

![Phase 3 robust Dice curve](../../results/figures/phase3_robustness_curve_dice_robust.png)

- y-axis is `Dice_sym_cons(shift) − Dice_sym_cons(base)` (to change-GT). Negative values indicate degraded agreement under shift.

### Tail risk / failure mode (single worst-case)
This figure intentionally shows **one** patient: the worst case selected by largest `|ΔΔV|` among shifted runs.
It is meant for diagnosing *how* shift breaks the pipeline, not for representing a typical patient.

![Phase 3 sensitive case](../../results/figures/phase3_sensitive_case.png)
