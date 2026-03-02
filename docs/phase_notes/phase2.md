# Phase 2: Longitudinal monitoring metrics + validation against change GT

This phase turns Phase 1 segmentations into **longitudinal monitoring signals** and validates them against the
`open_ms_data` change ground-truth (`gt.nii.gz`).

## Inputs
- `open_ms_data` (coregistered): `t0_flair`, `t1_flair`, `brainmask`, `gt_change`
- LST-AI outputs (Phase 1): `lesion_mask` at `t0` and `t1` (+ optional prob maps)

## Core definitions (v1)

### Volumes (mm³)
All volumes are reported in **mm³**, computed as:

`volume_mm3 = (#voxels) * (sx * sy * sz)`

The voxel spacing `(sx, sy, sz)` is read from the **reference grid** (t0 FLAIR header).

### Lesion volumes and ΔV
- `V0`: lesion volume at t0
- `V1`: lesion volume at t1
- `ΔV = V1 - V0`

### Change proxies vs change GT
Because `gt_change` is a *change-region* label (not necessarily “new lesions only”), multiple proxies are computed:

1) **Raw new-lesion proxy (sensitive)**
- `new_raw = mask_t1 & (~mask_t0)`

2) **Conservative new-lesion proxy (monitoring-friendly)**
- Dilate baseline lesions in *physical mm* to reduce boundary jitter:
  - `mask0_dil = dilate(mask_t0, radius_mm=2.0)`
- `new_cons = mask_t1 & (~mask0_dil)`
- Remove tiny components by a physical-volume threshold (default: `10 mm³`).

3) **Symmetric change proxy**
- `chg_sym_raw = xor(mask_t1, mask_t0)`
- `chg_sym_cons = xor(mask_t1, mask0_dil)` (+ small-component removal)

Dice scores to `gt_change` are reported for each proxy (with explicit empty-mask handling):
- both empty → Dice = 1.0
- one empty → Dice = 0.0

For each proxy, a simple **volume error** w.r.t. `gt_change` is also reported:
- `volume_error_mm3 = proxy_volume_mm3 - gt_change_volume_mm3`

## GT diagnostics (interpretability)
To make low-Dice cases easier to explain, the pipeline also reports:
- `gt_covered_by_t1_frac`: fraction of `gt_change` voxels covered by the **t1 lesion mask** (a crude “did the model hit the GT?” check).
- `gt_outside_brain_frac`: fraction of `gt_change` voxels outside the provided `brainmask` (flags potential mask/GT inconsistencies).

## Intensity-change (segmentation-independent evidence)
Direct `|FLAIR_t1 - FLAIR_t0|` is sensitive to global scaling differences. To handle this, the pipeline:
- robust-normalize each timepoint **within the brainmask** using percentile scaling (p1–p99),
- compute `diff = |norm(t1) - norm(t0)|` within the brainmask,
- report `diff_mean` and `diff_p95`.

## Design rationale

**Why two families of change proxy (new-lesion + symmetric)?**
The change-region GT in `open_ms_data` labels any visually identifiable change — not only newly appearing lesions, but also enlargement, shrinkage, or altered conspicuity. A "new-lesion" proxy (t1 minus t0) is the natural clinical quantity, but it misses change that manifests as lesion disappearance or boundary shift. The symmetric proxy (XOR of both masks) captures all voxels that differ between timepoints, which is a better semantic match to a change-region annotation. Both are reported because they answer different questions.

**Why 2 mm dilation radius?**
The conservative proxy dilates the baseline mask before computing new/changed regions. The radius needs to be large enough to absorb boundary jitter from segmentation noise and residual misalignment (typically sub-voxel to ~1 voxel), but small enough not to swallow genuinely new lesions that appear near existing ones. At the voxel spacings in this dataset (~0.7–0.9 mm in-plane), 2 mm corresponds to roughly 2–3 voxels — enough to suppress most boundary jitter while preserving lesions that are spatially distinct from the baseline.

**Why 10 mm³ minimum component volume?**
Small connected components in the change proxy are overwhelmingly likely to be noise (partial-volume effects, single-voxel segmentation flicker). At the voxel volumes in this dataset (~1.5–2.4 mm³), 10 mm³ corresponds to roughly 4–7 voxels — below which a "lesion" is not meaningfully interpretable.

**Why p1–p99 for intensity normalization?**
Percentile-based normalization is standard for brain MRI because it is robust to rare extreme-intensity voxels (e.g. vessels, artifacts at the brain boundary). Using p1–p99 rather than min–max excludes the tails while preserving nearly all brain tissue signal. The specific choice of 1st and 99th percentiles is a common convention; narrower ranges (e.g. p5–p95) would compress too much of the intensity range.

**Why both Dice and volume error?**
Dice measures voxel-level spatial overlap — it tells you whether the change proxy is in the right place. Volume error measures how much total change is detected — it tells you whether the magnitude is right. A proxy can have reasonable volume error but terrible Dice (detecting the right amount of change in the wrong place), or vice versa. Reporting both makes failure modes interpretable.

**Empty-mask Dice convention:** if both the proxy and GT are empty, Dice = 1.0 (agreement on "no change"). If exactly one is empty, Dice = 0.0 (total disagreement). This avoids undefined values and is recorded in every per-patient report.

## Expected failure modes / caveats
- Dice(new, GT) may be low if GT includes lesion enlargement/shrinkage, not only “new lesions”.
- New-lesion proxies are sensitive to residual misalignment and segmentation boundary jitter; the conservative proxy reduces this.
- Intensity scaling can vary between timepoints (scanner shift); intensity change is therefore computed after robust normalization.

## Observations on open\_ms\_data

### Overall Dice is low — and this is informative, not just disappointing

`dice_chg_sym_cons` has a median of ~0.16 and a maximum of 0.53 (patient14). 11 of 20 patients score below 0.2. This reflects both the GT semantic gap (change-region ≠ mask XOR) and real cross-timepoint segmentation instability. The takeaway is not “LST-AI is bad” but rather: using segmentation-derived proxies as voxel-level change detectors, without additional QC or robustness validation, is unreliable on this data.

### ΔV is a useful magnitude signal despite poor voxel overlap

|ΔV| correlates well with `change_gt_vol_mm3` (Spearman ~0.79), meaning volume change broadly tracks the scale of real change. However, `chg_sym_cons` systematically overestimates GT change volume: 18 of 20 patients have positive volume error, with a median overshoot of ~5,000 mm³. This is a structural property of the XOR-plus-dilation proxy (it picks up boundary jitter on both sides), not a model-specific artefact.

### Diagnostic fields reveal distinct failure modes

- **”Didn't hit the GT at all”**: patient04 has `gt_covered_by_t1_frac = 0` and `dice_chg_sym_cons = 0` — the t1 segmentation has no overlap with the GT change region despite a modest volume error (~230 mm³). This is a position failure, not a volume failure.
- **”Segmentation missed it but intensity evidence is present”**: patient09 has `gt_covered_by_t1_frac ≈ 0.001` but meaningful intensity-change signal in the GT region (`intensity_diff_frac_gt ≈ 0.069`). This illustrates why the pipeline reports segmentation-independent evidence alongside mask-based proxies.
- **Massive false change in large-lesion patients**: patient07 has `|chg_sym_cons_vol_err| ≈ 58,000 mm³` — the XOR of its large t0 and t1 masks produces an enormous symmetric-change proxy that dwarfs the actual GT change. This is the kind of case that Phase 3 (robustness) and Phase 4 (uncertainty) are designed to flag.

### Intensity drift drives inflated signals

`intensity_diff_p95` and `|ΔV|` correlate at r ≈ 0.59 across the 20 patients. The most extreme intensity-ratio cases (patient20: ratio 0.09, patient04: ratio 2.28) are also among the worst performers on Dice. Small-GT patients are particularly vulnerable: their real change signal is small relative to the drift-induced noise, making monitoring signals unreliable without additional safeguards.
