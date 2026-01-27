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
Because `gt_change` is a *change-region* label (not necessarily “new lesions only”), we compute multiple proxies:

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

We report Dice scores to `gt_change` for each proxy (with explicit empty-mask handling):
- both empty → Dice = 1.0
- one empty → Dice = 0.0

For each proxy we also report a simple **volume error** w.r.t. `gt_change`:
- `volume_error_mm3 = proxy_volume_mm3 - gt_change_volume_mm3`

## GT diagnostics (interpretability)
To make low-Dice cases easier to explain, we also report:
- `gt_covered_by_t1_frac`: fraction of `gt_change` voxels covered by the **t1 lesion mask** (a crude “did we hit the GT?” check).
- `gt_outside_brain_frac`: fraction of `gt_change` voxels outside the provided `brainmask` (flags potential mask/GT inconsistencies).

## Intensity-change (segmentation-independent evidence)
Direct `|FLAIR_t1 - FLAIR_t0|` is sensitive to global scaling differences. We therefore:
- robust-normalize each timepoint **within the brainmask** using percentile scaling (p1–p99),
- compute `diff = |norm(t1) - norm(t0)|` within the brainmask,
- report `diff_mean` and `diff_p95`.

## Expected failure modes / caveats
- Dice(new, GT) may be low if GT includes lesion enlargement/shrinkage, not only “new lesions”.
- New-lesion proxies are sensitive to residual misalignment and segmentation boundary jitter; the conservative proxy reduces this.
- Intensity scaling can vary between timepoints (scanner shift); intensity change is therefore computed after robust normalization.
