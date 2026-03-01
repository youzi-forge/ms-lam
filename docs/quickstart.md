# Full quickstart (recommended workflow)

This guide walks through the complete pipeline on `open_ms_data` with LST-AI. For a zero-dependency smoke test, see the README.

## 1) Get data (recommended: clone `open_ms_data`)

```bash
mkdir -p data/raw
git clone --depth 1 https://github.com/muschellij2/open_ms_data.git data/raw/open_ms_data
```

We use: `data/raw/open_ms_data/longitudinal/coregistered/`.

## 2) Build a manifest + run sanity checks

```bash
python3 scripts/01_make_manifest.py \
  --data-root data/raw/open_ms_data/longitudinal/coregistered \
  --out results/tables/phase0_manifest.csv

python3 scripts/02_phase0_sanity.py \
  --manifest results/tables/phase0_manifest.csv \
  --out-report results/tables/phase0_sanity_report.csv \
  --out-manifest results/tables/phase0_manifest_checked.csv
```

Optional: produce a few example figures:

```bash
python3 scripts/03_phase0_make_figures.py \
  --manifest results/tables/phase0_manifest_checked.csv \
  --out-dir results/figures \
  --num-patients 3
```

## 3) Run baseline inference (LST-AI)

Smoke test on 1–2 patients:

```bash
python3 scripts/04_run_lstai_batch.py --patients patient01,patient02 --timepoints both
```

Outputs per patient/timepoint:

* `data/processed/lstai_outputs/patientXX/t0/lesion_mask.nii.gz`
* `data/processed/lstai_outputs/patientXX/t0/lesion_prob.nii.gz`
* `data/processed/lstai_outputs/patientXX/t0/lesion_prob_model1.nii.gz` (and 2/3)

Example overlays:

```bash
python3 scripts/05_phase1_overlay_examples.py --patients patient01,patient02
```

Optional cohort scan (single PDF):

```bash
python3 scripts/06_phase1_qc_report.py
```

Notes:

* Do **not** pass `--stripped` for `open_ms_data` by default (brainmask exists, but images are not necessarily skull-stripped).
* In the `jqmcginnis/lst-ai:v1.2.0` Docker image, LST-AI `--output` behaves as a **directory** (despite some help text suggesting a file path).
* With `--probability_map`, LST-AI writes probmaps under `--temp`; this repo copies them into canonical filenames.

## 4) Generate monitoring metrics + validate against change-GT

```bash
python3 scripts/07_eval_longitudinal.py
```

Key outputs:

* `results/tables/phase2_longitudinal_metrics.csv`
* `results/reports/phase2/patientXX.json`
* `results/figures/phase2_examples.png`
* `results/figures/phase2_worst_case.png`
* `results/figures/phase2_deltaV_hist.png`

All volumes are in mm³ (spacing-aware). Two change proxies are computed: a conservative new-lesion proxy (baseline mask dilated by 2 mm before diffing, then small-component filtering) and a symmetric change proxy (XOR of both masks, similarly cleaned), which tends to align better with change-region GT semantics. Segmentation-independent intensity-change evidence (brainmask-normalized) is reported alongside.

## 5) Robustness sensitivity under simulated shift (shift_v1)

Smoke test:

```bash
python3 scripts/08_run_robustness_suite.py --patients patient01,patient02 --mode t1_only --levels 0,1
```

Representative mini-batch (rule-based selection from Phase 2 table):

```bash
python3 scripts/09_select_phase3_patients.py
python3 scripts/08_run_robustness_suite.py \
  --patients $(paste -sd, results/tables/phase3_selected_patients.txt) \
  --mode t1_only \
  --levels 0,1,2
```

Outputs:

* `results/tables/phase3_robustness.csv` + `results/tables/phase3_robustness_summary.csv`
* `results/figures/phase3_robustness_curve_deltaV.png`
* `results/figures/phase3_robustness_curve_dice.png`
* `results/figures/phase3_robustness_curve_deltaV_robust.png`
* `results/figures/phase3_robustness_curve_dice_robust.png`
* `results/figures/phase3_sensitive_case.png`

Notes:
- For interpretation (median/IQR vs mean±std; and why the sensitive-case figure shows exactly one patient), see `docs/phase_notes/phase3.md`.

## 6) Uncertainty maps (voxel → patient) + QC flags

Compute voxel-level uncertainty proxies from saved probability maps, aggregate to patient-level summaries, derive
cohort-quantile thresholds, and write QC flags + reports:

```bash
python3 scripts/10_phase4_uncertainty_qc.py
```

Optional:
- save voxel maps (can be large): `python3 scripts/10_phase4_uncertainty_qc.py --save-maps`
- show more examples in the overlay: `--example-patients patient01,patient04,patient07`
- annotate more points in the scatter: `--scatter-annotate qc` (default) or `all`
- relate uncertainty to shift sensitivity (requires Phase 3 outputs): `python3 scripts/11_phase4_uncertainty_vs_shift_sensitivity.py`

Outputs:
- `results/tables/phase4_uncertainty_metrics.csv`
- `results/tables/phase4_qc_thresholds.json`
- `results/reports/phase4/patientXX.json`
- `results/figures/phase4_unc_overlay.png`
- `results/figures/phase4_unc_vs_error.png`
- `results/tables/phase4_uncertainty_vs_shift_sensitivity.csv` *(requires Phase 3 outputs)*
- `results/figures/phase4_unc_vs_shift_sens_deltaV.png` *(requires Phase 3 outputs)*
- `results/figures/phase4_unc_vs_shift_sens_dice.png` *(requires Phase 3 outputs)*

## 7) Exploratory phenotyping (features → latent space)

Export a patient-level feature table (Phase 2 + Phase 4; optionally includes Phase 3 sensitivity if available):

```bash
python3 scripts/12_phase5_export_features.py
```

Run a minimal, reproducible phenotyping pipeline (PCA + k-means + stability):

```bash
python3 scripts/13_phase5_phenotyping.py --feature-set mode_a_pheno
```

Outputs:
- `results/tables/features_v1.csv`
- `results/tables/phenotype_assignments.csv`
- `results/tables/phase5_cluster_profiles.csv`
- `results/tables/phase5_k_selection.csv`
- `results/figures/phase5_latent_space_pca.png`
- `results/figures/phase5_coassignment_heatmap.png`

Notes:
- Phase 5 is exploratory; for interpretation and recommended visualizations, see `phase_notes/phase5.md`.
