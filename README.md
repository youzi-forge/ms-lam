# MS-LAM: Longitudinal Brain MRI Monitoring Benchmark (baseline-first, validation-first)

This repo is a reproducible **longitudinal brain MRI monitoring baseline** built on public datasets.

## Phase 0 Quickstart (data manifest + sanity + example figures)

Prereqs:
- Python 3
- Install deps: `pip3 install -r requirements.txt`

Run:
1) Build manifest (relative paths):
   - `python3 scripts/01_make_manifest.py --data-root data/raw/open_ms_data/longitudinal/coregistered --out results/tables/phase0_manifest.csv`
2) Sanity checks + checked manifest:
   - `python3 scripts/02_phase0_sanity.py --manifest results/tables/phase0_manifest.csv --out-report results/tables/phase0_sanity_report.csv --out-manifest results/tables/phase0_manifest_checked.csv`
3) Make 2–3 example figures:
   - `python3 scripts/03_phase0_make_figures.py --manifest results/tables/phase0_manifest_checked.csv --out-dir results/figures --num-patients 3`

Outputs:
- `results/tables/phase0_manifest.csv`
- `results/tables/phase0_sanity_report.csv`
- `results/figures/phase0_patientXX_zZZ.png`

## Phase 1: Pretrained baseline inference (LST-AI via Docker)

Prereqs:
- Docker Desktop running
- LST-AI image pulled: `docker pull jqmcginnis/lst-ai:v1.2.0`
- On Apple Silicon, run with `--platform linux/amd64` (slower; recommended for smoke tests only)

### Smoke test (1–2 patients)

Run LST-AI on t0/t1 and export canonical outputs:
- `python3 scripts/04_run_lstai_batch.py --patients patient01,patient02 --timepoints both`

Outputs per patient/timepoint:
- `data/processed/lstai_outputs/patientXX/t0/lesion_mask.nii.gz`
- `data/processed/lstai_outputs/patientXX/t0/lesion_prob.nii.gz`
- `data/processed/lstai_outputs/patientXX/t0/lesion_prob_model1.nii.gz` (and 2/3)

Runlog:
- `results/tables/phase1_lstai_runlog.csv`

Example overlays:
- `python3 scripts/05_phase1_overlay_examples.py --patients patient01,patient02`

Notes:
- Do **not** pass `--stripped` for `open_ms_data` by default (brainmask exists, but images are not necessarily skull-stripped).
- LST-AI `--output` is a **directory** (the image help text is misleading in some places).
- With `--probability_map`, LST-AI writes probmaps into `--temp`; this repo copies them into the canonical filenames above.
