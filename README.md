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
