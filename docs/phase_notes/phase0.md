# Phase 0: Data inventory + sanity checks + minimal longitudinal evidence

Deliverables:
- `results/tables/phase0_manifest.csv`
- `results/tables/phase0_sanity_report.csv`
- `results/figures/phase0_patientXX_zZZ.png`

Notes:
- The open_ms_data `longitudinal/coregistered` layout is stable (patient folders + fixed filenames).
- Sanity checks use **tolerant** floating comparisons for affine/spacing to avoid false failures from tiny float diffs.

