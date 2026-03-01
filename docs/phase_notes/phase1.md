# Phase 1: Baseline inference integration (LST-AI) + canonical outputs

This phase integrates **LST-AI** as a fixed pretrained lesion segmentation engine and defines a stable output
contract for downstream monitoring and validation.

## Why this design?

- **baseline-first**: avoid bespoke training and treat segmentation as an interchangeable engine.
- **validation-first**: export masks/probabilities in a consistent structure with per-run logs and metadata, so failures are auditable.

## Inputs

From `open_ms_data/longitudinal/coregistered/` (per patient):
- `study1_T1W.nii.gz` + `study1_FLAIR.nii.gz` (t0)
- `study2_T1W.nii.gz` + `study2_FLAIR.nii.gz` (t1)
- `brainmask.nii.gz` *(not passed to LST-AI by default; used downstream)*

## Engine: LST-AI via Docker

MS-LAM uses the Docker image:
- `jqmcginnis/lst-ai:v1.2.0`

On Apple Silicon, the runner uses `--platform linux/amd64` for compatibility.

### Important CLI semantics (validated)

- `--output` is a **directory**, not a single file path. LST-AI writes `space-flair_seg-lst.nii.gz` into it.
- With `--probability_map`, LST-AI writes probability maps under `--temp`.
  If `--temp` is not set, those files may be deleted when the run finishes.

Because upstream docs can be inconsistent, always trust `lst --help` from the actual runtime image.

## Canonical output contract

After each run, MS-LAM exports LST-AI outputs into:

`data/processed/lstai_outputs/patientXX/{t0,t1}/`

Canonical filenames:
- `lesion_mask.nii.gz` (binary lesion mask)
- `lesion_prob.nii.gz` (ensemble probability map)
- `lesion_prob_model1.nii.gz`
- `lesion_prob_model2.nii.gz`
- `lesion_prob_model3.nii.gz`

Run metadata and logs:
- `lstai_run.json` (arguments, returncode, runtime, resolved host paths)
- `lstai_stdout.txt`, `lstai_stderr.txt`

## Runner scripts

### Batch inference

`scripts/04_run_lstai_batch.py` reads the checked manifest and runs LST-AI for t0/t1.

Key properties:
- **resume**: a run directory is considered complete only if `lstai_run.json` indicates `returncode==0` and required outputs exist.
- **no silent fallbacks**: missing expected outputs or shape mismatches raise explicit errors.
- per-run directories prevent overwrite/contamination.

Runlog:
- `results/tables/phase1_lstai_runlog.csv`

### Visual spot-checks

- `scripts/05_phase1_overlay_examples.py`: lightweight PNG overlays for selected patients.
- `scripts/06_phase1_qc_report.py`: cohort PDF to scan for obvious failures without generating dozens of separate figures.

## Practical notes / common pitfalls

- **Do not pass `--stripped` by default** for `open_ms_data`.
  Images are coregistered/N4-corrected, but not guaranteed to be skull-stripped. Passing `--stripped` incorrectly will severely degrade segmentation.
- **Docker memory**: if you see return code `137`, it is typically an OOM kill.
  Increase Docker Desktop memory and consider `--fast-mode` + fewer `--threads`.
- **Apple Silicon**: CPU-mode Docker emulation (`--platform linux/amd64`) works but is slow (~2–2.5 min per timepoint). For faster iteration, consider running inference on a Linux machine or Colab and copying outputs back.

## Observations on open\_ms\_data

All 40 inference runs (20 patients × 2 timepoints) completed successfully (returncode = 0). Outputs are grid-consistent with inputs, so downstream logical operations (mask XOR, Dice, etc.) work without resampling.

The most striking observation at this stage is the cross-timepoint FLAIR intensity drift visible in `results/tables/phase1_mask_volumes.csv`. The `t1_over_t0_median_ratio` ranges from 0.09 (patient20 — follow-up dramatically darker) to 2.28 (patient04 — follow-up dramatically brighter). This is not a data-loading artefact: the images are coregistered and N4-corrected, but the underlying acquisitions clearly differ in global intensity scale. This drift directly impacts segmentation stability across timepoints, and much of what Phase 2 later measures as "longitudinal change" is, in the extreme cases, really "intensity-drift-induced segmentation instability."

Per-timepoint inference runtime on Apple Silicon (CPU Docker emulation, `fast_mode=True`): median ~141 seconds.

