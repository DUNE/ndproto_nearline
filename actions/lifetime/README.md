# DUNE 2x2 — Hit-Based Electron Lifetime

Exploratory electron lifetime measurement using charge packets from the DUNE 2x2 liquid argon time projection chamber (LArTPC).

## Current analysis

## Command to run the script

### General command (any user)

Run this from a working directory containing the necessary analysis dependencies, selected-packet caches, and access to the DUNE 2x2 FLOW files:

```bash
cd /path/to/your/analysis-working-directory
conda activate YOUR_CONDA_ENVIRONMENT

python -u /path/to/run_hit_based_v2_validation.py \
  --bootstrap 30 \
  --daq-audit \
  --per-file-plots \
  --outdir /path/to/output-directory
```

The `src/` script requires supporting analysis modules and external detector-data files. This repository alone is not a standalone numerical rerun environment.

### NERSC command (original analysis environment)

```bash
cd /pscratch/sd/a/alqaisia/hit_based_LAST
conda activate e-lifetime

python -u run_hit_based_v2_validation.py \
  --bootstrap 30 \
  --daq-audit \
  --per-file-plots \
  --outdir hit_based_v2_full
```

## Results

The saved July 2024 lifetime measurements and comparisons are organized in three folders:

- **[Current V2 results](results/v2/):** lifetime time series, per-IO/per-multiplicity charge MPVs, DAQ diagnostics, and plots comparing exact **n=2**, exact **n=3**, and shared **n=2+n=3** fits.
- **[Comparison with Brooke Russell](results/brooke/):** overlay plots comparing our V2 lifetime estimates with Brooke's combined-detector results. Brooke's values are approximately digitized and the plotted timestamp pairings are assumed; this is a qualitative comparison, not a verified run-matched agreement test.
- **[Earlier results](results/legacy/):** n=2 split-IO lifetime and MPV diagnostics retained for comparison.

The combined n=2+n=3 fit uses separate charge normalizations for each IO group and multiplicity but a common attenuation lifetime. The n=3-only result has larger uncertainties and greater variation than n=2; adding n=3 generally has a modest effect on the shared fitted lifetime.

## Environment

Python dependencies are listed in `requirements.txt`.

The numerical analysis was performed at NERSC using the `e-lifetime` conda environment.

## Data availability

Raw packet HDF5 files, FLOW files, and selected-packet caches are not distributed in this repository.

A complete rerun requires the relevant DUNE 2x2 datasets and cached selection metadata on NERSC.

## Current limitations

The physical interpretation of IO6 EXT as interaction t0, the precise same-channel retrigger grouping, and the treatment of above-threshold singletons are still under investigation.

No experimental or simulation discriminator-threshold correction is currently applied to the lifetime measurement.

This repository preserves a preliminary method-development analysis, not a finalized calibrated electron lifetime result.
