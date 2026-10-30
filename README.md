# legend-optical-map-ml

Machine-learning surrogates of the LEGEND liquid-argon (LAr) optical map, built on
[RESUM_FLEX](https://github.com/annkasch/RESUM_FLEX). This repository holds everything
specific to the optical-map application:

- `legend_optical_map_ml/`: LH5 reader, shared event cache, per-config prepared views,
  role-based splits, features and reports, and `provider.OpticalDataset`, the dataset
  provider RESUM_FLEX reads the prepared data through.
- `configs/`: data, event-model (CNP, MLP, mixup ablation) and GP configs for Ian Wang's
  centered box (`centered_425_375_375`).
- `scripts/slurm/train_surrogate.sbatch`: Perlmutter job script.
- [Configuration guide](docs/configuration.md): how to set up the data, training and GP YAML configs.
- [Optical data guide](docs/optical_data.md): data preparation, roles, event cache and inputs without hits.

Models, training, validation metrics and the multi-fidelity GP are RESUM_FLEX's.

## Setup

Install RESUM_FLEX (with the `gp` extra) and this package into one environment:

```bash
pip install -e /path/to/RESUM_FLEX[gp]
pip install -e .
```

On Perlmutter the RESUM_FLEX environment is reused: `.venv` links to `../RESUM_FLEX/.venv`,
which has this package installed in editable mode.

## Workflow (from the repository root)

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 1. Train an event model; the data are prepared (or updated) from data_config first
.venv/bin/python -m core.surrogates train configs/config.optical.centered.mlp.yaml
# 2. Mean predicted p for the unlabeled LF voxels (LF_primaries)
.venv/bin/python -m core.surrogates predict outputs/optical_resum/centered_mlp_v1/seed0/checkpoints/best \
    outputs/optical_centered_cnp_v0 gp_unlabeled --threads 8
# 3. Two-level GP: MLP mean -> observed HF fraction
.venv/bin/python -m core.surrogates mfgp configs/config.optical.centered.mfgp.yaml
```

On Perlmutter submit each step with
`sbatch -q regular -t 01:00:00 scripts/slurm/train_surrogate.sbatch configs/CONFIG.yaml [train|mfgp]`.
Outputs (prepared data, event cache, runs, Slurm logs) go to `outputs/` (not tracked).
Test sets stay blinded until a config sets `unblind_test: true`.

## Documentation

- [Configuration guide](docs/configuration.md): configure data preparation, event-model runs, and GP runs.
- [Optical data guide](docs/optical_data.md): raw inputs, role assignments, preparation, and the event cache.
- [RESUM_FLEX configuration reference](https://github.com/annkasch/RESUM_FLEX/blob/main/docs/configs.md): all model, training, and GP settings.
- [RESUM_FLEX surrogate workflow](https://github.com/annkasch/RESUM_FLEX/blob/main/docs/surrogates.md): training, validation, checkpoints, and GP workflow.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use small synthetic LH5 fixtures, never the real simulations.
