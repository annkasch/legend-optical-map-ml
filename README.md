# legend-optical-map-ml

Machine-learning surrogates of the LEGEND liquid-argon (LAr) optical map, built on
[RESUM_FLEX](https://github.com/annkasch/RESUM_FLEX). This repository holds everything
specific to the optical-map application:

- `legend_optical_map_ml/`: LH5 reader, shared event cache, per-config prepared views,
  role-based splits, features and reports, and `provider.OpticalDataset`, the dataset
  provider RESUM_FLEX reads the prepared data through.
- `configs/`: data, event-model (CNP, MLP, mixup ablation) and GP configs for Ian Wang's
  centered box (`centered_425_375_375`), plus [LBNL CSV-manifest data](configs/config.optical.lbnl_secondment.data.yaml)
  with the models compared on it (see the [benchmark protocol](docs/benchmark.md)):
  - event models: [MLP](configs/config.optical.lbnl_secondment.mlp.yaml), MLP with a Bayesian
    last layer (`mlp_bll`), neural fields (`field_f1` … `field_f8`, maximum Fourier frequency);
  - GPs: binomial count GP (`binomial`, with the MLP covariate: `binomial.mlp`,
    `binomial.mlp_lf2`), log-space MFGP on the MLP mean (`mfgp.log`) or raw LF (`mfgp.raw_lf.log`);
  - [countfield](https://github.com/annkasch/countfield) runs (`countfield.lbnl_secondment*`):
    neural process and Point Transformer variants.
- `scripts/`: `export_counts.py` / `export_photons.py` (countfield inputs),
  `score_event_model.py` (a Bayesian-last-layer checkpoint scored on HF validation voxels).
- `scripts/slurm/`: Perlmutter job scripts for individual stages and dependent pipelines.
- [Configuration guide](docs/configuration.md): how to set up the data, training and GP YAML configs.
- [Optical data guide](docs/optical_data.md): data preparation, roles, event cache and inputs without hits.

Models, training, validation metrics and the multi-fidelity GP are RESUM_FLEX's.

## Setup

Create one environment in this checkout and use RESUM_FLEX's lockfile to install its
dependencies, including CPU-only PyTorch. Run these commands from the optical-map repository
root; use Python 3.11 or 3.12:

```bash
APP_ROOT=$PWD
RESUM_FLEX_ROOT=/path/to/RESUM_FLEX
uv venv --python 3.11 .venv  # or use 3.12
(
  cd "$RESUM_FLEX_ROOT"
  UV_PROJECT_ENVIRONMENT="$APP_ROOT/.venv" uv sync --extra gp
)
uv pip install --python .venv/bin/python -e .
```

Replace `/path/to/RESUM_FLEX` with the path to your RESUM_FLEX checkout. `uv sync` installs
RESUM_FLEX into the app checkout's `.venv`; the last command installs this package there too.
Training commands and Slurm then use that same environment. A `.venv` symlink or environment
from another checkout is not required.

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

On Perlmutter, create the Slurm log directory and submit each step from the repository root:

```bash
mkdir -p outputs/slurm_logs
sbatch -q regular -t 01:00:00 scripts/slurm/train_surrogate.sbatch configs/CONFIG.yaml [train|mfgp]
```

To submit training, mean prediction, and GP fitting as dependent jobs, use:

```bash
scripts/slurm/submit_surrogate_pipeline.sh \
  configs/config.optical.centered.mlp.yaml \
  configs/config.optical.centered.mfgp.yaml \
  -q regular -t 01:00:00
```

See [the Slurm pipeline guide](docs/slurm_pipeline.md) for details.

The batch script uses this checkout's `.venv/bin/python` by default; set `RESUM_PYTHON` to
the executable path of another interpreter if needed.
Outputs (prepared data, event cache, runs, Slurm logs) go to `outputs/` (not tracked).
Test sets stay blinded until a config sets `unblind_test: true`.

## Documentation

- [Configuration guide](docs/configuration.md): configure data preparation, event-model runs, and GP runs.
- [Optical data guide](docs/optical_data.md): raw inputs, role assignments, preparation, and the event cache.
- [Benchmark protocol](docs/benchmark.md): how all models are compared on the same held-out HF voxels.
- [RESUM_FLEX configuration reference](https://github.com/annkasch/RESUM_FLEX/blob/main/docs/configs.md): all model, training, and GP settings.
- [RESUM_FLEX surrogate workflow](https://github.com/annkasch/RESUM_FLEX/blob/main/docs/surrogates.md): training, validation, checkpoints, and GP workflow.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use small synthetic LH5 fixtures, never the real simulations.
