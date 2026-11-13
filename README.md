# legend-optical-map-ml

Machine-learning surrogates of the LEGEND liquid-argon (LAr) optical map, built on
[RESUM_FLEX](https://github.com/annkasch/RESUM_FLEX). This repository holds everything
specific to the optical-map application:

- `legend_optical_map_ml/`: LH5 reader, shared event cache, prepared views (several per data
  YAML: `views:`), role-based splits, features and reports, and `provider.OpticalDataset`,
  the dataset provider RESUM_FLEX reads the prepared data through.
- `configs/`: data, event-model (CNP, MLP, mixup ablation) and GP configs for Ian Wang's
  centered box (`centered_425_375_375`), plus [LBNL CSV-manifest data](configs/voxel_sets_x-800-600_y+50+250_z+400+600_5mm/data.yaml)
  with the models compared on it (see the [benchmark protocol](docs/benchmark.md)):
  - event models: [MLP](configs/voxel_sets_x-800-600_y+50+250_z+400+600_5mm/mlp.yaml), MLP with a Bayesian
    last layer (`mlp_bll`), neural fields (`field_f1` … `field_f8`, maximum Fourier frequency);
  - GPs: binomial count GP (`binomial`, with the MLP covariate: `binomial.mlp`,
    `binomial.mlp_lf2`), log-space MFGP on the MLP mean (`mfgp.log`) or raw LF (`mfgp.raw_lf.log`);
  - [countfield](https://github.com/annkasch/countfield) runs (`configs/voxel_sets_x-800-600_y+50+250_z+400+600_5mm/countfield*.yaml`):
    neural process and Point Transformer variants.
- `scripts/`: `export_counts.py` / `export_photons.py` (countfield inputs).
- `scripts/slurm/`: Perlmutter job scripts for individual stages and dependent pipelines.
- `docs/`: one short file per topic, listed in the routing table of [CLAUDE.md](CLAUDE.md).

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
.venv/bin/python -m core.surrogates train configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mlp.yaml
# 2. Mean predicted p for the unlabeled LF voxels (LF_primaries)
.venv/bin/python -m core.surrogates predict outputs/optical_resum/centered_mlp_v1/seed0/checkpoints/best \
    outputs/optical_centered_cnp_v0 gp_unlabeled --threads 8
# 3. Two-level GP: MLP mean -> observed HF fraction
.venv/bin/python -m core.surrogates mfgp configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mfgp.yaml
```

On Perlmutter, create the Slurm log directory and submit each step from the repository root:

```bash
mkdir -p outputs/slurm_logs
sbatch -q regular -t 01:00:00 scripts/slurm/train_surrogate.sbatch configs/CONFIG.yaml [train|mfgp]
```

To submit training, mean prediction, and GP fitting as dependent jobs, use:

```bash
scripts/slurm/submit_surrogate_pipeline.sh \
  configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mlp.yaml \
  configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mfgp.yaml \
  -q regular -t 01:00:00
```

See [the Slurm guide](docs/slurm.md) for details.

The batch script uses this checkout's `.venv/bin/python` by default; set `RESUM_PYTHON` to
the executable path of another interpreter if needed.
Outputs (prepared data, event cache, runs, Slurm logs) go to `outputs/` (not tracked).
Test sets stay blinded until a config sets `unblind_test: true`.

## Documentation

The routing table in [CLAUDE.md](CLAUDE.md) lists one file per topic and when to read it: design, datasets,
data configs, preparation, event cache, run configs, photon store, benchmark, countfield
exports and Slurm. Toolkit topics (models, training, GPs, metrics): the
[RESUM_FLEX docs index](https://github.com/annkasch/RESUM_FLEX/blob/main/docs/index.md).

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests use small synthetic LH5 fixtures, never the real simulations.
