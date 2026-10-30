# Configuration guide

This guide explains the two YAML files that control an optical-map training run:

1. A **data config** describes raw LH5 inputs and how to turn them into prepared data.
2. A **run config** describes the model, training, validation, and output for one run.

The example configs target the `centered_425_375_375` data layout. Copy them for a new
dataset or experiment and change the paths and settings described below. The YAML schemas
reject unknown keys, so a spelling mistake is reported instead of silently ignored.
Commands in this guide assume the environment setup in the [README](../README.md#setup),
which installs RESUM_FLEX and this package into this checkout's `.venv`.

## Which command should I run?

The run config points to its data config using `data_config`. With that field set, the
training command checks whether the prepared data are current and prepares or updates them
before training starts. A separate preparation step is optional; use it when you want to
build data without starting a model run.

From this repository's root, the example MLP run is:

```bash
.venv/bin/python -m core.surrogates train configs/config.optical.centered.mlp.yaml
```

For the LBNL voxel-set example, use:

```bash
.venv/bin/python -m core.surrogates train configs/config.optical.lbnl_secondment.mlp.yaml
```

Its run config points to the CSV-manifest data config, so the first run prepares the listed
files automatically. The supplied lists do not include a separate CNP test set; this
baseline evaluates train and validation only.

To prepare data on its own, see [optical_data.md](optical_data.md#data-preparation). Both
commands use the same data config.

## Paths and outputs

Relative paths in a data config resolve from that data YAML's directory. Relative paths in
a training or GP run config resolve from that run YAML's directory. They do not resolve from
the shell's current directory. For example, in `configs/config.optical.centered.mlp.yaml`,
`data_directory: ../outputs/optical_centered_cnp_v0` and
`data_config: config.optical.centered.cnp.data.yaml` both point to paths under this
repository's `outputs/` and `configs/` directories respectively.

The data config's `output_directory` must resolve to the same directory as the run config's
`data_directory`. The provided configs already agree. Training writes checkpoints and
metrics to `output_directory`; choose an empty, new directory for each run. Prepared views,
the shared event cache, runs, and Slurm logs are under `outputs/` and are not tracked by Git.

## Data config

The example is [`config.optical.centered.cnp.data.yaml`](../configs/config.optical.centered.cnp.data.yaml).
Its main settings are:

| Setting | Meaning and how to choose it |
|---|---|
| `source` | Where raw LH5 files live and which data role each file belongs to. This is the part most users need to adapt. |
| `output_directory` | Where the prepared view and its metadata are written. Keep it aligned with the run config's `data_directory`. |
| `cache_directory` | Optional shared cache location. If omitted, the cache is placed next to `output_directory` in an `event_cache/` directory. |
| `theta` | How the voxel's nominal position becomes the model's spatial input. |
| `phi` | Which per-event inputs are used and how vertex and momentum are represented. |
| `target` | Which optical-hit outcome is predicted. |
| `split` | Seed and manifest for reproducible voxel assignment. In a role-based `source.sources` config, only `seed` and `manifest` apply. |
| `normalization` | How input features are scaled. Statistics are fitted on training data and reused for all roles. |

### Choosing raw sources and roles

The `source` block supports four layouts: `directory` for a single LF/HF tree,
`directories` for lists of folders by fidelity (`lf` or `hf`), `sources` for explicit
per-folder role assignment, or `manifests` for CSV file lists. Use exactly one layout. The
provided centered dataset uses `sources`, for example:

```yaml
source:
  sources:
  - path: /path/to/LF_sim/lf
    fractions: {cnp_train: 1.0}
  - path: /path/to/validation/lf/lf
    fractions: {cnp_validation: 0.5, cnp_test: 0.5}
  - path: /path/to/LF_primaries/lf
    fractions: {gp_unlabeled: 1.0}
  - path: /path/to/HF_sim/hf
    fidelity: hf
    fractions: {gp_train: 1.0}
```

Each path names one folder. `fidelity` defaults to `lf`; set it to `hf` for high-fidelity
files. Give each source either `fractions` or `counts`, not both. Fractions must sum to 1
for that folder. Counts assign a fixed number of files to each role; one role may use
`rest` to receive all remaining files. Role names describe how the files will be used:

- `cnp_train`, `cnp_validation`, `cnp_test`: labeled LF event-model data.
- `gp_train`: observed data used to fit the GP, commonly HF data.
- `gp_unlabeled`: inputs-only LF voxels where an event model supplies a prediction for the
  GP. These files do not need optical-hit data.
- `gp_validation`, `gp_test`: held-out data for GP model selection and final evaluation.

When the files already live together but CSVs define the splits, use `manifests`. Each CSV
is assigned one role and lists its files in an `stp_file` column; paths in that column are
relative to the CSV's directory. If a CSV has a `tier` column (`lf` or `hf`), fidelity is
read from it. Otherwise set `fidelity` in the manifest entry. If `x_mm`, `y_mm`, and `z_mm`
are present, they are used as the exact voxel center (converted from mm to m); otherwise the
center is parsed from the filename.

```yaml
source:
  manifests:
  - path: /data/voxel_sets/voxels_LF_train.csv
    role: cnp_train
  - path: /data/voxel_sets/voxels_CNP_val.csv
    role: cnp_validation
  - path: /data/voxel_sets/voxels_LF2_train.csv
    role: gp_unlabeled
  - path: /data/voxel_sets/voxels_HF_train.csv
    role: gp_train
  - path: /data/voxel_sets/voxels_GP_val.csv
    role: gp_validation
```

Each manifest file may appear only once, and a raw file may not be listed by multiple
manifests. The `voxels_all.csv` union should not be included alongside its component lists.
Manifest assignments are used directly; no random repartitioning is performed. CSV content
and listed file size/modification times are checked for updates before each run. See
[`config.optical.lbnl_secondment.data.yaml`](../configs/config.optical.lbnl_secondment.data.yaml)
for the concrete LBNL dataset mapping.

Do not put the same voxel in unrelated roles; the preparer checks for position overlap to
prevent data leakage. `gp_train` and `gp_unlabeled` may overlap by design for a nested GP
experiment. `cnp_validation` and `gp_validation` may also share positions because both are
held out from training. `exclude_files` lists raw files to omit, relative to their source
folder; use it for known bad or incomplete files.

### Features, target, and split

- `theta.coordinates` selects Cartesian or cylindrical voxel coordinates. With
  `center_source: filename`, position comes from the nominal voxel encoded in each filename;
  `observed_midrange` derives the center from the events. Cylindrical settings control angle
  encoding and whether azimuth is included.
- `phi.vertex` controls whether event vertex information is represented as an offset from
  the voxel center, as absolute coordinates, or omitted. `phi.momentum` chooses Cartesian
  components, direction plus magnitude, or omission.
- `target.kind: any_channel` labels an event positive when it has a hit in any channel.
  Choose `single_channel` and set `detector_id` to predict one detector channel instead.
- `split.seed` makes automatic splits reproducible. `split.manifest` stores the assignments
  so later preparation keeps the same voxels in each role.
- `normalization.method` is `standard`, `minmax`, or `none`. The usual `standard` setting
  standardizes features using the training role only. Avoid changing normalization between
  training and inference; the checkpoint checks that they match.
- `normalization.fit_fidelities` optionally restricts which fidelities contribute to the
  normalization statistics. If omitted, all eligible training-role data are used.

With explicit `source.sources`, the role assignment is the split, so leave the other split
fraction fields at their defaults. The centered example sets `split.seed` and `manifest`.

## Training run config

The example is [`config.optical.centered.mlp.yaml`](../configs/config.optical.centered.mlp.yaml).
The `cnp.yaml` and `cnp.mixup.yaml` examples use the same data but change the model or
sampling strategy.

### Data and destinations

| Setting | Meaning |
|---|---|
| `data_directory` | Prepared dataset directory; must match the data config's `output_directory`. |
| `data_provider` | Adapter that lets RESUM_FLEX load this package's data. For optical data, keep `legend_optical_map_ml.provider:OpticalDataset`. |
| `data_config` | Data YAML to check and prepare before this run. Keep it set when the run should automatically prepare/update data. |
| `partitions` | Maps the run's train, validation, and test splits to roles in the data config. |
| `output_directory` | Destination for this run's checkpoints and reports; use a fresh path. |

### Model and training

- `model.kind` chooses `mlp`, `cnp`, `transformer`, or `bdt`. An MLP is context-free;
  a CNP also receives context events. The fields under `architecture` configure that model.
- `training.backend` is `neural` for MLP/CNP/transformer or `bdt` for a boosted decision
  tree. Keep it consistent with `model.kind`.
- `n_passes` sets the run length in passes over the training targets. Alternatively set
  `n_steps`; use one or the other. `eval_every` controls how often validation selects a
  checkpoint, and `learning_rate` controls the neural optimizer step size.
- `batch_size` is the number of voxels sampled per training step. `n_events` is the number
  of events sampled from each voxel for that step. These are event-model training settings,
  not the total number of files in the data config.
- For a CNP, `context` controls how many events are shown to the model. `size: random` draws
  a context count from `n_min` up to `max_fraction * n_events`; `size: fixed` always uses
  `n`. With `targets: disjoint`, context events are not scored as targets. `joint` includes
  context events in the loss and is only supported with natural sampling.
- `sampling.strategy: natural` preserves the observed positive-event rate and gives
  probability-calibrated outputs. `class_aware_mixup` is an imbalance/mixup experiment, not
  the calibrated baseline. `weighting` and `objective` provide additional training choices;
  leave them at the example defaults unless intentionally running an ablation. The config
  validator enforces compatible combinations.
- `seed` controls training randomness; `device` selects `cpu` or `cuda`. `focal_gamma: 0`
  means standard Bernoulli cross-entropy.

### Checkpoint selection and evaluation

- `selection` chooses the validation score for the best checkpoint. Use
  `bernoulli_log_loss` for natural probability predictions; the mixup example uses
  `average_precision` because its outputs are ranking scores. `voxel_rate_mae` is another
  supported metric.
- `lf_validation` and `hf_validation` enable those validation evaluations. `partitions`
  determines which prepared role is loaded for each split.
- `test_fidelities` lists test datasets to evaluate. Test data remain blinded unless
  `unblind_test: true`; wait until modeling choices are final before unblinding.
- `validation_context_events`, `validation_context_sweep`, and `validation_seed` configure
  context-model readouts after training. A sweep compares several context sizes on the same
  targets; it does not change training.
- `selection_max_voxels` limits the validation subset used for checkpoint selection.
  `evaluation_threads` sets the CPU thread count for evaluation passes.

For the full list of model, training, and GP settings (including multi-stage training and
fine-tuning), see the [RESUM_FLEX configuration reference](https://github.com/annkasch/RESUM_FLEX/blob/main/docs/configs.md).

## GP config

The centered GP example is [`config.optical.centered.mfgp.yaml`](../configs/config.optical.centered.mfgp.yaml).
It is run after the event-model prediction file exists:

```bash
.venv/bin/python -m core.surrogates mfgp configs/config.optical.centered.mfgp.yaml
```

`levels` lists GP inputs from lowest to highest fidelity; the final level is the quantity
being predicted. In the example, `model_mean` reads the MLP's mean probability for
`gp_unlabeled/lf`, and `observed_fraction` reads the simulated hit fraction for
`gp_train/hf`. `inference` points to the saved predictions from the earlier `predict`
command. `evaluation` names held-out validation roles. Keep test roles out of the levels and
blinded in `evaluation` until choices are final.

`kernel`, `n_restarts`, and `seed` configure GP fitting. `output_transform: identity`
models rates directly; `log` models log-rates and uses the pseudocount/floor settings.
`data_directory`, `data_provider`, and `data_config` have the same meanings as for training.
The GP configuration reference linked above documents projections and all remaining GP
options.
