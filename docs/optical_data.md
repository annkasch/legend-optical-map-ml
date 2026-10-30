# LAr optical-map data

How the LEGEND LAr optical-map simulations become prepared datasets for RESUM_FLEX:
LH5 reading, the shared event cache, per-config views, role sources and splits. Model
training, validation metrics and the multi-fidelity GP are documented in RESUM_FLEX
(`docs/surrogates.md`); RESUM_FLEX reads these data through
`legend_optical_map_ml.provider:OpticalDataset`.

For a walkthrough of the YAML settings and how to connect a data config to a training or GP
run config, see [configuration.md](configuration.md).

## Data preparation

Prepare optical data with the library, after setting the input folders in a data config:

```python
from legend_optical_map_ml.config import load_optical_config
from legend_optical_map_ml import prepare_optical_data

config = load_optical_config("configs/config.optical.centered.cnp.data.yaml")
prepared = prepare_optical_data(config)
```

This caches new or changed LH5 files (see "Shared event cache"), assigns the split, fits
normalization and saves a small view plus split metadata; it does not run model training.
Training and GP configs point at the prepared directory (`data_directory`) and, with
`data_config`, regenerate it when the data config or raw files change.
Choose a new output directory for each training run.

## Role sources (event models and GP)

`source.sources` lists folders (`fidelity: lf` by default, or `hf`) and how each folder's
voxels are divided among the roles `cnp_train`, `cnp_validation`, `cnp_test` (event models),
`gp_train`, `gp_unlabeled` (inputs only), `gp_validation` and `gp_test` (GP). Each folder
gives either `fractions` (sum to 1, rounded per folder) or `counts` (file counts; one role may
take `rest`). Roles left out get no files, so an already split folder is a single role at 1.0:

```yaml
source:
  sources:
  - path: /data/lower_cyl/lf
    fractions: {cnp_train: 0.6, cnp_validation: 0.1, cnp_test: 0.0, gp_train: 0.3}
  - path: /data/lf_batch2
    counts: {cnp_train: 5000, cnp_validation: 1000, gp_train: rest}
  - path: /data/lf_val_only
    fractions: {cnp_validation: 1.0}
  - path: /data/hf
    fidelity: hf
    fractions: {gp_train: 1.0}
split:
  seed: 42
  manifest: outputs/my_data/splits/voxel_split.json
```

Splitting keeps whole voxels together and is seeded per folder; the global split fractions
are rejected in this layout. A voxel position shared by different roles is an error, except
between `gp_train` and `gp_unlabeled` (nested multi-fidelity design: HF voxels at positions of
LF inputs). Normalization is fitted on `cnp_train` and applied to every role and fidelity.
`python -m core.surrogates train` reads `cnp_train`, `cnp_validation` (with
`lf_validation: true`) and `cnp_test` (with `test_fidelities: [lf]` and `unblind_test`). Set
`hf_validation: false`; the HF roles are GP levels and evaluation sets (`mfgp` levels name the
role partitions directly, e.g. `gp_train`/`hf`).

Two settings in the training config make large validation sets cheap.
`selection_max_voxels: N` selects checkpoints during training on a seeded subset of at most N
LF validation voxels (seed `validation_seed`); the full validation and test sets are still
evaluated once after training. `evaluation_threads: N` runs evaluation passes with N torch CPU
threads and restores the training thread count afterwards; results are identical. Measured:
2.65 ms per 1000-event voxel on 1 thread, 0.9 ms on 16.

Set `data_config: <data YAML>` in the training config to keep the prepared data in sync.
Each `core.surrogates train` run then compares the data YAML with `config.json`, which only
preparation writes, and the raw files with `source_stamp.json` (path, size and modification
time of every file). If both match, the raw files are not read. Otherwise the run checks file
contents and regenerates the prepared data if anything differs, before training starts.
The data YAML's `output_directory` must be the training `data_directory`. Whether data was
regenerated is recorded in the run's `experiment.json`.

## Unlabeled inputs and context-free inference

Simulations without optical output (vertex and momentum only, e.g. `LF_primaries`) go into
the role `gp_unlabeled`. The event cache stores them with `labeled = False`; reading their
labels or hits raises, and preparation fails if such a file lands in a labeled role. Their
partition loads only through `OpticalView(dir)` (`legend_optical_map_ml.view`.load_inputs(partition, fidelity)`, whose
labels are a zero placeholder for the `StandardBatch` contract and must never be scored.

```bash
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR gp_unlabeled --threads 8
```

writes `RUN/inference/best/gp_unlabeled_lf.{npz,json}`: per voxel the file, nominal
center, normalized theta, event count and `mean_probability` (mean predicted p over the
voxel's events), plus provenance. Only context-free models (mlp, transformer, bdt) are
accepted, and the view normalization must equal the checkpoint's. Results are never
overwritten. Use voxels the model was not trained on, so no labels can be memorized.

## Shared event cache

Raw LH5 files are read once into a cache shared by all data configs: by default
`<output_directory parent>/event_cache`, or `cache_directory` in the data YAML. Per file it
keeps a float32 copy of each event's vertex and momentum (`[n, 6]`), the any-channel label,
and one hit-list row per (event, detector) with the number of hits. Coordinates and other
features are computed when a partition is loaded (about 0.2 µs per event), so changing feature
settings never rewrites the cache.

- `index.npy` has one row per file: folder, name, size, mtime, sha256, voxel centers, event
  and hit counts, and the largest `n_part`. `cache.json` lists the folders and shards.
- Each update reads only new or changed files (by size and mtime). Worker processes, one per
  available CPU, each read a share of the files and write one shard. A changed file goes into
  a new shard and its old row is marked invalid.
- `python -m legend_optical_map_ml.event_cache update DATA.yaml [--workers N]` fills the cache without preparing
  (run it on a compute node for large inputs). `python -m legend_optical_map_ml.event_cache compact CACHE_DIR`
  merges the shards of many small updates and drops invalid rows.
- One writer at a time: updates take a `.lock` directory. If a writer crashed, remove the lock
  by hand.

A prepared dataset is a view on the cache: `view.npy` lists each file's partition, fidelity,
path and sha256, and `view.json` stores the cache directory, data config and normalization.
`load_partition(directory, partition, fidelity)` (in `data.optical_pipeline`) builds the
normalized `StandardBatch` (phi in float32) and returns its provenance: a sha256 over the
selected files' contents and `view.json`. Loading checks that the cache still holds the same
file contents. Normalization is fitted with streaming moments over the training files, so
the training events never need to be in memory at once.

## Multiple raw input folders

A data config can list several folders per
fidelity under `source.directories.lf` and `source.directories.hf`. Each folder
must directly contain `.stp.lh5` files; fidelity comes from the configuration,
so folder names need not be `lf` or `hf`. For example:

```yaml
source:
  directories:
    lf: [/path/to/lf1500_part1, /path/to/lf1500_part2]
    hf: [/path/to/hf_part1, /path/to/hf_part2]
```

The previous `source.directory` form (one root containing `lf/` and `hf/`)
remains supported. Specify one form only. Relative paths are resolved against
the preparation YAML. Repeated references/symlinks to the same physical file
are loaded once. Different files with the same fidelity and basename, and the
same physical file assigned to both fidelities, are rejected as ambiguous.
Splitting and training-only normalization happen after collecting all folders.
Event counts may differ between voxels (see "Unequal event counts per voxel");
mixing primary budgets within LF remains a separate, retired modeling choice.

## Automatic preparation

A training or GP config with `data_config` calls `ensure_prepared_optical_data` before
loading data. The data config's output directory must equal `data_directory`. The helper checks raw-file fingerprints,
configuration, manifest, and the view; a missing or outdated view and normalization
are rebuilt, reading only files missing from the event cache. Unchanged valid inputs are reused. Changes
in source membership, file contents, or configuration automatically update the
manifest. Compatible existing assignments are retained; explicit split policies
take precedence. Invalid inputs and conflicting voxel groups still raise errors.

Preparation matches `stp/optical/evtid` to `vtx/evtid`, making one binary target
per simulated event: one if any configured detector recorded a hit, zero otherwise.
Multiple recorded hits for an event count once. This is an observed simulation
outcome, not a model-assigned label. Current selected inputs all have one primary
per event (`vtx/n_part == 1`). Theta is the filename center; phi contains vertex
offsets from that center and momentum components. Standardization is fitted on
training files only, then applied to both training and validation. Prepared
arrays, normalization, source fingerprints, split metadata and reports are saved.


## Folder names describe simulation statistics

The canonical input layout under
`/global/cfs/cdirs/m2676/users/aschuetz/data/lar_optical_map/by_primaries/` is:

```text
N500/    nonzero/    zero/
N750/    nonzero/    zero/
N1000/   nonzero/    zero/
N1500/   nonzero/    zero/
N5000/   nonzero/    zero/
```

`N` denotes simulated primary count. Each category contains symlinks to the
original `.stp.lh5` files. Counts (nonzero / zero) are 38/47, 157/70, 146/78,
159/65 and 56/9 respectively. The root manifest records every source and excludes
eight failed one-primary simulations. Original source folders are preserved.

Folders have no inherent LF/HF role: `source.directories` assigns that role when
preparing a dataset. To include zero-hit voxels, list the matching `zero` folder too. Preparation accepts unequal
event counts per voxel; this layout does not by itself reintroduce training pooled
across different LF primary counts.
