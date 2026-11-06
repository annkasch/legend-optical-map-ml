# Data preparation

Preparation turns a data config into a prepared view (`pipeline.py`). It caches new or
changed LH5 files (`docs/event-cache.md`), assigns roles, fits normalization and writes the
view and split metadata; it never trains.

```python
from legend_optical_map_ml import prepare_optical_data
from legend_optical_map_ml.config import load_optical_config

prepared = prepare_optical_data(load_optical_config("configs/config.optical.centered.cnp.data.yaml"))
```

Usually not needed by hand: a run config with `data_config` calls
`ensure_prepared_optical_data` before loading data.

## Keeping prepared data in sync

The data YAML's `output_directory` must equal the run's `data_directory`. Each run compares
the data YAML with `config.json` (written only by preparation) and the raw files with
`source_stamp.json` (path, size, mtime per file). If both match, no raw file is read.
Otherwise file contents are checked and the view is rebuilt if anything differs, reading only
files missing from the cache. Compatible existing role assignments are kept; explicit split
policies take precedence; invalid inputs and conflicting voxel groups raise. Whether data were
regenerated is recorded in the run's `experiment.json`.

## What preparation computes

- Label: `stp/optical/evtid` is matched to `vtx/evtid`; an event is 1 if any configured
  detector recorded a hit (multiple hits count once). Inputs have one primary per event
  (`vtx/n_part == 1`).
- θ = voxel centre; φ = vertex offset from the centre and momentum components (per config).
- Normalization from streaming moments over the training files (`cnp_train`), applied to every
  role and fidelity.
- The view (`view.npy`, `view.json`) lists each file's partition, fidelity, path and sha256.
  `load_partition(directory, partition, fidelity)` builds the normalized `StandardBatch`
  (φ float32) with a provenance sha256 over the files and `view.json`, and checks that the
  cache still holds the same file contents.

## Unlabeled inputs and context-free inference

Simulations without optical output (vertex and momentum only, e.g. `LF_primaries`) go into
`gp_unlabeled`. The cache marks them `labeled = False`; reading their labels or hits raises,
and preparation fails if one lands in a labeled role. They load only through
`OpticalView(dir).load_inputs(partition, fidelity)`; the labels there are zero placeholders
and must never be scored.

```bash
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR gp_unlabeled --threads 8
```

writes `RUN/inference/best/gp_unlabeled_lf.{npz,json}`: per voxel the file, centre,
normalized θ, event count and `mean_probability`, plus provenance. Only context-free models
(mlp, transformer, bdt); results are never overwritten. Predict voxels the model was not
trained on.
