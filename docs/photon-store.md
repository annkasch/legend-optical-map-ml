# Photon store (ungrouped photons)

Homogeneous campaigns (photons uniform over a volume, any shape) are not voxels; the neural
field trains on every photon of them. `photon_store.py` converts such LH5 files once into a
photon store (config `configs/photon_<volume>/photons.yaml`, schema `config.py:PhotonStoreConfig`), and
`photon_loader.PhotonStream` reads it during training. Formats in the module docstrings.

```bash
python -m legend_optical_map_ml.photon_store build configs/photon_x+300+500_y-100+100_z-100+100/photons.yaml --workers 8
python -m legend_optical_map_ml.photon_store info outputs/photon_x+300+500_y-100+100_z-100+100/data
# Large campaigns: plan once, convert tasks in a Slurm array, then finalize
python -m legend_optical_map_ml.photon_store plan CONFIG.yaml
python -m legend_optical_map_ml.photon_store run CONFIG.yaml --task-index $SLURM_ARRAY_TASK_ID --task-count N
python -m legend_optical_map_ml.photon_store finalize CONFIG.yaml
```

- **Records**: 28 bytes per photon, x, y, z (m) and px, py, pz (MeV) as float32, exactly as
  remage writes them, plus the any-channel label. Per-detector hits are kept in a sparse table
  per shard. No photon is dropped.
- **Shards** of about `shard_photons` photons (default 10⁸, ≈ 2.8 GB). The photons inside a
  shard are permuted once when it is written, so any contiguous block is a random sample.
  Each shard summary records the code that wrote it (`outputs.md`, Code provenance).
  Shard folders are striped over 8 Lustre targets when `lfs` exists.
- **Splits** per source file, from a seeded hash of its key (`<source name>/<relative path>`),
  so adding files never moves existing ones. A changed source file is an error.
  `shard_photons`, `split` and `seed` are fixed once the store exists; sources may be added.
- **Normalization**: shards store raw values. Each shard's moments are computed while
  converting; the train shards' moments are merged exactly for the normalization, without a
  pass over the data. Each trained model stores the normalization it used.
- **Loader**: per pass, every shard is cut into blocks with a new random boundary offset, all
  blocks are permuted, consecutive blocks form a buffer that is permuted on the device, and
  the buffers are cut into batches. Every photon is used once per pass, the grouping changes
  every pass, and there is no global permutation index. Background threads read whole blocks
  into pinned host buffers; copies to the GPU run on a side stream. `in_memory: true` keeps
  the rank's shards in RAM after the first read and uses small blocks (almost a full shuffle,
  no I/O). Several ranks get disjoint blocks and the same number of batches. A pass is
  deterministic, so training can resume at a batch index.

## Input transform

`event_source_options.transform` (schema `config.py:StreamTransformConfig`) transforms the
records for training: the same `cylindrical` and `phi` blocks as a data YAML (`theta.cylindrical`,
`phi`; `docs/data-config.md`), with an absolute or omitted vertex, e.g.

```yaml
event_source_options:
  transform:
    cylindrical: {rotational_symmetry: 6}
    phi:
      vertex: {representation: absolute, coordinates: cylindrical}
      momentum: {representation: components, coordinates: cylindrical}
```

- `batches()` transforms each buffer on the device after the copy (torch, `features.py:
  stream_features_torch`); `load()` uses the numpy reference (`stream_features`), which the
  data views also use. The store stays raw; records become [M, W + 1], label bits last.
- `normalization()` then comes from the transformed columns of a fixed sample, the first
  photons of every train shard (shards are permuted when written), at most
  `normalization_photons` (default 10⁷); it lists the column names and the transform.
- The view at the run's `data_directory` must build the same phi columns (same `cylindrical`
  and `phi`): RESUM_FLEX maps the view's phi columns one by one into the stream units without
  checking their meaning. `tests/test_configs.py` checks every stream config in `configs/`.

## Training and evaluation

`configs/photon_<volume>/field_*.yaml` are RESUM_FLEX stream configs (`core.surrogates stream`,
RESUM_FLEX `docs/stream.md`), trained on GPU with `scripts/slurm/train_photon_field.sbatch`
(`docs/slurm.md`). The field is selected on the store's validation photons and evaluated on
the voxel campaign with the RESUM_FLEX `score` stage (`data_directory`: the voxel view,
`data_provider`: `OpticalDataset`; `docs/benchmark.md`).

A stream model sees the view's φ columns as they are (RESUM_FLEX maps each into its stream
units), so the view's φ must be the stream records: absolute x, y, z, px, py, pz (view
`theta_absolute`), or the view with the loader's `transform` (above). A stream model never
uses θ; `dim_theta` must match the view: `null` on an event-only view (`theta.model_input:
false`, RESUM_FLEX after 2aa0677), or the view's θ width on a view with θ (the stream
configs here: `dim_theta: 3`, view `theta_absolute`), where θ is passed and ignored. `tests/test_stream_evaluation.py` trains a
stream model with a transform and checks its predictions on such a view.
