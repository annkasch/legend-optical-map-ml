# Photon store (ungrouped photons)

Homogeneous campaigns (photons uniform over a volume, any shape) are not voxels; the neural
field trains on every photon of them. `photon_store.py` converts such LH5 files once into a
photon store (config `configs/photons.*.yaml`, schema `config.py:PhotonStoreConfig`), and
`photon_loader.PhotonStream` reads it during training. Formats in the module docstrings.

```bash
python -m legend_optical_map_ml.photon_store build configs/photons.genesis_box_x+300+500.yaml --workers 8
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

## Training and evaluation

`configs/photon_field.*.yaml` are RESUM_FLEX stream configs (`core.surrogates stream`,
RESUM_FLEX `docs/stream.md`), trained on GPU with `scripts/slurm/train_photon_field.sbatch`
(`docs/slurm.md`). The field is selected on the store's validation photons and evaluated on
the voxel campaign through the usual provider with `scripts/score_event_model.py`.
