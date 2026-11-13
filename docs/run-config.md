# Run configs for optical data

The optical-specific parts of training and GP configs. All model, training and GP settings:
RESUM_FLEX `docs/configs.md` (models `docs/models.md`, Bayesian last layer
`docs/last-layer.md`, GP `docs/gp.md`). Relative paths resolve from the run YAML.

## Data and destinations (every run config)

| Setting | Meaning |
|---|---|
| `data_provider` | `legend_optical_map_ml.provider:OpticalDataset` |
| `data_config` | Data YAML checked and prepared before the run (`docs/preparation.md`) |
| `data_directory` | Prepared view: the data YAML's `output_directory` or one of its `views/<name>` (`docs/data-config.md`, Views) |
| `partitions` | Run splits (train, validation, test) → data-config roles |
| `train_sets` | Several role/fidelity sets for training (rate models: LF2 + HF, `docs/benchmark.md`) |
| `output_directory` | New, empty path per run; bump `_vN` instead of overwriting |

## Event-model training

Examples: `configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mlp.yaml`, `configs/voxel_sets_x-800-600_y+50+250_z+400+600_5mm/mlp.yaml`
(`mlp_bll`, `field_f*` for rate models).

- `train` reads `cnp_train`, `cnp_validation` (`lf_validation: true`) and `cnp_test`
  (`test_fidelities: [lf]`, needs `unblind_test`). Set `hf_validation: false`: HF roles are GP
  levels and benchmark sets.
- Checkpoints are selected on LF validation; `selection: bernoulli_log_loss` for natural
  sampling (`average_precision` only for the mixup ablation).
- Large validation sets: `selection_max_design_points: N` selects on a seeded subset (`validation_seed`;
  the full sets are still evaluated once at the end); `evaluation_threads: N` (same results;
  2.65 ms per 1000-event voxel on 1 thread, 0.9 ms on 16).
- LBNL lists have no CNP test set; those runs evaluate train and validation only.
- Rate models (`train_sets`: LF2 + HF, or `mlp_bll.lf1`) set `partitions.test: gp_test`:
  their test set is the blinded GP test, so the `score` stage refuses it without
  `unblind_test` (support models keep `cnp_test`). `test_fidelities: []` keeps training from
  evaluating it.
- Every training config sets `residual_test: {}`: the residual learnability test runs in
  the read-out (off by default in RESUM_FLEX; needs the `lgbm` extra). Runs trained without
  it get it afterwards with `residual.<model>.yaml` (`core.surrogates residual`).
- Another input representation (e.g. the absolute emission point): a named view in the data
  YAML, selected by `data_directory`, not a model-side transform. An event-only view
  (`theta.model_input: false`) suits `mlp` and `mlp_bll`, not the field model (needs θ).
  The `score` stage (`docs/benchmark.md`) scores it on the run's view; the provider moves
  donor photons into the scored voxel (`transfer_events`: absolute positions shifted by the
  centre difference, offsets unchanged). A support model whose
  `predict` output feeds a GP needs a view with θ for now (`docs/data-config.md`).

## GP runs (`mfgp`, `binomial`)

Example: `configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mfgp.yaml`, run after `predict` wrote the level-0 file.

- `levels` from lowest to highest fidelity; the last is predicted. Typical: `model_mean` of
  the MLP on `gp_unlabeled/lf`, then `observed_fraction` of `gp_train/hf`. Variants:
  `mfgp.log` (log space), `mfgp.raw_lf.log` (simulated LF fraction as level 0), `*.hetnoise`
  (per-voxel binomial noise), `binomial` (count GP; `.mlp`, `.mlp_lf2`: MLP covariate).
- `inference` points at the `predict` output; `evaluation` names held-out roles. Keep test
  roles out of `levels` and blinded until choices are final.
- GP fitting time grows steeply with the number of voxels: run on a compute node
  (`docs/slurm.md`).
