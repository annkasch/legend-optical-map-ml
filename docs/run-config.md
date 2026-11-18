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
  sampling (`average_precision` only for the mixup ablation). It weighs the voxel rate and the
  within-voxel separation alike: per voxel, the summed log loss = count NLL − within-voxel gain
  + a term of the data only.
- MLP + BLL (`mlp_bll*.yaml`, since 2026-11-16): `lr_schedule: {kind: cosine, final_fraction:
  0.01}` and `n_passes: 20`; with a constant rate the late checkpoints swung (count NLL sd
  0.05, ratio of means 0.5-1.3). Seeds: one config (`seed0`), `--run-seed 1`, `2` for the others
  (`docs/slurm.md`). Needs RESUM_FLEX ≥ c8d2c22.
- Hyperparameter sweep of MLP + BLL with input (b) θ + raw vertex + momentum (2026-11-16):
  `mlp_bll.theta_absolute.<variant>.yaml` changes one field of `mlp_bll.theta_absolute.yaml`
  (v2): `w128`/`w512`, `d2`/`d5` (hidden_dims), `lr3e-4`/`lr3e-3`, `b8`/`b32`, `ev250`/`ev1500`,
  `do10` (dropout 0.1); `n_passes` stays 20, so every variant sees the same photons. `n_events`
  can't exceed the fewest photons of a training voxel (1500 LF). Each has
  `score.mlp_bll.theta_absolute.<variant>.yaml`; `tests/test_sweep_configs.py` checks the
  one-field rule. Combinations (box only, 2026-11-17) join their parts with `_`:
  `lr3e-3_b8`, `lr3e-3_ev250`, `lr3e-3_d5`, `lr3e-3_w512` change the learning rate and one more
  field; they and `lr3e-3` run with seeds 0-2 (`--run-seed`).
- CNP + BLL with input (b) (`cnp_bll.theta_absolute.yaml`, box only, 2026-11-16): the MLP + BLL (b)
  v2 config with a CNP encoder (latent 64, [128, 128], as the earlier `cnp.yaml`) and the CNP
  context keys. Compared on LF validation only, where the context is the voxel's own first
  labelled photons; HF benchmark voxels have none, so it has no score config.
  `cnp_bll.mixup.theta_absolute.yaml` (v1) trains on class-aware mixup targets only
  (`objective: single`, `mix_context: false`, as the earlier `cnp.mixup.yaml`): its outputs are
  mixup scores, not probabilities, so it selects on `average_precision` and only its ranking
  metrics count. `cnp_bll.mixup.recalibrated.theta_absolute.yaml` (v2) starts from v1's best
  checkpoint (`initial_checkpoint`) and refits only the output layer on real photons (one stage,
  `trainable: output_layer`, natural sampling, `bernoulli_log_loss`), so all its metrics compare
  with the plain CNP. Mixup as a second loss term (`real_plus_mixup`, weight 0.001) gave nothing
  over the plain CNP (2026-11-16) and was dropped.
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
  YAML, selected by `data_directory`, not a model-side transform: RESUM_FLEX never adds θ
  to φ (since 2aa0677), so a model on the emission point (e.g. the field, `encoding.columns:
  [0, 1, 2]`) trains on a view with an absolute vertex. An event-only view
  (`theta.model_input: false`) suits `mlp`, `mlp_bll` and the field.
  The `score` stage (`docs/benchmark.md`) scores it on the run's view; the provider moves
  donor photons into the scored voxel (`transfer_events`: absolute positions shifted by the
  centre difference, cylindrical features re-encoded at the new azimuth, offsets unchanged). As a support model its `predict` output feeds a GP
  like any other (`docs/data-config.md`).

## GP runs (`mfgp`, `binomial`)

Example: `configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mfgp.yaml`, run after `predict` wrote the level-0 file.

- `levels` from lowest to highest fidelity; the last is predicted. Typical: `model_mean` of
  the MLP on `gp_unlabeled/lf`, then `observed_fraction` of `gp_train/hf`. Variants:
  `mfgp.log` (log space), `mfgp.raw_lf.log` (simulated LF fraction as level 0), `*.hetnoise`
  (per-voxel binomial noise), `binomial` (count GP; `.mlp`, `.mlp_lf2`: MLP covariate).
- MFGP without an event model on both voxel datasets (2026-11-17): `mfgp.raw_lf.log.yaml`
  (`learned` noise) and `mfgp.raw_lf.log.hetnoise.yaml` (`observation_noise: binomial`), levels
  = the MLP + BLL `train_sets` (simulated `gp_unlabeled/lf` → `gp_train/hf`), scored on
  `gp_validation/hf`; `tests/test_configs.py` checks the pair and the data. x-800: v2 (v1 before
  the design point rename).
- `inference` points at the `predict` output; `evaluation` names held-out roles. Keep test
  roles out of `levels` and blinded until choices are final.
- GP fitting time grows steeply with the number of voxels: run on a compute node
  (`docs/slurm.md`).
