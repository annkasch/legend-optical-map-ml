# Outputs and configs layout

Run and data folders under `outputs/` are grouped by the data set they were built from; the
configs that make them sit in `configs/<same folder>/` with short names: `data.yaml`,
`photons.yaml`, `benchmark.yaml`, one file per model (`mlp.yaml`, `mlp_bll.yaml`,
`binomial.mlp_lf2.yaml`, `mfgp.log.yaml`, `field_f4.yaml`, `countfield.pt.yaml`). A run config
lives in the folder of its training data, like its run (`tests/test_configs.py` checks both).

```
outputs/
  event_cache/                 shared event cache of all datasets (event-cache.md)
  slurm_logs/                  job logs (slurm.md)
  voxel_sets_<volume>/         one voxel campaign
    data/                      prepared view: the data YAML's output_directory
      views/<name>/            further views of the same data (data-config.md, Views)
    event_learning/<model>/vN/seedN  networks: MLP, CNP, BLL MLP, neural field
    regression/<model>/vN/seedN      GPs: binomial_*, mfgp_*
    countfield/                exports (*_counts.npz, *_photons.npz) and countfield runs
  photon_<volume>/             one homogeneous photon box
    data/                      photon store: the store YAML's `store:` (photon-store.md)
    event_learning/<model>/vN/seedN  networks trained on that photon store
```

- `<volume>` is the volume as in the raw campaign's name, e.g.
  `voxel_sets_x-800-600_y+50+250_z+400+600_5mm`,
  `voxel_sets_x+300+500_y-100+100_z-100+100_5mm`, `photon_x+300+500_y-100+100_z-100+100` and
  `voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm` (Ian's raw folder is named after the box
  centre, `centered_425_375_375`). Datasets are referred to by these names only.
- **A run goes in the folder of the data it was trained on**, not the data it is scored on:
  a neural field trained on the photon store goes under `photon_<volume>/`, even though it is
  scored on the voxel campaign's `gp_validation`.
- Model names omit the dataset; GP models start with their type (`binomial_mlp_lf2_hf`,
  `mfgp_mlp_hf_log`). A different model (another noise model, input or architecture:
  `mfgp_mlp_hf_log_hetnoise`, `cnp_mixup`, `field_f4`) gets its own folder.
- **Versions are subfolders** `vN`: a rerun of the same model with new settings is the next
  version, e.g. `mlp_bll/v1` (LF train) and `mlp_bll/v2` (LF2 + HF train),
  `mfgp_mlp_hf_log/v2` (GP_val split 250/250). A new version never overwrites an old one.
- GP names leave out the input MLP's version; the run's records and config name the MLP run
  (`event_learning/mlp/vN`) whose inference it reads, in the same dataset folder.
- An added evaluation of a finished run (RESUM_FLEX `within_point`) is not a new run: it
  writes into the run's folder, next to its read-out (`within_point.json`, arrays, plot), and
  leaves the run's own files unchanged. Config: `configs/<dataset>/<stage>.<model>.yaml`, e.g.
  `within_point.mlp_bll.yaml`.
- The residual learnability test (RESUM_FLEX `residual`) needs a new directory: it writes to
  `<run>/residual_test/`; config `residual.<model>.yaml`. Likewise the benchmark score of an
  event model (RESUM_FLEX `score`): `<run>/score_<partition>_<fidelity>/`, config
  `score.<model>.yaml`.
- Data YAMLs set `cache_directory: ../outputs/event_cache`; the default (next to
  `output_directory`) would start an empty cache inside the dataset folder.

Runs made before the RESUM_FLEX design point rename (a0469c0, 2026-11-10) no longer load in
RESUM_FLEX; they sit in `outputs/<dataset>/archive_pre_rename/` with their folder structure
and keep their `vN`, so new runs take the next version.
