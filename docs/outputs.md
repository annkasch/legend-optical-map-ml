# Outputs layout

Run and data folders under `outputs/` are grouped by the data set they were built from.

```
outputs/
  event_cache/                 shared event cache of all datasets (event-cache.md)
  slurm_logs/                  job logs (slurm.md)
  voxel_sets_<volume>/         one voxel campaign
    data/                      prepared view: the data YAML's output_directory
    event_learning/<run>/seedN networks: MLP, CNP, BLL MLP, neural field
    regression/<run>/seedN     GPs: binomial_*, mfgp_*
  photon_<volume>/             one homogeneous photon box
    data/                      photon store: the store YAML's `store:` (photon-store.md)
    event_learning/<run>/seedN networks trained on that photon store
```

- `<volume>` is the volume as in the raw campaign's name, e.g.
  `voxel_sets_x-800-600_y+50+250_z+400+600_5mm` (lbnl_secondment),
  `voxel_sets_x+300+500_y-100+100_z-100+100_5mm` and `photon_x+300+500_y-100+100_z-100+100`
  (genesis), `voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm` (centered; Ian's folder is
  named after the box centre, `centered_425_375_375`).
- **A run goes in the folder of the data it was trained on**, not the data it is scored on:
  a neural field trained on the photon store goes under `photon_<volume>/`, even though it is
  scored on the voxel campaign's `gp_validation`.
- Run names omit the dataset; GP runs start with their type (`binomial_mlp_lf2_hf`,
  `mfgp_mlp_v2_hf_log`). A GP that uses an MLP's inference reads it from the `event_learning/`
  run of the same folder.
- Data YAMLs set `cache_directory: ../outputs/event_cache`; the default (next to
  `output_directory`) would start an empty cache inside the dataset folder.

All runs made before this layout were moved here on 2026-11-07; their JSON records (paths,
view provenance) were rewritten to the new folders, including the centered runs' older
`RESUM_FLEX/outputs/...` paths from before the repository split. Runs whose recorded view hash
was already stale before the move still fail the residual stage's provenance check: lbnl
`mlp_v1`, `mlp_v2`, `mfgp_mlp_v2_hf_log*` (made before the lbnl view was
rebuilt on 2026-11-02) and every centered run (made from views built in RESUM_FLEX).
