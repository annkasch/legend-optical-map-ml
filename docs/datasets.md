# Datasets

All 5 mm voxels, LF = 1500 and HF = 5000 photons (one primary each) per voxel unless noted.
Config prefix in brackets. Check sizes and states against the data before relying on them.

## Ian Wang's centered box [`centered`]

`/global/cfs/projectdirs/m2676/users/iwang/sim_output/centered_425_375_375`, box x 0.365-0.485,
y 0.315-0.435, z 0.2975-0.4525 m, 25×25×32 grid. File names
`run{i}_v{voxel}_x_y_z.stp.lh5`. Data config: `config.optical.centered.cnp.data.yaml`.

| Folder | Voxels | Role |
|---|---|---|
| `LF_sim/lf` | 2000 | `cnp_train` |
| `validation/lf/lf` | 500 | `cnp_validation` / `cnp_test` 50/50 |
| `LF_primaries/lf` | 2000 | `gp_unlabeled` (inputs only, contains the 200 HF positions) |
| `HF_sim/hf` | 200 | `gp_train` |
| `validation/hf/hf` | 500 | `gp_validation` / `gp_test` 50/50 |

- Crashed or truncated files are listed in `exclude_files` of the data config (file survey,
  2026-10-22: `outputs/iwang_centered_survey/per_file.csv`).
- 7 LF_sim voxels near x≈0.425, y≈0.37 are only partly filled (they overlap a Ge detector);
  they stay as normal voxels.
- Ian's reboost HF maps (`iwang/output_maps_new`) store the same per-voxel hit ratio as the
  raw outputs (checked 2026-11-02, `analysis/2026-11-02_hf_maps`); use the raw outputs. Only
  per-channel maps would add information (a different target).

## LBNL secondment [`lbnl_secondment`]

Sofia Calgaro's voxel sets
`/global/cfs/cdirs/m2676/users/calgaro/lbnl_secondment/voxel_sets_x-800-600_y+50+250_z+400+600_5mm`,
CSV manifests per role (`voxels_{LF_train,CNP_val,LF2_train,HF_train,GP_val}.csv`). Data
config: `config.optical.lbnl_secondment.data.yaml`. About 6400 LF2 / 640 HF train voxels;
GP_val (500) is split seeded 50/50 into `gp_validation` / `gp_test`.

- 11 GP_val voxels share positions with CNP_val (lists sampled independently). Accepted for
  this dataset (`SHARED_VALIDATION_ROLES` includes `gp_test`); new campaigns must sample
  validation and test disjoint instead.
- Its reboost maps (`.../voxel_sets_x+600+800_y-250-50_z+400+600_5mm/maps`) are empty: y
  binning typo (-0.5 instead of -0.05 m). Not needed: the raw outputs are used.
- Results so far: `analysis/2026-11-03_lbnl_model_comparison/README.md`; benchmark config
  `configs/benchmark.lbnl_secondment.yaml`.

## Own genesis-remage campaign [`genesis_x+300+500`]

On `$PSCRATCH` (`/pscratch/sd/a/aschuetz/genesis-remage/`; purge policy, data can vanish):

- Voxel sets `voxel_sets_x+300+500_y-100+100_z-100+100_5mm`, same sampling and roles as LBNL.
  Data config: `config.optical.genesis_x+300+500_5mm.data.yaml`.
- Homogeneous box `box_x+300+500_y-100+100_z-100+100` (32 chunks of 400k isotropic photons,
  9.68 ± 0.22 eV) → photon store `/pscratch/sd/a/aschuetz/photon_stores/genesis_box_x+300+500`
  (`configs/photons.genesis_box_x+300+500.yaml`, `docs/photon-store.md`).

## Older by-primaries layout

`/global/cfs/cdirs/m2676/users/aschuetz/data/lar_optical_map/by_primaries/N{500,750,1000,1500,5000}/{nonzero,zero}/`:
symlinks to the original `.stp.lh5` files, by simulated primary count (nonzero / zero:
38/47, 157/70, 146/78, 159/65, 56/9). The root manifest excludes eight failed one-primary
runs. Folders have no LF/HF role; `source.directories` assigns it. List the `zero` folder too
to include zero-hit voxels.
