# legend-optical-map-ml — agent brief

Application repository: surrogates of the LEGEND liquid-argon (LAr) optical map on top of
RESUM_FLEX (`../RESUM_FLEX`, editable install in `.venv`, own `CLAUDE.md`). Data handling,
configs, job scripts and the benchmark live here; models, training, GPs and metrics are
RESUM_FLEX's: change toolkit code there, application code here. `../countfield` (own repo)
holds the neural-process / Point Transformer models.

Start of a task: read `.claude/status.md` if it exists (local, open items only). Then load
only what the task needs, using the table below; `docs/index.md` lists every doc.

| Task | Read | Code |
|---|---|---|
| Change library code | `docs/design.md` | the module it names |
| A dataset: paths, roles, bad files | `docs/datasets.md` | `configs/<dataset>/data.yaml` |
| Write or change a data YAML | `docs/data-config.md` | `config.py`, `split.py` |
| Preparation, sync, unlabeled inputs | `docs/preparation.md` | `pipeline.py`, `view.py`, `provider.py` |
| Event cache | `docs/event-cache.md` | `event_cache.py` |
| Where a config, data view or run goes | `docs/outputs.md` | `configs/<dataset>/` |
| Training or GP run config | `docs/run-config.md`, then RESUM_FLEX `docs/configs.md` | `configs/` |
| Homogeneous campaign, neural field | `docs/photon-store.md` | `photon_store.py`, `photon_loader.py` |
| Compare models, which data a model trains on | `docs/benchmark.md` | `benchmark.py`, `configs/<dataset>/benchmark.yaml` |
| countfield inputs | `docs/countfield.md` | `scripts/export_*.py` |
| Submit or check jobs | `docs/slurm.md`, RESUM_FLEX `docs/nersc.md` | `scripts/slurm/` |
| Earlier results of a question | `analysis/<date_topic>/README.md` (if present) | |

## Concepts
- voxel = design point θ (centre), photon = event φ (vertex offset, momentum), outcome =
  photon seen by any channel. Quantity of interest: hit rate per voxel (hits / photons).
- LF (1500 photons/voxel) and HF (5000). Roles: `cnp_train`, `cnp_validation`, `cnp_test`
  (LF event models), `gp_unlabeled` (LF2, inputs only), `gp_train` (HF), `gp_validation`,
  `gp_test` (HF benchmark, blinded).
- Rate models (binomial GP, MFGP, BLL MLP, neural field, countfield) all train on LF2 + HF
  train and are scored on `gp_validation/hf`; support networks feeding a GP train on LF train.
- Datasets, named by volume (folder in `configs/` and `outputs/`): `centered` (Ian Wang) =
  `voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm`, `lbnl_secondment` (Sofia Calgaro) =
  `voxel_sets_x-800-600_y+50+250_z+400+600_5mm`, genesis (own, raw data on `$PSCRATCH`: purge
  policy) = `voxel_sets_x+300+500_y-100+100_z-100+100_5mm` and `photon_x+300+500_y-100+100_z-100+100`.

## Commands
```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/pytest -q                       # 108 tests, ~2 min on a login node
.venv/bin/ruff check .                    # must stay clean
.venv/bin/python -m core.surrogates train|mfgp|binomial|stream CONFIG.yaml
.venv/bin/python -m core.surrogates predict RUN/checkpoints/best DATA_DIR PARTITION
.venv/bin/python -m legend_optical_map_ml.benchmark configs/<dataset>/benchmark.yaml  # reads runs only
```
Login nodes: one single-threaded process of up to ~15 min; real data, GP fits and training
go to Slurm (`docs/slurm.md`).

## Rules
- RESUM_FLEX rules apply: behaviour through YAML, explicit seeds, statistics from training
  roles only, test sets blinded (`unblind_test`), one never-overwritten `output_directory` per
  run (bump `_vN`).
- Data: use raw simulation outputs, not reboost maps. Exclude only truncated or broken files
  (`exclude_files`); keep voxels partly overlapping a Ge detector as normal voxels. New
  campaigns sample validation and test disjoint; don't add split rules to repair overlaps.
- Ad-hoc analyses go to `analysis/YYYY-MM-DD_topic/` (scripts, results, a short `README.md`
  with the finding), never committed; large intermediates to `outputs/`.
- Style: ruff (line 100, E,F,I,B,UP, py311), type hints, a synthetic-fixture test per change.
  Check the code rather than guess; ask when intent or data format is unclear.

## Don't
- Don't read `.venv/`, raw LH5 files (inspect with a short h5py script), or arrays,
  checkpoints and plots in `outputs/`; read a run's `metrics.json`, `compute.json`,
  `experiment.json` or a benchmark `report.md`.
- Don't commit `outputs/`, `analysis/`, `.claude/` or data. Don't put application code, paths
  or configs into RESUM_FLEX or countfield.

## Workflow
After each finished, tested step: commit the code, tests, configs and docs you changed (stage
files explicitly, leave the user's uncommitted edits out) and push the current branch to
`origin`. Never commit the local files (`.claude/`, `analysis/`, `outputs/`).

## Keeping context current
- Facts about the code, data or datasets go into the matching doc (one topic per file, listed
  in `docs/index.md`), not into memory or this file. New topic → new short file + index line +
  a row in the table above.
- When you add, rename or remove modules, commands, datasets or rules, update this file and
  the doc in the same commit.
- `.claude/status.md`: open items only, one line each, dated; delete finished items at the end
  of each task.
