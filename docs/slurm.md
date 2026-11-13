# Slurm jobs on Perlmutter

Login-node limits and general job settings: RESUM_FLEX `docs/nersc.md`. All scripts run from
the repository root, default to account m2676, `debug` QOS and 15 min, and log to
`outputs/slurm_logs/%x-%j.out` (`mkdir -p outputs/slurm_logs` first). Real runs:
`-q regular -t HH:MM:SS`. CPU scripts use `.venv/bin/python` (override with `RESUM_PYTHON`).

| Script | Runs |
|---|---|
| `train_surrogate.sbatch CONFIG.yaml [train\|mfgp\|binomial\|within_point\|residual\|score]` | CPU node, one stage; `BLAS_THREADS` (default 1) |
| `predict_surrogate.sbatch CHECKPOINT DATA_DIR PARTITION [options]` | CPU node, context-free mean prediction |
| `submit_surrogate_pipeline.sh RUN.yaml GP.yaml [sbatch options]` | train → predict → MFGP as dependent jobs |
| `train_photon_field.sbatch CONFIG.yaml` | GPU node, `core.surrogates stream`; NERSC `pytorch` module, RESUM_FLEX from `RESUM_FLEX_ROOT` (default `../RESUM_FLEX`); `--gpus-per-node=4` runs DDP via torchrun |

GP fits need many BLAS threads: `BLAS_THREADS=32` for `mfgp` / `binomial` (with 1 thread
the LBNL binomial GP timed out after 4 h; with 32 it took 34 min). Give a long time limit only
to the stage that needs it: queue wait grows with the limit, and a pending job's limit can be
lowered with `scontrol` but usually not raised.

## Dependent pipeline

```bash
scripts/slurm/submit_surrogate_pipeline.sh \
  configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mlp.yaml configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/mfgp.yaml \
  -q regular -t 01:00:00
```

The wrapper reads the run directory, data directory, model-mean partition, fidelity and
inference path from the two YAMLs. It needs a context-free model and one `model_mean` GP
level, submits three jobs with `afterok` dependencies and prints their IDs (`squeue -j ID`).
The training output directory must be empty and the inference output must not exist yet. A
failed stage leaves the later jobs blocked. Options after the configs apply to all three
jobs; `PREDICT_THREADS` (default 8) sets the prediction workers, `GP_BLAS_THREADS` (default
32) the MFGP BLAS threads, `GP_TIME` (e.g. `12:00:00`) the MFGP job's own time limit.

Compute per run (`compute.json`, or `sacct` for older jobs) enters the benchmark
(`docs/benchmark.md`).
