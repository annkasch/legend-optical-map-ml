# Submit a dependent Slurm pipeline

The training, prediction, and GP fitting stages can run as separate Slurm jobs. The
submission script connects them with `afterok` dependencies: prediction starts only after
training succeeds, and MFGP fitting starts only after prediction succeeds.

From the repository root, submit a run-config and its matching standalone GP config:

```bash
scripts/slurm/submit_surrogate_pipeline.sh \
  configs/config.optical.centered.mlp.yaml \
  configs/config.optical.centered.mfgp.yaml \
  -q regular -t 01:00:00
```

The wrapper reads the model run directory, data directory, model-mean partition, fidelity,
and inference output path from the two YAML files. It requires a context-free model and one
`model_mean` GP level. It submits three jobs and prints their IDs. Check their progress with
`squeue -j JOBID` or the Slurm logs in `outputs/slurm_logs/`.

The training output directory must be empty, and the configured inference output must not
already exist. Choose new output paths for another run. If a stage fails, the dependent jobs
remain blocked by `afterok` and will not run.

The wrapper accepts additional `sbatch` options after the two config paths. For example,
`-q regular -t 01:00:00` applies the requested queue and time limit to all three jobs. Set
`PREDICT_THREADS` to change the prediction worker count; its default is 8.
