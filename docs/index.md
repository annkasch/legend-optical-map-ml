# Docs index

One topic per file. Read the one you need, not all of them. Toolkit topics (models, training,
GP, metrics, run outputs, stream training) are documented in RESUM_FLEX `docs/index.md`.

| File | Read when |
|---|---|
| [../README.md](../README.md) | First use: setup, the three CLI steps |
| [design.md](design.md) | Changing code: data flow, which file to open, why the rules exist |
| [datasets.md](datasets.md) | Working with a dataset: paths, sizes, roles, known bad files, open issues |
| [data-config.md](data-config.md) | Writing or changing a data YAML: sources, manifests, roles, features, target |
| [preparation.md](preparation.md) | How data are prepared and kept in sync; unlabeled inputs and `predict` |
| [event-cache.md](event-cache.md) | Event cache layout, `update` / `compact`, locks, large inputs |
| [run-config.md](run-config.md) | Writing a training or GP run config for optical data |
| [photon-store.md](photon-store.md) | Homogeneous campaigns: photon store, loader, neural-field stream training |
| [benchmark.md](benchmark.md) | Which data each model trains on; scoring; the benchmark report tool |
| [countfield.md](countfield.md) | Exporting counts or photons for the countfield models |
| [slurm.md](slurm.md) | Job scripts on Perlmutter and the dependent train → predict → GP pipeline |
