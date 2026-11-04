# Benchmark protocol

The quantity of interest is the hit rate per voxel (hits / photons). Every model is scored on
the same held-out HF voxels: `gp_validation/hf` while developing, and `gp_test/hf` once at the
end (blinded until a config sets `unblind_test: true`).

Each model predicts a rate and its uncertainty for every voxel. These are compared with the
observed HF counts using the same metrics: count NLL, MAE, RMSE, CRPS, Pearson r, bias and
coverage. This holds whatever the pipeline:

- **with a GP stage:** binomial GP, MFGP on raw LF + HF, GP on MLP/CNP predictions + HF;
- **without a GP stage:** MLP or CNP with a Bayesian last layer, neural field.

Models may be trained on different data, but they are all tested on the same data.
