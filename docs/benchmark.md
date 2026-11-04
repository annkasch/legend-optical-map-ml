# Benchmark protocol

The quantity of interest is the hit rate per voxel (hits / photons). What is benchmarked is
the **rate model**: the model that outputs this rate, a GP or an equivalent alternative.

## Data

| Pool | Partitions | Used by |
|---|---|---|
| Rate training | `gp_unlabeled/lf` (LF2 train) + `gp_train/hf` (HF train) | every rate model |
| Support training | `cnp_train/lf` (LF train), selected on `cnp_validation/lf` | networks that only feed a GP |
| Benchmark | `gp_validation/hf`; `gp_test/hf` once at the end | every rate model |

1. **Rate models get the same data.** Every model that outputs the voxel rate trains on
   LF2 + HF train: the binomial GP, the MFGP, the neural process, and a network with a
   Bayesian last layer or a neural field (`train_sets` in the run config). Networks select
   their checkpoint on LF validation.
2. **Support models are part of a pipeline.** A network that only supplies inputs to a GP
   (NN + GP) trains on LF train and is selected on LF validation. It predicts the LF2 voxels
   from their inputs only, so LF2 needs no extra simulation; its data and compute count as
   the pipeline's cost and are reported with it.
3. **The benchmark sets are never used for training or checkpoint selection.**
   `gp_test/hf` stays blinded until a config sets `unblind_test: true`.

## Scoring

Each rate model predicts a rate and its uncertainty for every benchmark voxel. These are
compared with the observed HF counts using the same metrics: count NLL, MAE, RMSE, CRPS,
Pearson r, bias and coverage, together with the compute of the whole pipeline (core-hours,
from each run's `compute.json`).
