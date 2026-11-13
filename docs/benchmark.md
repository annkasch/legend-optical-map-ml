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
   For now it trains on a view with θ: GP stages cannot read the predictions of event-only
   models yet (`docs/data-config.md`).
3. **The benchmark sets are never used for training or checkpoint selection.**
   `gp_test/hf` stays blinded until a config sets `unblind_test: true`.

## Scoring

Each rate model predicts a rate and its uncertainty for every benchmark voxel. These are
compared with the observed HF counts using the same metrics: count NLL, MAE, RMSE, CRPS,
Pearson r, bias and coverage, together with the compute of the whole pipeline (core-hours,
from each run's `compute.json`).

## Running the benchmark

```bash
python -m legend_optical_map_ml.benchmark configs/voxel_sets_x-800-600_y+50+250_z+400+600_5mm/benchmark.yaml
```

The tool only reads finished runs, so it is cheap (fine on a login node). Each entry of
`models` names a `label`, a `kind` and the `run` directory holding its benchmark prediction:

| kind | prediction read from the run |
|---|---|
| `binomial_gp`, `mfgp` | `evaluation_<e>.npz` of the benchmark set (MFGP: target level) |
| `countfield` | `evaluation_<partition>_<fidelity>.npz` |
| `event_model` | `inference/<checkpoint>/…_predictive.npz` and `_scores.json`; score the checkpoint first with `scripts/score_event_model.py` (`checkpoint`, `donors` select the file) |

`compute` lists every run of the model's pipeline (`path:`): a run directory (`compute.json`)
or a json file with a `compute` entry. Without it, the run's own records are used; countfield
runs record none. All models must score the same voxels
in the same order; `reference` names the model for the paired NLL difference. Test
partitions need `unblind_test: true`.

`output_directory` receives `report.md` (table, definitions, runs), `benchmark.csv`,
`benchmark.json` and `benchmark.png`; the report is rewritten on every call.
