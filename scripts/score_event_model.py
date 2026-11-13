"""Score a context-free event model with a Bayesian last layer standalone on voxel counts.

    python scripts/score_event_model.py RUN/checkpoints/best DATA_DIR \
        [--partition gp_validation --fidelity hf --donors gp_validation/hf --seed 0]

Each scored voxel is predicted from its own position with the photon inputs of a seeded
random donor voxel (a fresh emission sample), so nothing of its own simulation enters. The
donors default to the scored set itself (never the voxel itself, same photon count, never
trained on); the provider moves them into the voxel (absolute positions shifted). The
counts k / n of the voxel are then scored against the binomial predictive mixed over the
last-layer draws (resum_flex core.gp_metrics.count_predictive_scores), the same measures as
the binomial GP. ``plugin_count_nll`` uses only the mean p, as in the model comparison table.
Writes RUN/inference/<checkpoint>/<partition>_<fidelity>_events-<donors>_scores.json, the
per-voxel predictive arrays (_predictive.npz) and coverage / z-score plots (to
--plot-directory, default next to the scores; --label names the model in titles and files).
``compute`` holds the scoring run's stages (resum_flex core.compute) and the inference
file's own compute record, whether predicted now or earlier.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from core.compute import ComputeLog
from core.gp_metrics import count_predictive_scores
from core.surrogates.checkpoints import load_surrogate
from core.surrogates.datasets import load_provider
from core.surrogates.inference import predict_partition
from scipy.stats import binom

from legend_optical_map_ml.benchmark import donor_stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("--partition", default="gp_validation")
    parser.add_argument("--fidelity", default="hf")
    parser.add_argument(
        "--donors", default=None, help="PARTITION/FIDELITY (default: the scored set itself)"
    )
    parser.add_argument("--seed", type=int, default=0, help="donor draw seed")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--plot-directory", type=Path, default=None)
    parser.add_argument("--label", default=None, help="model name for plot titles and files")
    parser.add_argument("--coverage-ylim", type=float, nargs=2, default=(0.0, 0.0065))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    compute = ComputeLog()
    if args.donors is None:
        args.donors = f"{args.partition}/{args.fidelity}"
    donors = tuple(args.donors.split("/"))
    model = load_surrogate(args.checkpoint)
    if getattr(model, "last_layer", None) is None:
        raise SystemExit("The checkpoint has no Bayesian last layer")
    folder = args.checkpoint.parent.parent / "inference" / args.checkpoint.name
    path = folder / (donor_stem(f"{args.partition}_{args.fidelity}", args.donors) + ".npz")
    if path.exists():  # inference files are never overwritten: reuse the earlier prediction
        saved = json.loads(path.with_suffix(".json").read_text())["event_inputs"]
        if saved["seed"] != args.seed:
            raise SystemExit(f"{path} was predicted with donor seed {saved['seed']}")
    else:
        path = predict_partition(
            args.checkpoint,
            args.data_directory,
            args.partition,
            args.fidelity,
            threads=args.threads,
            events_from=donors,
            seed=args.seed,
        )
    compute.lap("checkpoint_and_inference")
    inference_compute = json.loads(path.with_suffix(".json").read_text()).get("compute")
    provider = load_provider(model.metadata.get("data_provider"), args.data_directory)
    batch, _ = provider.load(args.partition, args.fidelity)
    hits = np.where(batch.event_mask, batch.labels, 0).sum(axis=1)
    trials = batch.events_per_design_point
    with np.load(path) as arrays:
        draws = arrays["probability_draws"].astype(float)  # [S, B]
        plugin = arrays["mean_probability"]  # [B] mean p with the trained weights
    scores, arrays = count_predictive_scores(hits, trials, draws, seed=model.config.last_layer.seed)
    scores.update(
        plugin_count_nll=float(-binom.logpmf(hits, trials, plugin).mean()),
        checkpoint=str(args.checkpoint.resolve()),
        partition=f"{args.partition}/{args.fidelity}",
        donors=args.donors,
        donor_seed=args.seed,
        model=model.config.model_dump(mode="json"),
        last_layer=model.metadata.get("last_layer"),
    )
    compute.lap("scoring")
    np.savez_compressed(path.with_name(path.stem + "_predictive.npz"), **arrays)
    _plots(arrays, args, path)
    compute.lap("plots")
    scores["compute"] = dict(run=compute.summary(), inference=inference_compute)
    out = path.with_name(path.stem + "_scores.json")
    out.write_text(json.dumps(scores, indent=2, allow_nan=False))
    print(out)
    print(
        {
            k: scores[k]
            for k in (
                "predictive_nll",
                "plugin_count_nll",
                "mae",
                "rmse",
                "pearson_r",
                "coverage",
                "crps",
            )
        }
    )


def _plots(arrays, args, path):
    """Coverage of the observed fractions and z-scores, as for the binomial GP."""
    from viz.dispatch import plot_coverage_test
    from viz.zscore import plot_zscores

    label = args.label or path.parent.parent.parent.name
    directory = args.plot_directory or path.parent
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_{args.partition}_{args.fidelity}"
    title = f"{label}, {args.partition}/{args.fidelity}"
    intervals = {k: (arrays[f"lower_{k}"], arrays[f"upper_{k}"]) for k in (1, 2, 3)}
    plot_coverage_test(
        arrays["observed"],
        arrays["mean"],
        arrays["sigma"],
        out_path=directory / f"{stem}_coverage.png",
        title=f"{title}: observation-predictive coverage (Bayesian last layer)",
        xlabel="Validation voxel index",
        predicted_label="Predictive mean",
        raw_label="Observed fraction",
        ylim=tuple(args.coverage_ylim),
        intervals=intervals,
        show_predicted=False,
    )
    z = (arrays["observed"] - arrays["mean"]) / arrays["sigma"]
    plot_zscores(
        z,
        arrays["mean"],
        directory / f"{stem}_zscore",
        title=f"{title}: z = (observed − predictive mean) / predictive sd",
        rate_label="Predicted rate",
    )


if __name__ == "__main__":
    main()
