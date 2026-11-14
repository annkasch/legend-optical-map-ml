"""Benchmark report: every rate model scored the same way on the same held-out voxels.

    python -m legend_optical_map_ml.benchmark BENCHMARK.yaml

The tool only reads finished runs; it never trains or predicts. Each model's run must already
hold its per-voxel prediction of the benchmark set (``partition``/``fidelity``):

==============  ==========================================================================
kind            file read from ``run``
==============  ==========================================================================
binomial_gp     ``evaluation_<e>.npz`` (RESUM_FLEX binomial GP; ``e`` from metrics.json)
mfgp            ``evaluation_<e>.npz`` (RESUM_FLEX MFGP, target level)
countfield      ``evaluation_<partition>_<fidelity>.npz`` (countfield run)
event_model     ``score.json`` and ``<partition>_<fidelity>_<checkpoint>_predictive.npz`` of the
                RESUM_FLEX ``score`` stage (directory ``score``, default
                ``<run>/score_<partition>_<fidelity>``)
==============  ==========================================================================

From these arrays (hits k, trials n, predictive mean, 1/2/3 sigma intervals) every model gets
the same metrics; see ``docs/benchmark.md``. All models must cover the same voxels in the same
order. Compute is summed over the runs listed per model (``compute.json`` of RESUM_FLEX runs,
or the ``compute`` entry of score/inference files).

Writes ``report.md``, ``benchmark.csv``, ``benchmark.json`` and ``benchmark.png`` into
``output_directory``; a report is derived data and is rewritten on every call.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binom

from legend_optical_map_ml.config import StrictModel
from legend_optical_map_ml.provenance import code_block

LEVELS = (1, 2, 3)
NOMINAL = {1: 0.6827, 2: 0.9545, 3: 0.9973}


class ComputeSource(StrictModel):
    """One run of a pipeline: a run directory (compute.json) or a json file with compute."""

    path: Path


class BenchmarkModel(StrictModel):
    label: str
    kind: Literal["binomial_gp", "mfgp", "countfield", "event_model"]
    run: Path
    # event_model only: output directory of the RESUM_FLEX score stage (default
    # <run>/score_<partition>_<fidelity>), the scored checkpoint, and which photons it was scored
    # on: "donors" (the protocol, docs/benchmark.md) or "own".
    score: Path | None = None
    checkpoint: str = "best"
    event_inputs: Literal["donors", "own"] = "donors"
    # Runs whose compute adds up to this model's cost (e.g. MLP training + inference + GP).
    # Empty: the run itself (its compute.json; for event_model also the scoring file).
    compute: list[ComputeSource] = Field(default_factory=list)
    note: str | None = None


class BenchmarkConfig(StrictModel):
    title: str
    partition: str = "gp_validation"
    fidelity: Literal["lf", "hf"] = "hf"
    # Test partitions (name ending in "test") are read only with unblind_test: true.
    unblind_test: bool = False
    # Label of the model the paired NLL difference refers to; default: the first model.
    reference: str | None = None
    # Rates (MAE, RMSE, CRPS) are reported in units of this value.
    rate_unit: float = Field(1e-4, gt=0)
    output_directory: Path
    models: list[BenchmarkModel] = Field(min_length=1)

    @model_validator(mode="after")
    def _checks(self):
        labels = [m.label for m in self.models]
        if len(set(labels)) != len(labels):
            raise ValueError("Model labels must be unique")
        if self.reference is not None and self.reference not in labels:
            raise ValueError(f"reference {self.reference!r} is not a model label")
        if self.partition.endswith("test") and not self.unblind_test:
            raise ValueError(
                f"{self.partition} is a test partition; set unblind_test: true once all "
                "model choices are final"
            )
        return self

    @property
    def reference_label(self) -> str:
        return self.reference or self.models[0].label


def load_benchmark_config(path) -> BenchmarkConfig:
    """Load a benchmark YAML; paths are relative to the file."""
    import yaml

    path = Path(path).resolve()
    config = BenchmarkConfig.model_validate(yaml.safe_load(path.read_text()))

    def resolve(value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else (path.parent / value).resolve()

    config.output_directory = resolve(config.output_directory)
    for model in config.models:
        model.run = resolve(model.run)
        if model.score is not None:
            model.score = resolve(model.score)
        for source in model.compute:
            if source.path is not None:
                source.path = resolve(source.path)
    return config


@dataclass
class VoxelPrediction:
    """Per-voxel observation predictive of one model on the benchmark set."""

    hits: np.ndarray  # [B] observed hits k
    trials: np.ndarray  # [B] photons n
    mean: np.ndarray  # [B] predictive mean of the hit fraction
    intervals: dict  # level -> (lower [B], upper [B]) of the hit fraction
    source: Path
    native: dict = field(default_factory=dict)  # the run's own crps / predictive_nll


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    rows = json.loads(path.read_text())
    return rows if isinstance(rows, list) else [rows]


def _arrays(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with np.load(path) as a:
        out = {key: a[key] for key in ("observed", "target_events", "mean")}
        out["intervals"] = {k: (a[f"lower_{k}"], a[f"upper_{k}"]) for k in LEVELS}
        out["hits"] = a["hits"] if "hits" in a.files else None
    if out["hits"] is None:  # MFGP files store fractions only
        exact = out["observed"] * out["target_events"]
        out["hits"] = np.rint(exact).astype(np.int64)
        if not np.allclose(out["hits"], exact, atol=1e-6):
            raise ValueError(f"{path}: observed * target_events is not an integer count")
    return out


def _native(row: dict, *, count_nll: bool) -> dict:
    nll = row.get("predictive_nll") if count_nll else None
    return dict(crps=row.get("crps"), predictive_nll=nll)


def score_directory(model: BenchmarkModel, partition: str, fidelity: str) -> Path:
    """Output directory of the score stage for an event model."""
    return model.score or model.run / f"score_{partition}_{fidelity}"


def load_prediction(model: BenchmarkModel, partition: str, fidelity: str) -> VoxelPrediction:
    """The benchmark-set prediction of one model, from the files of its finished run."""
    run, key = model.run, f"{partition}/{fidelity}"
    if model.kind in ("binomial_gp", "mfgp"):
        rows = [
            r
            for r in _rows(run / "metrics.json")
            if (r.get("partition"), r.get("fidelity")) == (partition, fidelity)
        ]
        if not rows:
            raise ValueError(f"{run}: no evaluation of {key} in metrics.json")
        row = max(rows, key=lambda r: r.get("level", 0))  # MFGP: the target level
        path = run / f"evaluation_{row['evaluation']}.npz"
        # The MFGP's predictive_nll is a density, not a count mass.
        native = _native(row, count_nll=model.kind == "binomial_gp")
    elif model.kind == "countfield":
        rows = [r for r in _rows(run / "metrics.json") if r.get("group") == key]
        if not rows:
            raise ValueError(f"{run}: no evaluation of {key} in metrics.json")
        path = run / f"evaluation_{partition}_{fidelity}.npz"
        native = _native(rows[0], count_nll=True)
    else:
        folder = score_directory(model, partition, fidelity)
        report = folder / "score.json"
        if not report.is_file():
            raise FileNotFoundError(
                f"{report} not found; score the run first: core.surrogates score <score config>"
            )
        rows = [
            r
            for r in json.loads(report.read_text())["rows"]
            if (r["partition"], r["fidelity"], r["checkpoint"])
            == (partition, fidelity, model.checkpoint)
        ]
        if not rows:
            raise ValueError(f"{report}: no score of {key} for checkpoint {model.checkpoint}")
        used = "own" if rows[0]["event_inputs"] == "own" else "donors"
        if used != model.event_inputs:
            raise ValueError(
                f"{report}: scored on {used} photons, the entry expects {model.event_inputs}"
            )
        path = folder / f"{partition}_{fidelity}_{model.checkpoint}_predictive.npz"
        native = _native(rows[0], count_nll=True)
    a = _arrays(path)
    return VoxelPrediction(
        hits=np.asarray(a["hits"], dtype=np.int64),
        trials=np.asarray(a["target_events"], dtype=np.int64),
        mean=np.asarray(a["mean"], dtype=float),
        intervals=a["intervals"],
        source=path,
        native=native,
    )


def voxel_nll(pred: VoxelPrediction) -> np.ndarray:
    """Plug-in count NLL per voxel: -log Binomial(k | n, predictive mean) [B]."""
    m = pred.mean
    if not np.isfinite(m).all() or np.any(m <= 0) or np.any(m >= 1):
        raise ValueError(f"{pred.source}: predictive means must lie strictly inside (0, 1)")
    return -binom.logpmf(pred.hits, pred.trials, m)


def score(pred: VoxelPrediction) -> dict:
    """The same metrics for every model, from its per-voxel arrays."""
    observed = pred.hits / pred.trials
    m = pred.mean
    residual = m - observed
    ratio = m.mean() / observed.mean()
    # Delta-method standard error of the ratio of means (paired over voxels).
    bias_se = (m - ratio * observed).std(ddof=1) / np.sqrt(len(m)) / observed.mean()
    coverage = {
        k: float(((observed >= lo) & (observed <= hi)).mean())
        for k, (lo, hi) in pred.intervals.items()
    }
    return dict(
        voxels=len(m),
        count_nll=float(voxel_nll(pred).mean()),
        predictive_nll=pred.native.get("predictive_nll"),
        mae=float(np.abs(residual).mean()),
        rmse=float(np.sqrt(np.square(residual).mean())),
        crps=pred.native.get("crps"),
        pearson_r=float(np.corrcoef(m, observed)[0, 1]),
        bias=float(ratio - 1),
        bias_se=float(bias_se),
        coverage=coverage,
    )


def _file_compute(path: Path) -> list[dict]:
    """Total records of a run directory (compute.json) or a json file with a compute entry."""
    if path.is_dir():
        path = path / "compute.json"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        records = [json.loads(path.read_text())]
    else:
        entry = json.loads(path.read_text()).get("compute")
        if entry is None:
            raise ValueError(f"{path} has no compute entry")
        # Inference files: one record; older score files: {"run": ..., "inference": ...}.
        records = [entry] if "total" in entry else [v for v in entry.values() if v]
    return [
        dict(
            source=str(path),
            wall_seconds=r["total"]["wall_seconds"],
            cpu_seconds=r["total"]["cpu_seconds"],
            core_hours=r["total"]["core_hours"],
        )
        for r in records
    ]


def model_compute(model: BenchmarkModel, config: BenchmarkConfig):
    """Summed compute of a model's pipeline, or None when nothing is recorded."""
    sources = list(model.compute)
    if not sources:
        if (model.run / "compute.json").is_file():
            sources.append(ComputeSource(path=model.run))
        if model.kind == "event_model":
            folder = score_directory(model, config.partition, config.fidelity)
            if (folder / "compute.json").is_file():
                sources.append(ComputeSource(path=folder))
    if not sources:
        return None
    records = []
    for source in sources:
        records.extend(_file_compute(source.path))
    return dict(
        cpu_hours=sum(r["cpu_seconds"] for r in records) / 3600,
        core_hours=sum(r["core_hours"] for r in records),
        wall_hours=sum(r["wall_seconds"] for r in records) / 3600,
        runs=records,
    )


def run_benchmark(config: BenchmarkConfig) -> Path:
    """Score every model, write the report files and return the report path."""
    predictions = {
        m.label: load_prediction(m, config.partition, config.fidelity) for m in config.models
    }
    first_label = config.models[0].label
    first = predictions[first_label]
    for label, pred in predictions.items():
        if not (
            np.array_equal(pred.hits, first.hits) and np.array_equal(pred.trials, first.trials)
        ):
            raise ValueError(
                f"{label} ({pred.source}) does not score the same voxels in the same order as "
                f"{first_label} ({first.source})"
            )
    reference = voxel_nll(predictions[config.reference_label])
    results = []
    for model in config.models:
        pred = predictions[model.label]
        scores = score(pred)
        difference = voxel_nll(pred) - reference
        scores["delta_nll"] = float(difference.mean())
        scores["delta_nll_se"] = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        results.append(
            dict(
                label=model.label,
                kind=model.kind,
                run=str(model.run),
                prediction=str(pred.source),
                note=model.note,
                scores=scores,
                compute=model_compute(model, config),
            )
        )
    directory = config.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    observed = first.hits / first.trials
    summary = dict(
        title=config.title,
        benchmark_set=f"{config.partition}/{config.fidelity}",
        voxels=len(observed),
        mean_observed_rate=float(observed.mean()),
        mean_trials=float(first.trials.mean()),
        reference=config.reference_label,
        rate_unit=config.rate_unit,
        config=config.model_dump(mode="json"),
        code=code_block(),
        models=results,
    )
    (directory / "benchmark.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    _write_csv(directory / "benchmark.csv", results)
    _plot(directory, results, config)
    path = directory / "report.md"
    path.write_text(_report(summary, config))
    return path


def _write_csv(path: Path, results: list[dict]):
    fields = [
        "label", "kind", "count_nll", "delta_nll", "delta_nll_se", "predictive_nll", "mae",
        "rmse", "crps", "pearson_r", "bias", "bias_se", "coverage_1", "coverage_2",
        "coverage_3", "cpu_hours", "core_hours",
    ]  # fmt: skip
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            s, c = r["scores"], r["compute"] or {}
            writer.writerow(
                dict(
                    label=r["label"],
                    kind=r["kind"],
                    **{k: s[k] for k in fields[2:12]},
                    **{f"coverage_{k}": s["coverage"][k] for k in LEVELS},
                    cpu_hours=c.get("cpu_hours"),
                    core_hours=c.get("core_hours"),
                )
            )


def _fmt(value, digits, scale=1.0):
    return "–" if value is None else f"{value / scale:.{digits}f}"


def _report(summary: dict, config: BenchmarkConfig) -> str:
    unit = config.rate_unit
    exponent = f"{unit:.0e}".replace("e-0", "e-").replace("1e", "10^")
    lines = [
        f"# {summary['title']}",
        "",
        f"Benchmark set: `{summary['benchmark_set']}`, {summary['voxels']} voxels, mean observed "
        f"rate {summary['mean_observed_rate']:.3e}, {summary['mean_trials']:.0f} photons per "
        f"voxel on average. Paired differences refer to **{summary['reference']}**.",
        "",
        "| Model | NLL | ΔNLL | Pred. NLL | MAE | RMSE | CRPS | r | Bias % | "
        "Coverage 1σ / 2σ / 3σ | CPU-h | Core-h |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["models"]:
        s, c = r["scores"], r["compute"] or {}
        cov = " / ".join(f"{s['coverage'][k]:.2f}" for k in LEVELS)
        cells = [
            r["label"],
            f"{s['count_nll']:.3f}",
            f"{s['delta_nll']:+.3f} ± {s['delta_nll_se']:.3f}",
            _fmt(s["predictive_nll"], 3),
            *(_fmt(s[key], 2, unit) for key in ("mae", "rmse", "crps")),
            f"{s['pearson_r']:.3f}",
            f"{100 * s['bias']:+.1f} ± {100 * s['bias_se']:.1f}",
            cov,
            _fmt(c.get("cpu_hours"), 1),
            _fmt(c.get("core_hours"), 1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        f"MAE, RMSE and CRPS in units of {exponent}. Nominal coverage: 0.68 / 0.95 / 1.00.",
        "",
        "## Definitions",
        "",
        "- **NLL**: plug-in count NLL, −log Binomial(k | n, predictive mean) averaged over "
        "voxels (nats). Computed identically for every model; lower is better.",
        f"- **ΔNLL**: paired NLL difference to {summary['reference']} with its standard error "
        "over voxels; differences within about two standard errors are not significant.",
        "- **Pred. NLL**: the run's own count NLL of its full predictive (counts mixed over "
        "its uncertainty). Not available (–) for models whose predictive is a density (MFGP).",
        "- **MAE, RMSE**: of the predictive mean against the observed fraction k/n. They "
        "include the binomial counting noise of the observations.",
        "- **CRPS**: as reported by each run, from its own predictive draws of the fraction.",
        "- **r**: Pearson correlation of the predictive mean and the observed fraction.",
        "- **Bias %**: mean predicted / mean observed − 1, with its delta-method standard error.",
        "- **Coverage**: share of voxels whose observed fraction lies in the model's central "
        "1σ / 2σ / 3σ-equivalent observation interval.",
        "- **CPU-h**: CPU time used, summed over threads and over the pipeline's runs. "
        "**Core-h**: wall time × allocated physical cores, what the allocation costs. "
        "– means the runs record no compute.",
        "",
        "![Benchmark](benchmark.png)",
        "",
        "## Runs",
        "",
    ]
    for r in summary["models"]:
        lines.append(f"- **{r['label']}** ({r['kind']}): `{r['prediction']}`")
        if r["note"]:
            lines.append(f"  - {r['note']}")
        for source in (r["compute"] or {}).get("runs", []):
            lines.append(
                f"  - compute: {source['source']}: {source['cpu_seconds'] / 3600:.2f} CPU-h, "
                f"{source['core_hours']:.2f} core-h"
            )
    return "\n".join(lines) + "\n"


# Reference palette (dataviz skill): one hue for the data, ink and muted tones for the rest.
SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"


def _plot(directory: Path, results: list[dict], config: BenchmarkConfig):
    """Dot plots: paired NLL difference and coverage per level, one row per model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    y = np.arange(len(labels))[::-1]  # first model on top
    fig, axes = plt.subplots(
        1, 4, figsize=(14, 1.0 + 0.45 * len(labels)), sharey=True, layout="constrained"
    )
    fig.set_facecolor(SURFACE)
    delta = np.array([r["scores"]["delta_nll"] for r in results])
    se = np.array([r["scores"]["delta_nll_se"] for r in results])
    panels = [(axes[0], delta, se, 0.0, f"ΔNLL vs {config.reference_label} (± SE)")]
    for ax, k in zip(axes[1:], LEVELS, strict=True):
        cov = np.array([r["scores"]["coverage"][k] for r in results])
        panels.append((ax, cov, None, NOMINAL[k], f"Coverage {k}σ (nominal {NOMINAL[k]:.3f})"))
    for ax, values, errors, line, title in panels:
        ax.set_facecolor(SURFACE)
        ax.axvline(line, color=MUTED, lw=1, ls="--", zorder=1)
        ax.errorbar(
            values, y, xerr=errors, fmt="o", ms=8, color=SERIES, ecolor=SERIES, elinewidth=2,
            capsize=0, mec=SURFACE, mew=2, zorder=3,
        )  # fmt: skip
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.grid(axis="x", color=MUTED, alpha=0.15, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    axes[0].set_yticks(y, labels, color=INK)
    fig.suptitle(
        f"{config.title}: {config.partition}/{config.fidelity}", color=INK, fontsize=11, x=0.01,
        ha="left",
    )  # fmt: skip
    fig.savefig(directory / "benchmark.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="benchmark YAML")
    args = parser.parse_args(argv)
    print(run_benchmark(load_benchmark_config(args.config)), flush=True)


if __name__ == "__main__":
    main()
