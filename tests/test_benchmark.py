"""Benchmark report from synthetic finished runs of every supported kind."""

import json

import numpy as np
import pytest
import yaml
from pydantic import ValidationError
from scipy.stats import binom

from legend_optical_map_ml.benchmark import (
    BenchmarkConfig,
    donor_stem,
    load_benchmark_config,
    run_benchmark,
)

B, N = 40, 5000
rng = np.random.default_rng(0)
TRUE_P = rng.uniform(0.8e-3, 1.6e-3, B)
HITS = rng.binomial(N, TRUE_P)


def _arrays(mean, *, hits=True):
    sigma = np.sqrt(mean * (1 - mean) / N) * 1.2
    out = dict(
        observed=HITS / N,
        target_events=np.full(B, N),
        mean=mean,
        sigma=sigma,
        **{f"lower_{k}": mean - k * sigma for k in (1, 2, 3)},
        **{f"upper_{k}": mean + k * sigma for k in (1, 2, 3)},
    )
    if hits:
        out["hits"] = HITS
    return out


def _compute(folder, cpu_seconds, core_hours):
    record = dict(total=dict(wall_seconds=60.0, cpu_seconds=cpu_seconds, core_hours=core_hours))
    (folder / "compute.json").write_text(json.dumps(record))


def _runs(root):
    """One finished run per kind; returns {kind: run directory}."""
    runs = {k: root / k for k in ("binomial_gp", "mfgp", "countfield", "event_model")}
    for run in runs.values():
        run.mkdir(parents=True)
    # Binomial GP: two evaluation sets, the benchmark set is the second one.
    np.savez(runs["binomial_gp"] / "evaluation_1.npz", **_arrays(TRUE_P))
    (runs["binomial_gp"] / "metrics.json").write_text(
        json.dumps(
            [
                dict(evaluation=0, partition="gp_train", fidelity="hf"),
                dict(
                    evaluation=1,
                    partition="gp_validation",
                    fidelity="hf",
                    crps=3e-4,
                    predictive_nll=2.4,
                ),
            ]
        )
    )
    _compute(runs["binomial_gp"], 7200.0, 10.0)
    # MFGP: fractions only, target level 1; its density NLL must not be reported.
    np.savez(runs["mfgp"] / "evaluation_0.npz", **_arrays(TRUE_P * 1.1, hits=False))
    (runs["mfgp"] / "metrics.json").write_text(
        json.dumps(
            [
                dict(
                    evaluation=0,
                    partition="gp_validation",
                    fidelity="hf",
                    level=1,
                    crps=4e-4,
                    predictive_nll=-9.0,
                ),
            ]
        )  # fmt: skip
    )
    np.savez(runs["countfield"] / "evaluation_gp_validation_hf.npz", **_arrays(TRUE_P * 0.95))
    (runs["countfield"] / "metrics.json").write_text(
        json.dumps([dict(group="gp_validation/hf", crps=3.1e-4, predictive_nll=2.41)])
    )
    folder = runs["event_model"] / "inference/best"
    folder.mkdir(parents=True)
    stem = folder / "gp_validation_hf_events-cnp_train_lf"
    np.savez(f"{stem}_predictive.npz", **_arrays(0.5 * (TRUE_P + TRUE_P.mean())))
    scores = dict(
        crps=3.5e-4,
        predictive_nll=2.45,
        compute=dict(
            run=dict(total=dict(wall_seconds=10.0, cpu_seconds=360.0, core_hours=0.5)),
            inference=None,
        ),
    )
    stem.with_name(stem.name + "_scores.json").write_text(json.dumps(scores))
    _compute(runs["event_model"], 3600.0, 2.0)
    return runs


def _config(tmp_path, runs, **changes):
    raw = dict(
        title="Synthetic",
        output_directory=str(tmp_path / "report"),
        models=[
            {"label": "GP", "kind": "binomial_gp", "run": str(runs["binomial_gp"])},
            {"label": "MFGP", "kind": "mfgp", "run": str(runs["mfgp"])},
            {"label": "NP", "kind": "countfield", "run": str(runs["countfield"])},
            {"label": "MLP + BLL", "kind": "event_model", "run": str(runs["event_model"])},
        ],
        **changes,
    )
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_benchmark_config(path)


def test_report_scores_every_kind_the_same_way(tmp_path):
    runs = _runs(tmp_path / "runs")
    report = run_benchmark(_config(tmp_path, runs))
    summary = json.loads((report.parent / "benchmark.json").read_text())
    rows = {r["label"]: r for r in summary["models"]}
    gp = rows["GP"]["scores"]
    expected = -binom.logpmf(HITS, N, TRUE_P).mean()
    assert gp["count_nll"] == pytest.approx(expected)
    assert gp["delta_nll"] == 0 and gp["predictive_nll"] == 2.4 and gp["crps"] == 3e-4
    mfgp = rows["MFGP"]["scores"]
    assert mfgp["predictive_nll"] is None  # density NLL dropped
    assert mfgp["bias"] == pytest.approx(1.1 * TRUE_P.mean() / (HITS / N).mean() - 1)
    assert mfgp["delta_nll"] > 0
    assert rows["NP"]["scores"]["crps"] == 3.1e-4
    # Compute: compute.json of the run plus the scoring file for the event model.
    assert rows["GP"]["compute"]["cpu_hours"] == pytest.approx(2.0)
    assert rows["MLP + BLL"]["compute"]["core_hours"] == pytest.approx(2.5)
    assert rows["MFGP"]["compute"] is None
    text = report.read_text()
    assert "| GP | " in text and "| MFGP | " in text and "benchmark.png" in text
    for name in ("benchmark.csv", "benchmark.png"):
        assert (report.parent / name).is_file()
    assert len((report.parent / "benchmark.csv").read_text().splitlines()) == 5


def test_reference_and_mismatched_voxels(tmp_path):
    runs = _runs(tmp_path / "runs")
    report = run_benchmark(_config(tmp_path, runs, reference="NP"))
    rows = {
        r["label"]: r for r in json.loads((report.parent / "benchmark.json").read_text())["models"]
    }
    assert rows["NP"]["scores"]["delta_nll"] == 0
    arrays = _arrays(TRUE_P)
    arrays["hits"] = np.roll(HITS, 1)
    np.savez(runs["binomial_gp"] / "evaluation_1.npz", **arrays)
    with pytest.raises(ValueError, match="same voxels"):
        run_benchmark(_config(tmp_path, runs))


def test_missing_scoring_file_and_blinded_test(tmp_path):
    runs = _runs(tmp_path / "runs")
    scores = runs["event_model"] / "inference/best/gp_validation_hf_events-cnp_train_lf"
    scores.with_name(scores.name + "_scores.json").unlink()
    with pytest.raises(FileNotFoundError, match="score_event_model"):
        run_benchmark(_config(tmp_path, runs))
    raw = dict(title="t", output_directory="o", models=[{"label": "a", "kind": "mfgp", "run": "r"}])
    with pytest.raises(ValidationError, match="unblind_test"):
        BenchmarkConfig(**raw, partition="gp_test")
    BenchmarkConfig(**raw, partition="gp_test", unblind_test=True)
    with pytest.raises(ValidationError, match="unique"):
        BenchmarkConfig(**{**raw, "models": raw["models"] * 2})
    with pytest.raises(ValidationError, match="slurm_job"):  # runs record compute.json
        BenchmarkConfig(**{**raw, "models": [{**raw["models"][0], "compute": [{"slurm_job": 1}]}]})


def test_donor_stem():
    assert donor_stem("gp_validation_hf", None) == "gp_validation_hf"
    assert donor_stem("a_hf", "b/lf") == "a_hf_events-b_lf"
