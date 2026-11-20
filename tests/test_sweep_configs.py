"""Sweep configs mlp_bll.theta_absolute.<variant>.yaml change exactly the fields their name lists.

A variant is one part (`w512`) or several joined by `_` (`lr3e-3_d5`); each part changes one field.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = "mlp_bll.theta_absolute.yaml"
# variant prefix -> the one changed field, as a path into the dumped run config
FIELDS = {
    "w": ("model", "architecture", "hidden_dims"),
    "d": ("model", "architecture", "hidden_dims"),
    "lr": ("training", "learning_rate"),
    "b": ("training", "batch_size"),
    "ev": ("training", "n_events"),
    "do": ("model", "architecture", "dropout"),
}
SWEEP = sorted(ROOT.glob("configs/*/mlp_bll.theta_absolute.*.yaml"))


def _changed(a, b, path=()):
    """Paths of the leaves that differ between two nested dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in a.keys() | b.keys() for p in _changed(a.get(k), b.get(k), (*path, k))]
    return [] if a == b else [path]


def test_combined_variant_names_resolve_to_distinct_fields():
    # synthetic: the name parsing alone, no config files
    parts = "lr3e-3_d5".split("_")
    assert [re.match(r"[a-z]+", p).group() for p in parts] == ["lr", "d"]
    assert FIELDS["lr"] != FIELDS["d"] and FIELDS["w"] == FIELDS["d"]


def _variant(path: Path) -> str:
    return path.name.removeprefix("mlp_bll.theta_absolute.").removesuffix(".yaml")


def test_sweep_exists_for_both_voxel_datasets():
    datasets = {p.parent.name for p in SWEEP}
    assert len(datasets) == 2 and all(d.startswith("voxel_sets_") for d in datasets)


@pytest.mark.parametrize("path", SWEEP, ids=lambda p: f"{p.parent.name[:14]}/{_variant(p)}")
def test_variant_changes_its_fields_and_has_its_own_run_and_score(path):
    from schemas.surrogates import load_score_config, load_surrogate_config

    variant = _variant(path)
    fields = {FIELDS[re.match(r"[a-z]+", part).group()] for part in variant.split("_")}
    assert len(fields) == len(variant.split("_")), "two parts change the same field"
    base = load_surrogate_config(path.parent / BASE)
    config = load_surrogate_config(path)
    changed = _changed(base.model_dump(mode="json"), config.model_dump(mode="json"))
    assert sorted(changed) == sorted([("output_directory",), *fields])
    model = f"mlp_bll_theta_absolute_{variant}"
    assert config.output_directory.parts[-3:] == (model, "v1", "seed0")

    score = load_score_config(path.parent / f"score.{path.name}")
    base_score = load_score_config(path.parent / f"score.{BASE}")
    assert score.run == config.output_directory
    assert score.output_directory == config.output_directory / "score_rate_validation_hf"
    changed = _changed(base_score.model_dump(mode="json"), score.model_dump(mode="json"))
    assert sorted(changed) == [("output_directory",), ("run",)]
