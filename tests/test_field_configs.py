"""Field training-setting variants field_f2.<parts>.yaml change exactly the fields their name lists.

Parts are joined by `_` (`cos_lr3e-3_p40`): `cos` = cosine learning-rate decay, `lr<x>` = learning
rate, `p<n>` = n_passes. The base `field_f2.yaml` of each box folder is the old setting (constant
rate 1e-3, 10 passes); each variant and the base have a score config on rate_validation/hf.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = (
    "photon_x+300+500_y-100+100_z-100+100",  # stream field on the photon store
    "voxel_sets_x+300+500_y-100+100_z-100+100_5mm",  # field on the voxel campaign
)
FIELDS = {
    "cos": ("training", "lr_schedule"),
    "lr": ("training", "learning_rate"),
    "p": ("training", "n_passes"),
}
VARIANTS = sorted(p for f in FOLDERS for p in (ROOT / "configs" / f).glob("field_f2.*.yaml"))


def _part_field(part: str):
    return FIELDS[re.match(r"cos$|[a-z]+", part).group()]


def _changed(a, b, path=()):
    """Paths of the leaves that differ between two nested dicts."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in a.keys() | b.keys() for p in _changed(a.get(k), b.get(k), (*path, k))]
    return [] if a == b else [path]


def _load(path: Path):
    from schemas.surrogates import load_stream_config, load_surrogate_config

    stream = "event_store" in yaml.safe_load(path.read_text())
    return (load_stream_config if stream else load_surrogate_config)(path)


def _variant(path: Path) -> str:
    return path.name.removeprefix("field_f2.").removesuffix(".yaml")


def test_part_names_resolve_to_distinct_fields():
    # synthetic: the name parsing alone, no config files
    assert [_part_field(p) for p in "cos_lr3e-3_p40".split("_")] == [
        FIELDS["cos"],
        FIELDS["lr"],
        FIELDS["p"],
    ]


def test_both_box_folders_have_variants():
    assert {p.parent.name for p in VARIANTS} == set(FOLDERS)


@pytest.mark.parametrize("folder", FOLDERS)
def test_base_is_the_old_constant_setting_with_a_score(folder):
    from schemas.surrogates import load_score_config

    base = _load(ROOT / "configs" / folder / "field_f2.yaml")
    assert base.model.kind == "field" and base.model.encoding.max_frequency == 2.0
    assert base.training.lr_schedule.kind == "constant"
    assert (base.training.learning_rate, base.training.n_passes) == (1e-3, 10)
    path = ROOT / "configs" / folder / "score.field_f2.yaml"
    assert load_score_config(path).run == base.output_directory
    assert yaml.safe_load(path.read_text())["sets"] == ["rate_validation/hf"]


@pytest.mark.parametrize("path", VARIANTS, ids=lambda p: f"{p.parent.name[:10]}/{_variant(p)}")
def test_variant_changes_its_fields_and_has_its_own_run_and_score(path):
    from schemas.surrogates import load_score_config

    variant = _variant(path)
    fields = [_part_field(part) for part in variant.split("_")]
    assert len(set(fields)) == len(fields), "two parts change the same field"
    base = _load(path.parent / "field_f2.yaml")
    config = _load(path)
    changed = _changed(base.model_dump(mode="json"), config.model_dump(mode="json"))
    changed = {c[:2] if c[:2] == FIELDS["cos"] else c for c in changed}  # schedule sub-keys
    assert sorted(changed) == sorted([("output_directory",), *fields])
    if "cos" in variant:
        assert config.training.lr_schedule.kind == "cosine"
    assert config.output_directory.parts[-3:] == (f"field_f2_{variant}", "v1", "seed0")

    score = load_score_config(path.parent / f"score.{path.name}")
    base_score = load_score_config(path.parent / "score.field_f2.yaml")
    assert score.run == config.output_directory
    assert score.output_directory == config.output_directory / "score_rate_validation_hf"
    changed = _changed(base_score.model_dump(mode="json"), score.model_dump(mode="json"))
    assert sorted(changed) == [("output_directory",), ("run",)]


def test_raw_position_field_differs_from_its_fourier_twin_only_in_the_encoding():
    from schemas.surrogates import load_score_config

    folder = ROOT / "configs" / FOLDERS[0]
    fourier = _load(folder / "field_f2.cos_lr3e-3.yaml")
    raw = _load(folder / "field_raw.cos_lr3e-3.yaml")
    assert raw.model.encoding.kind == "none" and list(raw.model.encoding.columns) == [0, 1, 2]
    changed = _changed(fourier.model_dump(mode="json"), raw.model_dump(mode="json"))
    assert ("output_directory",) in changed
    assert all(c == ("output_directory",) or c[:2] == ("model", "encoding") for c in changed)
    assert raw.output_directory.parts[-3:] == ("field_raw_cos_lr3e-3", "v1", "seed0")
    score = load_score_config(folder / "score.field_raw.cos_lr3e-3.yaml")
    assert score.run == raw.output_directory
