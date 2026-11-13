"""Named data views: one split, own features and normalization, selected by data_directory."""

import json

import numpy as np
import pytest

from legend_optical_map_ml.config import OpticalDataConfig
from legend_optical_map_ml.pipeline import ensure_prepared_optical_data, load_partition
from legend_optical_map_ml.provider import OpticalDataset
from legend_optical_map_ml.view import OpticalView
from tests.test_optical_pipeline import dataset

pytest.importorskip("h5py")

ABSOLUTE = {"phi": {"vertex": {"representation": "absolute"}}}


def with_views(c, views):
    return OpticalDataConfig.model_validate({**c.model_dump(), "views": views})


def test_views_share_split_and_differ_in_features(tmp_path):
    c = with_views(dataset(tmp_path), {"absolute": ABSOLUTE})
    top = ensure_prepared_optical_data(c)
    named = c.view_directory("absolute")
    assert named == top / "views" / "absolute"
    base, view = OpticalView(top), OpticalView(named)
    assert np.array_equal(base.rows, view.rows)  # same files in the same partitions
    for name in ("view.json", "config.json"):
        assert "views" not in json.loads((top / name).read_text()).get("config", {})
    assert view.config.phi.vertex.representation == "absolute"
    assert view.config.event_cache == c.event_cache
    offset = load_partition(top, "train", "lf")[0]
    absolute = load_partition(named, "train", "lf")[0]
    assert offset.phi.shape == absolute.phi.shape
    assert np.array_equal(offset.labels, absolute.labels)
    # Each view normalizes its own columns: absolute positions differ from offsets.
    t, a = (json.loads((d / "normalization.json").read_text()) for d in (top, named))
    assert t["theta"] == a["theta"] and t["phi"]["offset"] != a["phi"]["offset"]
    assert t["fit_files"] == a["fit_files"]


def test_changed_view_rewrites_only_that_view(tmp_path):
    c = with_views(dataset(tmp_path), {"absolute": ABSOLUTE})
    top = ensure_prepared_optical_data(c)
    named = c.view_directory("absolute")
    stamps = {d: (d / "view.json").stat().st_mtime_ns for d in (top, named)}
    ensure_prepared_optical_data(c)
    assert {d: (d / "view.json").stat().st_mtime_ns for d in stamps} == stamps
    manifest = c.split.manifest.read_text()
    changed = with_views(c, {"absolute": {**ABSOLUTE, "normalization": {"method": "minmax"}}})
    ensure_prepared_optical_data(changed)
    assert (top / "view.json").stat().st_mtime_ns == stamps[top]
    assert (named / "view.json").stat().st_mtime_ns != stamps[named]
    assert c.split.manifest.read_text() == manifest
    # A view added later leaves the existing ones and their provenance untouched.
    before = OpticalView(top).provenance("train", "lf")
    more = with_views(changed, {**changed.model_dump()["views"], "direction": {
        "phi": {"momentum": {"representation": "direction_and_magnitude"}}}})
    ensure_prepared_optical_data(more)
    assert OpticalView(top).provenance("train", "lf") == before
    assert load_partition(more.view_directory("direction"), "train", "lf")[0].phi.shape[-1] == 7


def test_provider_selects_view_by_data_directory(tmp_path):
    c = with_views(dataset(tmp_path), {"absolute": ABSOLUTE})
    path = tmp_path / "data.yaml"
    import yaml

    path.write_text(yaml.safe_dump(c.model_dump(mode="json")))
    record = OpticalDataset.prepare(path, c.view_directory("absolute"))
    assert record["view"] == "views/absolute"
    assert OpticalDataset.prepare(path, c.output_directory)["view"] == "."
    with pytest.raises(ValueError, match="none of the views"):
        OpticalDataset.prepare(path, c.output_directory / "views" / "missing")


@pytest.mark.parametrize(
    "views",
    [
        {"Bad-Name": ABSOLUTE},
        {"cyl": {"theta": {"coordinates": "cylindrical",
                           "cylindrical": {"include_azimuth": False}}}},
        {"midrange": {"theta": {"center_source": "observed_midrange"}}},
        {"target": {"target": {"kind": "any_channel"}}},
    ],
)
def test_views_reject_names_grouping_and_shared_settings(tmp_path, views):
    with pytest.raises(ValueError):
        with_views(dataset(tmp_path, number=1), views)


EMISSION = {"theta": {"model_input": False}, **ABSOLUTE}


def test_event_only_view_keeps_voxel_rows_and_metadata(tmp_path):
    c = with_views(dataset(tmp_path), {"emission": EMISSION})
    top = ensure_prepared_optical_data(c)
    named = c.view_directory("emission")
    full, event = load_partition(top, "train", "lf")[0], load_partition(named, "train", "lf")[0]
    assert event.mode.value == "event_only" and event.theta is None
    assert event.phi.shape == full.phi.shape  # one row per voxel, as with theta
    assert np.array_equal(event.labels, full.labels)
    assert np.array_equal(event.events_per_design_point, full.events_per_design_point)
    # Voxel centres stay available as design-point metadata, normalized as in the theta view.
    base, provider = OpticalDataset(top), OpticalDataset(named)
    theta, ids = provider.theta("train", "lf")
    assert np.allclose(theta, full.theta) and ids == base.theta("train", "lf")[1]
    assert np.array_equal(provider.coordinates("train", "lf"), base.coordinates("train", "lf"))
    inputs = OpticalView(named).load_inputs("train", "lf")[0]
    assert inputs.theta is None and inputs.phi.shape == event.phi.shape
    metadata = json.loads((named / "metadata.json").read_text())["train"]["lf"]
    assert metadata["mode"] == "event_only"
    assert metadata["features"]["theta"]["model_input"] is False


def test_event_only_view_needs_phi_inputs(tmp_path):
    omit = {"vertex": {"representation": "omit"}, "momentum": {"representation": "omit"}}
    with pytest.raises(ValueError, match="needs phi inputs"):
        with_views(dataset(tmp_path, number=1), {"none": {"theta": {"model_input": False},
                                                          "phi": omit}})


def test_mlp_bll_trains_and_predicts_on_event_only_view(tmp_path):
    import torch
    from core.surrogates.experiment import run_experiment
    from core.surrogates.inference import predict_partition
    from schemas.surrogates import SurrogateRunConfig

    from tests.test_optical_pipeline import role_config, role_folder
    from tests.test_unlabeled_inputs import strip_hits

    torch.set_num_threads(1)
    labeled = role_folder(tmp_path / "labeled", 6)
    unlabeled = strip_hits(role_folder(tmp_path / "inputs", 4, offset=1.0, first_run=10))
    c = role_config(
        tmp_path,
        [
            {"path": str(labeled), "fractions": {"cnp_train": 0.5, "cnp_validation": 0.5}},
            {"path": str(unlabeled), "fractions": {"gp_unlabeled": 1.0}},
        ],
        views={"emission": EMISSION},
    )
    ensure_prepared_optical_data(c)
    data = c.view_directory("emission")
    config = SurrogateRunConfig.model_validate({
        "model": {"kind": "mlp", "architecture": {"hidden_dims": [8]},
                  "last_layer": {"prior_precision": "evidence", "draws": 100, "seed": 0}},
        "training": {"backend": "neural", "loss": "bernoulli", "n_steps": 2, "eval_every": 1,
                     "batch_size": 1, "n_events": 4, "seed": 0, "device": "cpu"},
        "data_directory": str(data),
        "data_provider": "legend_optical_map_ml.provider:OpticalDataset",
        "partitions": {"train": "cnp_train", "validation": "cnp_validation", "test": "cnp_test"},
        "output_directory": str(tmp_path / "run"),
        "hf_validation": False,
    })
    output = run_experiment(config)
    metrics = json.loads((output / "metrics.json").read_text())
    validation = [m for m in metrics if m["split"] == "validation"]
    assert validation and all(m["design_points"] == 3 for m in validation)
    # Context-free inference on voxels the model never saw, with voxel ids and centres.
    path = predict_partition(output / "checkpoints" / "best", data, "gp_unlabeled")
    provider = OpticalDataset(data)
    with np.load(path) as arrays:
        p = arrays["mean_probability"]
        assert p.shape == (4,) and np.all((p > 0) & (p < 1))
        assert arrays["design_point_ids"].tolist() == provider.theta("gp_unlabeled", "lf")[1]
        assert np.array_equal(arrays["coordinates"], provider.coordinates("gp_unlabeled", "lf"))
