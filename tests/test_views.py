"""Named data views: one split, own features and normalization, selected by data_directory."""

import json

import numpy as np
import pytest

from legend_optical_map_ml.config import OpticalDataConfig
from legend_optical_map_ml.features import decode_absolute_phi
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


FIELD = {"kind": "field", "architecture": {"hidden_dims": [8]},
         "encoding": {"kind": "fourier", "columns": [0, 1, 2], "n_frequencies": 2,
                      "max_frequency": 1.0},
         "direction": {"phi_columns": [3, 4, 5], "degree": 1}}


@pytest.mark.parametrize("kind", ["mlp", "field"])
def test_mlp_bll_trains_and_predicts_on_event_only_view(tmp_path, kind):
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
        "model": {**({"kind": "mlp", "architecture": {"hidden_dims": [8]}} if kind == "mlp"
                     else FIELD),  # field: the absolute emission point encoded, θ unused
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
        # The voxel centres GP stages check (binomial covariate, MFGP level), without theta input.
        np.testing.assert_allclose(
            arrays["theta"], provider.theta("gp_unlabeled", "lf")[0], rtol=0, atol=1e-12
        )
    # Score stage: donors from the scored set itself, never the voxel's own, moved into the voxel.
    from core.surrogates.score_stage import run_score
    from schemas.surrogates import ScoreRunConfig

    scored = run_score(ScoreRunConfig(
        run=output, sets=["cnp_validation/lf"], donors="scored", seed=1,
        output_directory=output / "score_cnp_validation_lf",
    ))
    (row,) = json.loads((scored / "score.json").read_text())["rows"]
    record = row["event_inputs"]
    assert record["own_excluded"] and record["transfer"] == "provider transfer_events"
    assert all(d != i for i, d in enumerate(record["donor_index"]))
    assert np.isfinite(row["predictive_nll"]) and row["design_points"] == 3


def test_transfer_moves_absolute_donor_positions_into_the_target_voxel(tmp_path):
    c = with_views(dataset(tmp_path), {"absolute": ABSOLUTE})
    top = ensure_prepared_optical_data(c)
    offset, absolute = OpticalDataset(top), OpticalDataset(c.view_directory("absolute"))
    source, target = ("train", "lf"), ("validation", "lf")
    donor_abs = absolute.load(*source)[0].phi
    n_target = absolute.load(*target)[0].batch_size
    rows = np.arange(n_target) % donor_abs.shape[0]
    moved = absolute.transfer_events(donor_abs[rows], source, rows, target)
    # Physical positions: moved - target centre equals the donor's offset from its own centre.
    def physical(provider, phi):
        norm = provider.normalization()["phi"]
        return phi * np.asarray(norm["scale"]) + np.asarray(norm["offset"])

    counts = absolute.load(*source)[0].events_per_design_point[rows]
    valid = np.arange(moved.shape[1]) < counts[:, None]  # [B, N]
    relative = physical(absolute, moved)[..., :3] - absolute.coordinates(*target)[:, None, :]
    donor_offsets = physical(offset, offset.load(*source)[0].phi[rows])[..., :3]
    np.testing.assert_allclose(relative[valid], donor_offsets[valid], atol=1e-6)
    np.testing.assert_array_equal(moved[..., 3:], donor_abs[rows][..., 3:])  # momenta unchanged
    # Offset views need no move.
    phi = offset.load(*source)[0].phi[rows]
    assert offset.transfer_events(phi, source, rows, target) is phi


CYLINDRICAL_ABSOLUTE = {"phi": {
    "vertex": {"representation": "absolute", "coordinates": "cylindrical"},
    "momentum": {"coordinates": "cylindrical"},
}}


@pytest.mark.parametrize("symmetry", [1, 6])
def test_transfer_moves_cylindrical_absolute_donors_like_cartesian_ones(tmp_path, symmetry):
    cyl = {**CYLINDRICAL_ABSOLUTE, "theta": {"cylindrical": {"rotational_symmetry": symmetry}}}
    c = with_views(dataset(tmp_path), {"absolute": ABSOLUTE, "cyl": cyl})
    ensure_prepared_optical_data(c)
    cart, cyl = (OpticalDataset(c.view_directory(n)) for n in ("absolute", "cyl"))
    source, target = ("train", "lf"), ("validation", "lf")
    n_target = cyl.load(*target)[0].batch_size
    rows = np.arange(n_target) % cyl.load(*source)[0].phi.shape[0]
    counts = cyl.load(*source)[0].events_per_design_point[rows]

    def decoded(provider, phi):  # physical cartesian (positions, momenta) [B, N, 3] each
        norm = provider.normalization()["phi"]
        raw = phi * np.asarray(norm["scale"]) + np.asarray(norm["offset"])
        b, n, d = raw.shape
        near = np.repeat(provider.coordinates(*target), n, axis=0)  # moved into the target voxels
        pos, mom = decode_absolute_phi(raw.reshape(b * n, d), provider.view.config, near=near)
        return pos.reshape(b, n, 3), mom.reshape(b, n, 3)

    moved = [p.transfer_events(p.load(*source)[0].phi[rows], source, rows, target)
             for p in (cart, cyl)]
    valid = np.arange(moved[0].shape[1]) < counts[:, None]
    if symmetry > 1:  # sector unknown for voxels on the axis (fixture voxel 0): not compared
        off_axis = [np.hypot(*cyl.coordinates(*s)[i, :2].T) > 0.01 for s, i in
                    ((source, rows), (target, slice(None)))]
        valid &= (off_axis[0] & off_axis[1])[:, None]
        assert valid.any()
    (pc, mc), (py, my) = decoded(cart, moved[0]), decoded(cyl, moved[1])
    np.testing.assert_allclose(py[valid], pc[valid], atol=1e-5)
    np.testing.assert_allclose(my[valid], mc[valid], rtol=1e-4, atol=1e-6)
