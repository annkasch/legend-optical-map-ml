"""Repository configs: configs/<dataset>/*.yaml load and write into outputs/<dataset>/."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"


def config_files():
    return sorted(CONFIGS.glob("*/*.yaml"))


def load(path: Path):
    """Load a repository config with the loader its name and keys select."""
    from schemas.surrogates import (
        load_binomial_config,
        load_mfgp_config,
        load_residual_config,
        load_score_config,
        load_stream_config,
        load_surrogate_config,
        load_within_point_config,
    )

    from legend_optical_map_ml.benchmark import load_benchmark_config
    from legend_optical_map_ml.config import load_optical_config, load_photon_store_config

    name = path.name
    if name == "data.yaml":
        return load_optical_config(path)
    if name == "photons.yaml":
        return load_photon_store_config(path)
    if name == "benchmark.yaml":
        return load_benchmark_config(path)
    if name.startswith("countfield"):
        return None  # read by the countfield repository
    if name.startswith("binomial"):
        return load_binomial_config(path)
    if name.startswith("within_point"):
        return load_within_point_config(path)
    if name.startswith("residual"):
        return load_residual_config(path)
    if name.startswith("score"):
        return load_score_config(path)
    if name.startswith("mfgp"):
        return load_mfgp_config(path)
    if "event_store" in yaml.safe_load(path.read_text()):
        return load_stream_config(path)
    return load_surrogate_config(path)


def test_configs_are_grouped_by_dataset():
    assert not list(CONFIGS.glob("*.yaml")), "configs belong in configs/<dataset>/"
    for folder in CONFIGS.iterdir():
        assert re.fullmatch(r"(voxel_sets|photon)_x.+", folder.name), folder


@pytest.mark.parametrize("path", config_files(), ids=lambda p: f"{p.parent.name[:14]}/{p.name}")
def test_config_loads_and_writes_into_its_dataset(path):
    config = load(path)
    dataset = OUTPUTS / path.parent.name
    if path.name in ("data.yaml", "photons.yaml"):
        target = config.output_directory if path.name == "data.yaml" else config.store
        assert target == dataset / "data"
        return
    if config is None or path.name == "benchmark.yaml":
        return
    if path.name.startswith(("within_point", "residual", "score")):  # added evaluations of a run
        run = config.run.relative_to(dataset)
        assert re.fullmatch(r"(event_learning|regression)/[^/]+/v\d+/seed\d+", str(run)), run
        return
    out = config.output_directory.relative_to(dataset)  # a run goes with its training data
    assert re.fullmatch(r"(event_learning|regression)/[^/]+/v\d+/seed\d+", str(out)), out
    if hasattr(config, "data_config") and config.data_config is not None:
        from legend_optical_map_ml.config import load_optical_config

        views = load_optical_config(config.data_config).view_configs().values()
        assert config.data_directory in [v.output_directory for v in views]


def stream_transform_mismatch(config):
    """Message if a stream config's loader transform differs from its evaluation view, else None.

    The loader's ``transform`` (event_source_options) must build the same phi columns as the
    view at ``data_directory``; RESUM_FLEX maps the view's phi columns one by one into the
    stream units and does not check their meaning.
    """
    from legend_optical_map_ml.config import StreamTransformConfig, load_optical_config

    options = getattr(config, "event_source_options", None) or {}
    if getattr(config, "data_config", None) is None:
        return None
    views = load_optical_config(config.data_config).view_configs().values()
    view = next(v for v in views if v.output_directory == config.data_directory)
    transform = StreamTransformConfig.model_validate(options.get("transform") or {
        # untransformed records: absolute cartesian x, y, z, px, py, pz
        "phi": {"vertex": {"representation": "absolute"}},
    })
    if options.get("transform") is None and view.phi.vertex.representation != "absolute":
        return None  # raw stream evaluated on an offset view (before the RESUM_FLEX change)
    if transform.matches(view):
        return None
    return f"{config.data_directory}: view phi {view.phi} / {view.theta.cylindrical} != {transform}"


@pytest.mark.parametrize(
    "path",
    [p for p in config_files() if "event_store" in yaml.safe_load(p.read_text())],
    ids=lambda p: f"{p.parent.name[:14]}/{p.name}",
)
def test_stream_transform_equals_evaluation_view(path):
    assert stream_transform_mismatch(load(path)) is None


def test_stream_transform_mismatch_is_detected(tmp_path):
    from types import SimpleNamespace

    from legend_optical_map_ml.config import OpticalDataConfig

    data = tmp_path / "data.yaml"
    cylindrical = {"vertex": {"representation": "absolute", "coordinates": "cylindrical"},
                   "momentum": {"coordinates": "cylindrical"}}
    config = OpticalDataConfig.model_validate({
        "source": {"directory": str(tmp_path / "raw")},
        "output_directory": str(tmp_path / "out"),
        "theta": {"model_input": False, "cylindrical": {"rotational_symmetry": 6}},
        "phi": cylindrical,
        "views": {"cartesian": {"phi": {"vertex": {"representation": "absolute"}}}},
    })
    data.write_text(yaml.safe_dump(config.model_dump(mode="json", exclude_none=True)))
    transform = {"cylindrical": {"rotational_symmetry": 6}, "phi": cylindrical}

    def run(directory, settings):
        return SimpleNamespace(data_config=data, data_directory=directory,
                               event_source_options={"transform": settings})

    assert stream_transform_mismatch(run(tmp_path / "out", transform)) is None
    assert stream_transform_mismatch(run(tmp_path / "out" / "views" / "cartesian", transform))
    assert stream_transform_mismatch(run(tmp_path / "out", {**transform, "cylindrical": {}}))


def test_cnp_bll_differs_from_its_mlp_bll_only_in_the_model():
    """cnp_bll.theta_absolute.yaml: same data, training and selection as mlp_bll (b)."""
    from schemas.surrogates import load_surrogate_config

    folder = CONFIGS / "voxel_sets_x+300+500_y-100+100_z-100+100_5mm"
    cnp = load_surrogate_config(folder / "cnp_bll.theta_absolute.yaml")
    mlp = load_surrogate_config(folder / "mlp_bll.theta_absolute.yaml")
    a, b = cnp.model_dump(mode="json"), mlp.model_dump(mode="json")
    assert {k for k in a if a[k] != b[k]} == {
        "model", "output_directory", "validation_context_events", "validation_context_sweep"
    }
    assert cnp.model.kind == "cnp" and cnp.model.last_layer == mlp.model.last_layer
    assert max(cnp.validation_context_sweep) <= cnp.training.context_range()[1]


def test_cnp_bll_mixup_differs_from_cnp_bll_only_in_the_mixup_sampling():
    """cnp_bll.mixup.theta_absolute.yaml: pure mixup, real context, ranking selection."""
    from schemas.surrogates import load_surrogate_config

    folder = CONFIGS / "voxel_sets_x+300+500_y-100+100_z-100+100_5mm"
    mix = load_surrogate_config(folder / "cnp_bll.mixup.theta_absolute.yaml")
    cnp = load_surrogate_config(folder / "cnp_bll.theta_absolute.yaml")
    a, b = mix.model_dump(mode="json"), cnp.model_dump(mode="json")
    assert {k for k in a if a[k] != b[k]} == {"training", "selection", "output_directory"}
    ta, tb = a["training"], b["training"]
    assert {k for k in ta if ta[k] != tb[k]} == {"sampling"}
    assert ta["objective"]["strategy"] == "single"
    assert ta["sampling"]["strategy"] == "class_aware_mixup"
    assert ta["sampling"]["mixup"]["mix_context"] is False
    assert mix.selection == "average_precision"


def test_cnp_bll_mixup_recalibrated_refits_only_the_output_layer_of_the_mixup_run():
    """cnp_bll.mixup.recalibrated.theta_absolute.yaml: v1 best, output layer on real photons."""
    from schemas.surrogates import load_surrogate_config

    folder = CONFIGS / "voxel_sets_x+300+500_y-100+100_z-100+100_5mm"
    mix = load_surrogate_config(folder / "cnp_bll.mixup.theta_absolute.yaml")
    rec = load_surrogate_config(folder / "cnp_bll.mixup.recalibrated.theta_absolute.yaml")
    assert rec.model == mix.model
    assert rec.initial_checkpoint == mix.output_directory / "checkpoints" / "best"
    assert rec.output_directory.parent.name == "v2" and mix.output_directory.parent.name == "v1"
    [stage] = rec.stages
    assert (stage.start_from, stage.trainable) == ("initial", "output_layer")
    assert stage.training == rec.training
    assert stage.training.sampling.strategy == "natural"
    assert stage.training.objective.strategy == "single"
    assert stage.selection == rec.selection == "bernoulli_log_loss"
    a, b = rec.model_dump(mode="json"), mix.model_dump(mode="json")
    assert {k for k in a if a[k] != b[k]} == {
        "training", "selection", "stages", "initial_checkpoint", "output_directory"
    }
