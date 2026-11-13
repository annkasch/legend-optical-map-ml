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
    if path.name.startswith(("within_point", "residual")):  # added evaluations of a run
        run = config.run.relative_to(dataset)
        assert re.fullmatch(r"(event_learning|regression)/[^/]+/v\d+/seed\d+", str(run)), run
        return
    out = config.output_directory.relative_to(dataset)  # a run goes with its training data
    assert re.fullmatch(r"(event_learning|regression)/[^/]+/v\d+/seed\d+", str(out)), out
    if hasattr(config, "data_config") and config.data_config is not None:
        from legend_optical_map_ml.config import load_optical_config

        views = load_optical_config(config.data_config).view_configs().values()
        assert config.data_directory in [v.output_directory for v in views]
