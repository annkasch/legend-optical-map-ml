"""Repository configs: configs/<dataset>/*.yaml load and write into outputs/<dataset>/."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"
# Known not to load: observation_noise is not (yet) an MFGP config key.
KNOWN_BROKEN = {"mfgp.log.hetnoise.yaml", "mfgp.raw_lf.log.hetnoise.yaml"}


def config_files():
    return sorted(CONFIGS.glob("*/*.yaml"))


def load(path: Path):
    """Load a repository config with the loader its name and keys select."""
    from schemas.surrogates import (
        load_binomial_config,
        load_mfgp_config,
        load_stream_config,
        load_surrogate_config,
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
    if path.name in KNOWN_BROKEN:
        with pytest.raises(ValueError, match="observation_noise"):
            load(path)
        return
    config = load(path)
    dataset = OUTPUTS / path.parent.name
    if path.name in ("data.yaml", "photons.yaml"):
        target = config.output_directory if path.name == "data.yaml" else config.store
        assert target == dataset / "data"
        return
    if config is None or path.name == "benchmark.yaml":
        return
    out = config.output_directory.relative_to(dataset)  # a run goes with its training data
    assert re.fullmatch(r"(event_learning|regression)/[^/]+/v\d+/seed\d+", str(out)), out
    if hasattr(config, "data_config") and config.data_config is not None:
        from legend_optical_map_ml.config import load_optical_config

        assert load_optical_config(config.data_config).output_directory == config.data_directory
