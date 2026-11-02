"""CSV manifests split across roles by fractions or counts."""

import csv
import json

import pytest

from legend_optical_map_ml.config import ManifestSource, OpticalDataConfig
from legend_optical_map_ml.pipeline import prepare_optical_data
from tests.test_optical_pipeline import fixture_file

h5py = pytest.importorskip("h5py")


def write_manifest(path, files, tier):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stp_file", "tier"])
        writer.writeheader()
        for file in files:
            writer.writerow({"stp_file": file.relative_to(path.parent).as_posix(), "tier": tier})
    return path


def config(tmp_path, split, seed=3):
    root = tmp_path / "sims"
    train = [fixture_file(root, "lf", i, (0.05 * i, 0.1, 0.2)) for i in range(4)]
    validation = [fixture_file(root, "hf", 10 + i, (1.0 + 0.05 * i, 0.1, 0.2)) for i in range(10)]
    return OpticalDataConfig.model_validate({
        "source": {"manifests": [
            {"path": write_manifest(tmp_path / "train.csv", train, "lf"), "role": "cnp_train"},
            {"path": write_manifest(tmp_path / "val.csv", validation, "hf"), **split},
        ]},
        "output_directory": tmp_path / "prepared",
        "split": {"manifest": tmp_path / "split.json", "seed": seed},
    })


def roles(c):
    files = json.loads(c.split.manifest.read_text())["files"]
    return {r["file"]: r["split"] for r in files}


def test_manifest_rows_split_by_fractions_and_counts(tmp_path):
    c = config(tmp_path, {"fractions": {"gp_validation": 0.5, "gp_test": 0.5}})
    prepare_optical_data(c, workers=1)
    split = roles(c)
    hf = [role for name, role in split.items() if name.startswith("hf/")]
    assert sorted(hf) == ["gp_test"] * 5 + ["gp_validation"] * 5
    assert all(role == "cnp_train" for name, role in split.items() if name.startswith("lf/"))
    assert json.loads(c.split.manifest.read_text())["settings"]["seed"] == 3
    # Same seed, same assignment; another seed reshuffles.
    again = config(tmp_path / "b", {"fractions": {"gp_validation": 0.5, "gp_test": 0.5}})
    prepare_optical_data(again, workers=1)
    assert sorted(roles(again).values()) == sorted(split.values())
    counted = config(tmp_path / "c", {"counts": {"gp_validation": 3, "gp_test": "rest"}})
    prepare_optical_data(counted, workers=1)
    hf = [role for name, role in roles(counted).items() if name.startswith("hf/")]
    assert sorted(hf) == ["gp_test"] * 7 + ["gp_validation"] * 3


def test_single_role_manifests_keep_their_settings(tmp_path):
    c = config(tmp_path, {"role": "gp_validation"})
    prepare_optical_data(c, workers=1)
    assert set(roles(c).values()) == {"cnp_train", "gp_validation"}
    assert "seed" not in json.loads(c.split.manifest.read_text())["settings"]


def test_manifest_needs_exactly_one_assignment(tmp_path):
    with pytest.raises(ValueError, match="exactly one of role, fractions or counts"):
        ManifestSource(path=tmp_path / "a.csv")
    with pytest.raises(ValueError, match="exactly one of role, fractions or counts"):
        ManifestSource(path=tmp_path / "a.csv", role="gp_test", fractions={"gp_test": 1.0})
    with pytest.raises(ValueError, match="sum to 1"):
        ManifestSource(path=tmp_path / "a.csv", fractions={"gp_test": 0.4})
