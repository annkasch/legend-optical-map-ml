"""Photon store (conversion, splits, moments) and its training loader on synthetic chunks."""

import json
import os

import numpy as np
import pytest

from legend_optical_map_ml.config import PhotonStoreConfig, StreamTransformConfig
from legend_optical_map_ml.features import stream_features, stream_features_torch
from legend_optical_map_ml.photon_loader import PhotonStream, labels
from legend_optical_map_ml.photon_store import (
    build,
    load_manifest,
    normalization,
    shards,
    split_of,
)

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")


def chunk(root, index, n=40, detected=(1, 5, 5, 9)):
    """A homogeneous-campaign chunk (secondment layout); px = unique photon id."""
    path = root / f"chunk{index:05d}.stp.lh5"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(index)
    ids = np.arange(n, dtype=np.int32)[::-1]
    with h5py.File(path, "w") as h:
        h["number_of_simulated_events"] = n

        def ds(name, values, units=None):
            d = h.create_dataset(name, data=values)
            if units:
                d.attrs["units"] = units

        ds("vtx/evtid", ids)
        for axis in "xyz":
            ds(f"vtx/{axis}loc", rng.uniform(-1, 1, n).astype(np.float32), "m")
        ds("vtx/time", np.zeros(n), "ns")
        ds("vtx/n_part", np.ones(n, dtype=np.int32))
        ds("particles/evtid", ids[::-1].copy())
        ds("particles/px", (index * 1000 + ids[::-1]).astype(np.float32), "MeV")
        for axis in "yz":
            ds(f"particles/p{axis}", rng.normal(size=n).astype(np.float32), "MeV")
        hits = np.array(detected, dtype=np.int32)
        ds("stp/optical/evtid", hits)
        ds("stp/optical/det_uid", np.arange(len(hits), dtype=np.int32) % 2 + 7)
        ds("stp/optical/time", np.zeros(len(hits)), "ns")
        ds("stp/optical/wavelength", np.full(len(hits), 128.0), "nm")
    return path


def raw_photons(paths):
    """Expected records: (id, x, y, z, px, py, pz, label) per photon, sorted by id."""
    rows = []
    for path in paths:
        with h5py.File(path, "r") as h:
            ev = h["vtx/evtid"][:]
            pos = np.column_stack([h[f"vtx/{a}loc"][:] for a in "xyz"])
            pev = h["particles/evtid"][:]
            mom = np.column_stack([h[f"particles/p{a}"][:] for a in "xyz"])
            mom = mom[np.argsort(pev)][np.searchsorted(np.sort(pev), ev)]
            label = np.isin(ev, h["stp/optical/evtid"][:]).astype(np.float32)
        rows.append(np.column_stack([mom[:, 0], pos, mom, label]))
    rows = np.concatenate(rows)
    return rows[np.argsort(rows[:, 0])]


def store_config(tmp_path, **updates):
    return PhotonStoreConfig.model_validate(
        {
            "sources": [{"path": str(tmp_path / "box" / "results"), "name": "box"}],
            "store": str(tmp_path / "store"),
            "shard_photons": 100,
            "split": {"seed": 3, "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2}},
            **updates,
        }
    )


def stored(store, split=None):
    """Stored records float32 [N, 7] of a split (all splits if None)."""
    manifest = load_manifest(store)
    parts = []
    for name in [split] if split else ("train", "validation", "test"):
        for shard in shards(manifest, name):
            parts.append(np.fromfile(store / shard["records"], dtype=np.float32).reshape(-1, 7))
    return np.concatenate(parts)


def as_rows(records):
    """Records [N, 7] -> (id, x, y, z, px, py, pz, label) sorted by id (px = id)."""
    rows = np.column_stack([records[:, 3], records[:, :6], labels(records).astype(np.float32)])
    return rows[np.argsort(rows[:, 0])]


@pytest.fixture
def campaign(tmp_path):
    paths = [chunk(tmp_path / "box" / "results", i) for i in range(12)]
    config = store_config(tmp_path)
    build(config, workers=2)
    return config, paths


def test_store_keeps_every_photon_bit_exact(campaign):
    config, paths = campaign
    np.testing.assert_array_equal(as_rows(stored(config.store)), raw_photons(paths))
    manifest = load_manifest(config.store)
    assert sorted(manifest["files"]) == [f"box/chunk{i:05d}.stp.lh5" for i in range(12)]
    for shard in manifest["shards"].values():
        records = np.fromfile(config.store / shard["records"], dtype=np.float32).reshape(-1, 7)
        hits = np.load(config.store / shard["hits"])
        assert shard["n_photons"] == len(records) and shard["n_hits"] == len(hits)
        # Hit rows follow the permutation: every hit row is a detected photon and vice versa.
        assert set(hits["row"].tolist()) == set(np.flatnonzero(labels(records)).tolist())
        assert len(hits) == 4 * len(shard["sources"])  # pairs (1,7) (5,7) (5,8) (9,8) per file
        assert shard["n_photons"] <= config.shard_photons + 40
        assert shard["code"]["packages"]["h5py"] and "dirty" in shard["code"]["resum_flex"]


def test_photons_are_permuted_inside_shards(campaign):
    config, _ = campaign
    records = stored(config.store, "train")
    ids = records[:, 3]
    assert not np.all(np.diff(ids) > 0)


def test_splits_follow_file_keys_and_stay_when_files_are_added(tmp_path, campaign):
    config, _ = campaign
    manifest = load_manifest(config.store)
    before = {k: (v["shard"], v["split"]) for k, v in manifest["files"].items()}
    for key, (_, split) in before.items():
        assert split == split_of(key, config.split)
    shard_files = {
        name: (config.store / s["records"]).stat().st_mtime_ns
        for name, s in manifest["shards"].items()
    }
    for i in range(12, 16):
        chunk(tmp_path / "box" / "results", i)
    result = build(config)
    manifest = load_manifest(config.store)
    assert result["files"] == 16 and result["new_shards"] >= 1
    for key, value in before.items():
        assert (manifest["files"][key]["shard"], manifest["files"][key]["split"]) == value
    for name, mtime in shard_files.items():  # existing shards untouched
        assert (config.store / manifest["shards"][name]["records"]).stat().st_mtime_ns == mtime
    assert build(config)["new_shards"] == 0


def test_changed_source_file_and_changed_settings_are_errors(tmp_path, campaign):
    config, paths = campaign
    os.utime(paths[0], ns=(1, 1))
    with pytest.raises(ValueError, match="changed after it was stored"):
        build(config)
    with pytest.raises(ValueError, match="fixed once the store exists"):
        build(store_config(tmp_path, shard_photons=50))


def test_normalization_equals_direct_moments_of_train_photons(campaign):
    config, _ = campaign
    manifest = load_manifest(config.store)
    columns = stored(config.store, "train")[:, :6].astype(np.float64)
    norm = normalization(manifest)
    np.testing.assert_allclose(norm["offset"], columns.mean(0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(norm["scale"], columns.std(0), rtol=1e-10)
    assert norm["n_photons"] == len(columns)


def collect(stream, batch_size, pass_index, **kwargs):
    batches = list(stream.batches(batch_size, pass_index, **kwargs))
    return batches, torch.cat(batches).numpy() if batches else np.zeros((0, 7), np.float32)


@pytest.mark.parametrize("in_memory", [False, True])
def test_every_photon_once_per_pass_and_order_changes(campaign, in_memory):
    config, _ = campaign
    stream = PhotonStream(
        config.store, "train", block_photons=7, buffer_photons=20, in_memory=in_memory, threads=2
    )
    expected = as_rows(stored(config.store, "train"))
    orders = []
    for p in range(3):
        batches, data = collect(stream, 16, p)
        assert len(batches) == stream.n_batches(16, p)
        assert all(len(b) == 16 for b in batches[:-1])
        np.testing.assert_array_equal(as_rows(data), expected)
        orders.append(data[:, 3])
    assert not np.array_equal(orders[0], orders[1])
    _, again = collect(stream, 16, 1)
    np.testing.assert_array_equal(again[:, 3], orders[1])  # deterministic per pass


@pytest.mark.parametrize("in_memory", [False, True])
def test_ranks_are_disjoint_and_equal_in_batches(campaign, in_memory):
    config, _ = campaign
    world = 3
    seen, counts = [], []
    for rank in range(world):
        stream = PhotonStream(
            config.store,
            "train",
            rank=rank,
            world_size=world,
            block_photons=5,
            buffer_photons=15,
            in_memory=in_memory,
        )
        batches, data = collect(stream, 8, 2)
        assert all(len(b) == 8 for b in batches)
        counts.append(len(batches))
        seen.append(data[:, 3])
    assert len(set(counts)) == 1 and counts[0] > 0
    ids = np.concatenate(seen)
    assert len(np.unique(ids)) == len(ids)
    assert set(ids.tolist()) <= set(stored(config.store, "train")[:, 3].tolist())


def test_resume_reproduces_the_remaining_batches(campaign):
    config, _ = campaign
    stream = PhotonStream(config.store, "train", block_photons=7, buffer_photons=20)
    full, _ = collect(stream, 6, 4)
    for start in (0, 1, 5, len(full) - 1):
        resumed, _ = collect(stream, 6, 4, start_batch=start)
        assert len(resumed) == len(full) - start
        for a, b in zip(resumed, full[start:], strict=True):
            np.testing.assert_array_equal(a.numpy(), b.numpy())


def test_load_and_early_stop(campaign):
    config, _ = campaign
    stream = PhotonStream(config.store, "validation", block_photons=7, buffer_photons=14)
    expected = as_rows(stored(config.store, "validation"))
    np.testing.assert_array_equal(as_rows(stream.load()), expected)
    iterator = stream.batches(4)
    next(iterator)
    iterator.close()  # stops the reader thread without hanging
    info = json.loads((config.store / "manifest.json").read_text())
    assert info["record"]["columns"] == ["x", "y", "z", "px", "py", "pz"]


# -- input transform ----------------------------------------------------------------------------

CYLINDRICAL = {"phi": {"vertex": {"representation": "absolute", "coordinates": "cylindrical"},
                       "momentum": {"coordinates": "cylindrical"}}}
TRANSFORMS = [
    CYLINDRICAL,
    {**CYLINDRICAL, "cylindrical": {"rotational_symmetry": 6}},
    {**CYLINDRICAL, "cylindrical": {"rotational_symmetry": 6, "angle_weighting": "radius"}},
    {**CYLINDRICAL, "cylindrical": {"angle_encoding": "radians", "rotational_symmetry": 6}},
    {"phi": {"vertex": {"representation": "absolute"}, "momentum": {
        "representation": "direction_and_magnitude", "coordinates": "cylindrical"}}},
    {"phi": {"vertex": {"representation": "omit"}, "momentum": {"representation": "components"}}},
]


def photons(n=500, seed=0):
    """Raw records [n, 7] incl. points on the axis and at the +-180 degree wrap."""
    rng = np.random.default_rng(seed)
    records = np.empty((n, 7), dtype=np.float32)
    records[:, :3] = rng.uniform(-1, 1, (n, 3))
    records[:, 3:6] = rng.normal(size=(n, 3)) * 1e-5
    records[:3, :2] = [[0, 0], [-0.7, 1e-7], [-0.7, -1e-7]]
    records.view(np.uint32)[:, 6] = rng.integers(0, 4, n)
    return records


@pytest.mark.parametrize("settings", TRANSFORMS)
def test_torch_transform_matches_numpy_reference(settings):
    transform = StreamTransformConfig.model_validate(settings)
    records = photons()
    expected, names, _ = stream_features(records, transform)
    got = stream_features_torch(torch.from_numpy(records), transform).numpy()
    assert got.shape == expected.shape == (len(records), len(names))
    rows = slice(3, None) if transform.cylindrical.angle_encoding == "radians" else slice(None)
    np.testing.assert_allclose(got[rows], expected[rows], rtol=1e-6, atol=1e-6)  # float32


def test_six_fold_transform_repeats_every_60_degrees():
    transform = StreamTransformConfig.model_validate(TRANSFORMS[2])
    records = photons()
    a = np.radians(60)
    rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    turned = records.copy()
    turned[:, :3] = records[:, :3] @ rotation.T
    turned[:, 3:6] = records[:, 3:6] @ rotation.T
    same = [
        stream_features_torch(torch.from_numpy(r), transform).numpy() for r in (records, turned)
    ]
    # Row 0 lies on the axis: azimuth undefined (taken as 0), so its momentum frame cannot turn.
    np.testing.assert_allclose(same[1][1:], same[0][1:], rtol=1e-5, atol=1e-6)


def test_transform_settings_are_checked():
    for bad in ({"phi": {"vertex": {"representation": "offset"}}},
                {"phi": {"vertex": {"representation": "omit"},
                         "momentum": {"coordinates": "cylindrical"}}},
                {"cylindrical": {"angle_encoding": "radians", "angle_weighting": "radius"}}):
        with pytest.raises(ValueError):
            StreamTransformConfig.model_validate(bad)


@pytest.mark.parametrize("in_memory", [False, True])
def test_loader_transforms_batches_load_and_normalization(campaign, in_memory):
    config, _ = campaign
    settings = {**CYLINDRICAL, "cylindrical": {"rotational_symmetry": 6}}
    stream = PhotonStream(config.store, "train", block_photons=7, buffer_photons=20,
                          in_memory=in_memory, transform=settings)
    plain = PhotonStream(config.store, "train", block_photons=7, buffer_photons=20,
                         in_memory=in_memory)
    names, units = stream.columns
    assert names == ["r", "sin_azimuth", "cos_azimuth", "z", "p_r_local", "p_t_local", "pz"]
    batches, _ = collect(plain, 16, 1)
    transformed = list(stream.batches(16, 1))
    assert [len(b) for b in transformed] == [len(b) for b in batches]
    for raw, got in zip(batches, transformed, strict=True):  # same photons, same order
        assert got.shape[1] == len(names) + 1
        np.testing.assert_array_equal(labels(got.numpy()), labels(raw.numpy()))
        np.testing.assert_allclose(got[:, :-1].numpy(), stream._transformed(raw.numpy())[:, :-1],
                                   rtol=1e-6, atol=1e-6)
    # load(): the numpy reference of every photon of the split, labels unchanged.
    raw = plain.load()
    loaded = stream.load()
    np.testing.assert_array_equal(labels(loaded), labels(raw))
    np.testing.assert_array_equal(loaded[:, :-1], stream_features(raw, stream.transform)[0])
    # Normalization: moments of the transformed train photons (all of them: sample > store).
    norm = stream.normalization()
    values = stream_features(stored(config.store, "train"), stream.transform)[0].astype(float)
    np.testing.assert_allclose(norm["offset"], values.mean(0), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(norm["scale"], values.std(0), rtol=1e-6)
    assert norm["columns"] == names and norm["n_photons"] == len(values)
    assert plain.normalization()["columns"] == ["x", "y", "z", "px", "py", "pz"]


def test_transform_normalization_samples_the_first_photons_of_every_train_shard(campaign):
    config, _ = campaign
    stream = PhotonStream(config.store, "validation", transform=CYLINDRICAL,
                          normalization_photons=10)
    train = shards(load_manifest(config.store), "train")
    per_shard = -(-10 // len(train))
    norm = stream.normalization()
    assert norm["n_photons"] == sum(min(per_shard, s["n_photons"]) for s in train)
    sample = np.concatenate([
        np.fromfile(config.store / s["records"], dtype=np.float32).reshape(-1, 7)[:per_shard]
        for s in train
    ])
    values = stream_features(sample, stream.transform)[0].astype(float)
    np.testing.assert_allclose(norm["offset"], values.mean(0), rtol=1e-6, atol=1e-9)
