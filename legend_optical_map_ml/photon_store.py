"""Photon store: every photon of ungrouped (e.g. homogeneous) LH5 campaigns, for event models.

Unlike the event cache, files are not voxels: each photon is one training example. The store
keeps every photon (none are dropped) as fixed 28-byte records, in exactly the precision
remage writes::

    <store>/manifest.json                   settings, source files, shards, moments
    <store>/shards/<split>/s000042.bin      RECORD rows, photons in random order
    <store>/shards/<split>/s000042.hits.npy HIT_DTYPE (row in shard, detector, hits)
    <store>/shards/<split>/s000042.json     shard summary; written last, marks the shard complete
    <store>/tasks/s000042.json              planned, not yet finished shards

A RECORD is x, y, z (m), px, py, pz (MeV) as float32 and ``label`` (1 if any detector saw the
photon) as uint32, so the rows read as float32 [N, 7] whose last column holds the label bits.
Photons are permuted inside each shard when it is written; a shard holds the photons of
several source files of one split, so any contiguous block is a random sample of its sources.

Splits are assigned per source file by a seeded hash of its key (``PhotonSource``), so adding
files never moves existing ones. Shards store raw physical values; each shard's moments
(count, mean, M2, min, max per column, float64) are computed while converting, and
``normalization`` merges those of the train shards exactly (Chan's update), so training never
needs a pass over the data for it.

Building is incremental: new files are planned into tasks, tasks are converted in parallel
(one process per task here, or one Slurm array element per task subset), and ``finalize``
merges the finished shards into the manifest. Shards are immutable; a changed source file
is an error. ``plan`` and ``finalize`` write the manifest and must not run concurrently;
``run`` tasks may run in any number of processes or nodes at once.

Command line::

    python -m legend_optical_map_ml.photon_store build CONFIG.yaml [--workers N]
    python -m legend_optical_map_ml.photon_store plan CONFIG.yaml
    python -m legend_optical_map_ml.photon_store run CONFIG.yaml --task-index I --task-count N
    python -m legend_optical_map_ml.photon_store finalize CONFIG.yaml
    python -m legend_optical_map_ml.photon_store info STORE
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from legend_optical_map_ml.config import (
    PhotonSplitConfig,
    PhotonStoreConfig,
    load_photon_store_config,
)
from legend_optical_map_ml.event_cache import HIT_DTYPE
from legend_optical_map_ml.features import AffineTransform, RunningMoments
from legend_optical_map_ml.reader import read_arrays

VERSION = 1
COLUMNS = ("x", "y", "z", "px", "py", "pz")
UNITS = ("m", "m", "m", "MeV", "MeV", "MeV")
RECORD = np.dtype([*[(c, "<f4") for c in COLUMNS], ("label", "<u4")])  # 28 bytes
assert RECORD.itemsize == 28
MOMENT_ROWS = 4_000_000  # rows converted to float64 at once for the moments (~200 MB)
SPLITS = ("train", "validation", "test")
# Settings that must not change once a store exists (sources may be added).
FIXED_SETTINGS = ("shard_photons", "split", "seed")


# --- Keys, splits, listing --------------------------------------------------------------------


def split_of(key: str, split: PhotonSplitConfig) -> str:
    """Split of a source file: a uniform number from a seeded hash of its key."""
    digest = hashlib.sha256(f"{split.seed}:{key}".encode()).digest()
    u = int.from_bytes(digest[:8], "little") / 2**64
    edge = 0.0
    for name in SPLITS:
        edge += split.fractions.get(name, 0.0)
        if u < edge:
            return name
    return next(name for name in reversed(SPLITS) if split.fractions.get(name, 0.0) > 0)


def list_files(config: PhotonStoreConfig) -> list[tuple[str, Path]]:
    """(key, path) of every source file, sorted by key."""
    files = {}
    for source in config.sources:
        if not source.path.is_dir():
            raise FileNotFoundError(f"Photon source folder not found: {source.path}")
        for path in sorted(source.path.glob(source.pattern)):
            if not path.is_file():
                continue
            key = f"{source.key_prefix}/{path.relative_to(source.path).as_posix()}"
            if key in files:
                raise ValueError(f"Duplicate photon file key {key}")
            files[key] = path
    if not files:
        raise FileNotFoundError("No photon source files found")
    return sorted(files.items())


def simulated_photons(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as h:
        return int(h["number_of_simulated_events"][()])


# --- Manifest -----------------------------------------------------------------------------------


def manifest_path(store: Path) -> Path:
    return Path(store) / "manifest.json"


def load_manifest(store: str | Path) -> dict:
    path = manifest_path(Path(store))
    if not path.is_file():
        raise FileNotFoundError(f"No photon store manifest in {store}")
    manifest = json.loads(path.read_text())
    if manifest.get("version") != VERSION:
        raise ValueError(f"Unsupported photon store version in {store}")
    return manifest


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(f".tmp-{path.name}")
    temporary.write_text(json.dumps(value, indent=1) + "\n")
    temporary.replace(path)


def _empty_manifest(config: PhotonStoreConfig) -> dict:
    return {
        "version": VERSION,
        "settings": config.model_dump(mode="json", include=set(FIXED_SETTINGS)),
        "sources": [s.model_dump(mode="json") for s in config.sources],
        "record": {"columns": list(COLUMNS), "units": list(UNITS), "label": "any_channel"},
        "files": {},
        "shards": {},
        "next_shard": 0,
    }


def _open_manifest(config: PhotonStoreConfig) -> dict:
    path = manifest_path(config.store)
    if not path.is_file():
        return _empty_manifest(config)
    manifest = load_manifest(config.store)
    settings = config.model_dump(mode="json", include=set(FIXED_SETTINGS))
    if manifest["settings"] != settings:
        raise ValueError(
            f"{config.store}: shard_photons, split and seed are fixed once the store exists "
            f"(stored {manifest['settings']}, given {settings})"
        )
    manifest["sources"] = [s.model_dump(mode="json") for s in config.sources]
    return manifest


def shard_name(shard: int) -> str:
    return f"s{shard:06d}"


def shard_paths(store: Path, split: str, shard: int) -> dict[str, Path]:
    base = Path(store) / "shards" / split / shard_name(shard)
    return {
        "records": base.with_suffix(".bin"),
        "hits": base.with_suffix(".hits.npy"),
        "summary": base.with_suffix(".json"),
    }


def _set_striping(directory: Path, count: int = 8, size: str = "16M") -> None:
    """Stripe new shard folders over several Lustre targets (best effort, Lustre only)."""
    if shutil.which("lfs") is None:
        return
    subprocess.run(
        ["lfs", "setstripe", "-c", str(count), "-S", size, str(directory)],
        check=False,
        capture_output=True,
    )


# --- Planning -----------------------------------------------------------------------------------


def plan(config: PhotonStoreConfig, *, threads: int = 32) -> list[int]:
    """Plan new source files into shard tasks; returns the new task (shard) numbers.

    Known files must be unchanged (size and modification time); finished but not finalized
    shards are finalized first, so planning never repeats work.
    """
    store = config.store
    store.mkdir(parents=True, exist_ok=True)
    finalize(config)
    manifest = _open_manifest(config)
    pending = {key for task in _tasks(store) for key, _ in task["files"]}
    new = []
    for key, path in list_files(config):
        info = os.stat(path)
        known = manifest["files"].get(key)
        if known is not None:
            if (known["size"], known["mtime_ns"]) != (info.st_size, info.st_mtime_ns):
                raise ValueError(f"{path} changed after it was stored; shards are immutable")
            continue
        if key not in pending:
            new.append((key, path))
    if not new:
        _write_json(manifest_path(store), manifest)
        return []
    with ThreadPoolExecutor(threads) as pool:
        counts = list(pool.map(simulated_photons, [p for _, p in new]))
    by_split = {}
    for (key, path), n in zip(new, counts, strict=True):
        by_split.setdefault(split_of(key, config.split), []).append((key, str(path), n))
    (store / "tasks").mkdir(exist_ok=True)
    created = []
    for split in SPLITS:
        group, photons = [], 0
        files = by_split.get(split, [])
        for i, (key, path, n) in enumerate(files):
            group.append((key, path))
            photons += n
            if photons >= config.shard_photons or i == len(files) - 1:
                shard = manifest["next_shard"]
                manifest["next_shard"] += 1
                directory = store / "shards" / split
                if not directory.exists():
                    directory.mkdir(parents=True)
                    _set_striping(directory)
                task = {"shard": shard, "split": split, "files": group, "photons": photons}
                _write_json(store / "tasks" / f"{shard_name(shard)}.json", task)
                created.append(shard)
                group, photons = [], 0
    _write_json(manifest_path(store), manifest)
    return created


def _tasks(store: Path) -> list[dict]:
    folder = Path(store) / "tasks"
    if not folder.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(folder.glob("s*.json"))]


# --- Conversion of one shard --------------------------------------------------------------------


def read_photons(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Records [n] of one LH5 file in vertex order and its hits (row, detector, hits)."""
    import h5py

    with h5py.File(path, "r") as h:
        a = read_arrays(h)
    ids = a["event_ids"]
    order = np.argsort(ids)
    rows = order[np.searchsorted(ids, a["hit_event_ids"], sorter=order)]
    pairs, counts = np.unique(
        np.column_stack([rows, a["detector_ids"]]).astype(np.int64), axis=0, return_counts=True
    )
    hits = np.empty(len(pairs), dtype=HIT_DTYPE)
    hits["row"], hits["detector"], hits["hits"] = pairs[:, 0], pairs[:, 1], counts
    records = np.zeros(len(ids), dtype=RECORD)
    for i, c in enumerate(COLUMNS[:3]):
        records[c] = a["positions"][:, i]
    for i, c in enumerate(COLUMNS[3:]):
        records[c] = a["momenta"][:, i]
    records["label"][rows] = 1
    return records, hits


def moments(records: np.ndarray) -> RunningMoments:
    """Moments of the six physical columns, in float64, chunked."""
    result = RunningMoments()
    columns = records.view(np.float32).reshape(-1, 7)[:, :6]
    for start in range(0, len(records), MOMENT_ROWS):
        result.add(columns[start : start + MOMENT_ROWS])
    return result


def convert_task(store: str | Path, task: dict, seed: int) -> dict:
    """Write one shard (records, hits, summary) for a planned task; returns the summary."""
    store = Path(store)
    shard, split = task["shard"], task["split"]
    paths = shard_paths(store, split, shard)
    if paths["summary"].is_file():
        return json.loads(paths["summary"].read_text())
    parts, hit_parts, files, offset = [], [], [], 0
    for key, path in task["files"]:
        path = Path(path)
        info = os.stat(path)
        try:
            records, hits = read_photons(path)
        except (KeyError, ValueError, OSError, TypeError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        hits["row"] += offset
        parts.append(records)
        hit_parts.append(hits)
        files.append(
            dict(
                key=key,
                path=str(path),
                size=info.st_size,
                mtime_ns=info.st_mtime_ns,
                n_photons=len(records),
                n_detected=int(records["label"].sum()),
            )
        )
        offset += len(records)
    records = np.concatenate(parts)
    hits = np.concatenate(hit_parts)
    if offset >= 2**31:
        raise ValueError("A shard must hold fewer than 2^31 photons (HIT_DTYPE rows are int32)")
    permutation = np.random.default_rng([seed, shard]).permutation(len(records))
    records = records[permutation]
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    hits["row"] = inverse[hits["row"]]
    hits = hits[np.argsort(hits["row"], kind="stable")]
    summary = dict(
        shard=shard,
        split=split,
        records=paths["records"].relative_to(store).as_posix(),
        hits=paths["hits"].relative_to(store).as_posix(),
        n_photons=len(records),
        n_detected=int(records["label"].sum()),
        n_hits=len(hits),
        moments=moments(records).to_dict(),
        files=files,
    )
    for name, write in (
        ("records", lambda f: records.tofile(f)),
        ("hits", lambda f: np.save(f, hits)),
    ):
        temporary = paths[name].with_name(f".tmp-{paths[name].name}")
        with open(temporary, "wb") as handle:
            write(handle)
        temporary.replace(paths[name])
    _write_json(paths["summary"], summary)
    return summary


def run(config: PhotonStoreConfig, *, task_index=0, task_count=1, workers=1) -> int:
    """Convert planned tasks ``task_index::task_count``; returns how many shards were written."""
    tasks = _tasks(config.store)[task_index::task_count]
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            convert_task(config.store, task, config.seed)
        return len(tasks)
    with ProcessPoolExecutor(min(workers, len(tasks)), mp_context=get_context("spawn")) as pool:
        list(pool.map(convert_task, [config.store] * len(tasks), tasks, [config.seed] * len(tasks)))
    return len(tasks)


def finalize(config: PhotonStoreConfig) -> int:
    """Merge finished shards into the manifest and remove their tasks; returns their count."""
    store = config.store
    tasks = _tasks(store)
    if not tasks:
        return 0
    manifest = _open_manifest(config)
    done = 0
    for task in tasks:
        paths = shard_paths(store, task["split"], task["shard"])
        if not paths["summary"].is_file():
            continue
        summary = json.loads(paths["summary"].read_text())
        files = summary.pop("files")
        for record in files:
            key = record.pop("key")
            manifest["files"][key] = dict(record, shard=summary["shard"], split=summary["split"])
        summary["sources"] = [key for key, _ in task["files"]]
        manifest["shards"][str(summary["shard"])] = summary
        done += 1
    _write_json(manifest_path(store), manifest)
    for task in tasks:
        if shard_paths(store, task["split"], task["shard"])["summary"].is_file():
            (store / "tasks" / f"{shard_name(task['shard'])}.json").unlink()
    return done


def build(config: PhotonStoreConfig, *, workers: int = 1) -> dict:
    """Plan, convert and finalize everything new, in this process (and its workers)."""
    created = plan(config)
    run(config, workers=workers)
    finalize(config)
    remaining = len(_tasks(config.store))
    if remaining:
        raise RuntimeError(f"{remaining} photon store tasks did not finish")
    return dict(new_shards=len(created), **summary(load_manifest(config.store)))


# --- Reading helpers ----------------------------------------------------------------------------


def shards(manifest: dict, split: str) -> list[dict]:
    """Shard summaries of one split, in shard order."""
    return sorted(
        (s for s in manifest["shards"].values() if s["split"] == split), key=lambda s: s["shard"]
    )


def split_moments(manifest: dict, split: str = "train") -> RunningMoments:
    total = RunningMoments()
    for shard in shards(manifest, split):
        total.merge(RunningMoments.from_dict(shard["moments"]))
    return total


def normalization(manifest: dict, method: str = "standard", split: str = "train") -> dict:
    """Affine normalization of position and momentum from the split's merged shard moments."""
    moments = split_moments(manifest, split)
    if not moments.n:
        raise ValueError(f"No {split} photons in the store")
    transform = AffineTransform.from_moments(moments, method)
    return dict(
        columns=list(COLUMNS),
        units=list(UNITS),
        split=split,
        n_photons=int(moments.n),
        **transform.to_dict(),
    )


def summary(manifest: dict) -> dict:
    result = {"files": len(manifest["files"])}
    for split in SPLITS:
        selected = shards(manifest, split)
        n = sum(s["n_photons"] for s in selected)
        detected = sum(s["n_detected"] for s in selected)
        result[split] = dict(
            shards=len(selected),
            photons=n,
            detected=detected,
            detected_fraction=detected / n if n else None,
        )
    return result


# --- Command line -------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "plan", "run", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("config")
        if name in ("build", "run"):
            command.add_argument("--workers", type=int, default=1)
        if name == "run":
            command.add_argument("--task-index", type=int, default=0)
            command.add_argument("--task-count", type=int, default=1)
    info = commands.add_parser("info")
    info.add_argument("store")
    args = parser.parse_args(argv)
    if args.command == "info":
        manifest = load_manifest(args.store)
        print(json.dumps(dict(summary(manifest), normalization=normalization(manifest)), indent=1))
        return
    config = load_photon_store_config(args.config)
    if args.command == "build":
        print(json.dumps(build(config, workers=args.workers), indent=1))
    elif args.command == "plan":
        print(f"{len(plan(config))} new tasks in {config.store / 'tasks'}")
    elif args.command == "run":
        count = run(
            config, task_index=args.task_index, task_count=args.task_count, workers=args.workers
        )
        print(f"{count} shards written")
    else:
        print(f"{finalize(config)} shards finalized")


if __name__ == "__main__":
    main()
