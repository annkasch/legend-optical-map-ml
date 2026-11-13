"""Prepare optical data as a view on the shared event cache and load its partitions."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from data.batch_source import FixedBatchSource
from schemas.data_models import InputMode, StandardBatch

from legend_optical_map_ml.config import UNLABELED_ROLES, OpticalDataConfig
from legend_optical_map_ml.event_cache import EventCache, cached_runs
from legend_optical_map_ml.features import (
    AffineTransform,
    RunningMoments,
    center,
    feature_info,
    theta_features,
)
from legend_optical_map_ml.reader import source_stamp
from legend_optical_map_ml.reports import generate_reports
from legend_optical_map_ml.split import (
    FIDELITIES,
    assign_splits,
    partition_names,
    save_manifest,
    training_partition,
)
from legend_optical_map_ml.view import OpticalView, input_mode, phi_chunks, write_view


class PreparedOpticalData:
    """Prepared partitions, loaded from the view on first access."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.view = OpticalView(self.directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        self.normalization = self.view.meta["normalization"]
        self._loaded = {}

    @property
    def partitions(self) -> list[str]:
        return self.view.partitions

    def available(self, partition: str) -> list[str]:
        """Fidelities with at least one file in ``partition``."""
        return [f for f in FIDELITIES if (partition, f) in self.view.available()]

    def batch(self, partition: str, fidelity: str) -> StandardBatch:
        key = partition, fidelity
        if key not in self._loaded:
            self._loaded[key] = self.view.load(partition, fidelity)
        return self._loaded[key]

    def training_source(self, fidelity="lf"):
        return FixedBatchSource(self.batch(self.partitions[0], fidelity), replace_events=False)


def load_prepared_batch(path: str | Path) -> StandardBatch:
    """Batch saved as .npz (older prepared data)."""
    with np.load(path, allow_pickle=False) as arrays:
        return StandardBatch(
            mode=InputMode(str(arrays["mode"].item())),
            theta=arrays["theta"],
            labels=arrays["labels"],
            phi=arrays["phi"] if "phi" in arrays else None,
            event_counts=arrays["event_counts"] if "event_counts" in arrays else None,
        )


def load_partition(root: str | Path, partition: str, fidelity: str):
    """(batch, provenance) of a prepared partition; .npz batches take precedence over the view.

    provenance identifies the exact inputs: file contents, settings and normalization.
    """
    root = Path(root)
    path = root / "batches" / partition / f"{fidelity}.npz"
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return load_prepared_batch(path), dict(path=str(path.resolve()), sha256=digest)
    if not (root / "view.json").is_file():
        raise FileNotFoundError(f"No prepared {fidelity} data for {partition} in {root}")
    view = OpticalView(root)
    return view.load(partition, fidelity), view.provenance(partition, fidelity)


def prepared_partitions(root: str | Path) -> list[str]:
    """Partition names of prepared data, in their saved order."""
    root = Path(root)
    if (root / "view.json").is_file():
        return json.loads((root / "view.json").read_text())["partitions"]
    return sorted(p.name for p in (root / "batches").iterdir() if p.is_dir())


def fit_normalization(runs, config: OpticalDataConfig) -> dict:
    """Affine transforms of theta and phi, fitted on the training files ``runs`` only.

    phi moments are accumulated in chunks, so the training events never have to be in
    memory at once.
    """
    if not runs:
        raise ValueError("No training files; adjust split or provide more data")
    cache = runs[0].cache
    rows = np.array([r.row for r in runs])
    centers = np.array([center(r, config) for r in runs])  # [B, 3]
    theta = AffineTransform.fit(theta_features(centers, config)[0], config.normalization.method)
    phi = None
    if feature_info(config)["phi"]["dimension"]:
        moments = RunningMoments()
        for _, _, values in phi_chunks(cache, rows, centers, config):
            moments.add(values)
        phi = AffineTransform.from_moments(moments, config.normalization.method)
    return {
        "theta": theta.to_dict(),
        "phi": phi.to_dict() if phi else None,
        "fit_files": [r.file for r in runs],
        "minmax_range": [0, 1],
    }


def prepare_optical_data(
    config: OpticalDataConfig,
    *,
    update_manifest=False,
    workers: int | None = None,
    views: list[str | None] | None = None,
) -> PreparedOpticalData:
    """Cache new or changed inputs, assign partitions, fit normalization and write the views.

    All views (top-level and ``config.views``) share the cached files and one split; each gets
    its own normalization. ``views`` selects which are written (None = top-level; default all).
    Per view: view.npy/view.json (see legend_optical_map_ml.view), metadata.json,
    normalization.json, config.json, source_stamp.json and manifest_snapshot.json; reports/
    go to config.output_directory. Batches are built from a view when loaded. Returns the
    top-level view.
    """
    runs, excluded = cached_runs(config.source, EventCache(config.event_cache), workers=workers)
    if config.target.kind == "single_channel" and not any(
        np.any(r.cache.hits(r.row)["detector"] == config.target.detector_id)
        for r in runs
        if r.labeled
    ):
        raise ValueError("Selected detector ID never observed; cannot validate it without a roster")
    assignments, manifest = assign_splits(runs, config, update=update_manifest)
    missing = sorted(
        r.file for r in runs if not r.labeled and assignments[r.file] not in UNLABELED_ROLES
    )
    if missing:
        raise ValueError(
            f"{len(missing)} files without optical hit data are in labeled partitions "
            f"(only {', '.join(UNLABELED_ROLES)} may hold them), e.g. {missing[:3]}"
        )
    resolved = config.view_configs()
    for name in resolved if views is None else views:
        _write_view_directory(runs, assignments, manifest, resolved[name])
    generate_reports(runs, assignments, excluded, config, manifest)
    save_manifest(config.split.manifest, manifest)
    return PreparedOpticalData(config.output_directory)


def _write_view_directory(runs, assignments, manifest, config: OpticalDataConfig) -> None:
    """Normalization, view and records of one view (``config`` without views)."""
    training = [
        r
        for r in runs
        if assignments[r.file] == training_partition(config)
        and (
            config.normalization.fit_fidelities is None
            or r.fidelity in config.normalization.fit_fidelities
        )
    ]
    normalization = fit_normalization(training, config)
    partitions = partition_names(config)
    features = feature_info(config)
    entries = [(assignments[r.file], r) for r in runs]
    metadata = {}
    for split in partitions:
        metadata[split] = {}
        for fidelity in FIDELITIES:
            selected = [r for p, r in entries if p == split and r.fidelity == fidelity]
            if not selected:
                metadata[split][fidelity] = {"files": [], "status": "empty"}
                continue
            counts = np.array([r.n_events for r in selected], dtype=np.int64)
            ragged = bool(np.any(counts != counts.max()))
            metadata[split][fidelity] = {
                "files": [r.file for r in selected],
                "nominal_centers": [r.nominal_center.tolist() for r in selected],
                "measured_centers": [r.measured_center.tolist() for r in selected],
                "features": features,
                "shape": [len(selected), int(counts.max())],
                "event_counts": counts.tolist() if ragged else None,
                "mode": input_mode(config, normalization["phi"] is not None).value,
            }
    out = config.output_directory
    out.mkdir(parents=True, exist_ok=True)
    # Batches of older preparations would take precedence over the view.
    shutil.rmtree(out / "batches", ignore_errors=True)
    write_view(out, entries, partitions, config, normalization)
    # The stamp lets ensure_prepared_optical_data skip even the cache check.
    for name, payload in [
        ("source_stamp", source_stamp(config.source)),
        ("metadata", metadata),
        ("normalization", normalization),
        ("config", config.stored()),
    ]:
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "manifest_snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")


def ensure_prepared_optical_data(config: OpticalDataConfig) -> Path:
    """Prepare absent/stale artifacts; reuse data only after checking raw inputs.

    Source/configuration changes automatically update the manifest, the view and the
    normalization. Compatible existing split assignments are retained; explicit split
    policies take precedence. Invalid inputs still fail validation. A matching configuration
    and raw-file stamp (name, size, mtime per file) reuses the data without touching the
    cache; any mismatch prepares again, reading only new or changed files.
    """
    stamp = source_stamp(config.source)
    stale = [
        name
        for name, view in config.view_configs().items()
        if not (_stamp_matches(view.output_directory, stamp) and _matches_saved(view)
                and _view_complete(view.output_directory))
    ]
    if not stale and config.split.manifest.is_file():
        return config.output_directory
    # A source change makes every view stale; a changed view alone rewrites only that view,
    # with the split kept by the saved manifest.
    prepare_optical_data(config, update_manifest=True, views=stale or None)
    return config.output_directory


def _stamp_matches(out: Path, stamp) -> bool:
    path = out / "source_stamp.json"
    return path.is_file() and json.loads(path.read_text()) == stamp


def _matches_saved(config: OpticalDataConfig) -> bool:
    """True when the settings stored at the last preparation equal the current ones."""
    try:
        path = config.output_directory / "config.json"
        return OpticalDataConfig.model_validate(json.loads(path.read_text())) == config
    except (ValueError, OSError):
        return False


def _view_complete(out: Path) -> bool:
    """The view lists exactly the files of the saved manifest, in the same partitions."""
    try:
        manifest = json.loads((out / "manifest_snapshot.json").read_text())
        view = OpticalView(out)
    except (ValueError, OSError):
        return False
    saved = sorted((r["split"], r["fidelity"]) for r in manifest["files"])
    listed = sorted(
        (view.partitions[p], FIDELITIES[f])
        for p, f in zip(
            view.rows["partition"].tolist(), view.rows["fidelity"].tolist(), strict=True
        )
    )
    return saved == listed and Path(view.meta["cache"], "cache.json").is_file()
