"""LH5 input validation. Importing this module does not require h5py."""

import csv
import errno
import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from legend_optical_map_ml.config import ANOMALOUS_FILE, SourceConfig

# An optional `_v{voxel index}` segment (`run00001_v00003913_x+...`) is accepted and ignored.
NAME = re.compile(
    r"run\d+(?:_v\d+)?_x([+-]?\d+(?:\.\d+)?)_y([+-]?\d+(?:\.\d+)?)"
    r"_z([+-]?\d+(?:\.\d+)?)\.stp\.lh5$"
)


@dataclass
class OpticalRun:
    file: str
    fidelity: str
    fingerprint: str
    nominal_center: np.ndarray
    measured_center: np.ndarray
    event_ids: np.ndarray
    positions: np.ndarray
    momenta: np.ndarray
    vertex_time: np.ndarray
    n_part: np.ndarray
    hit_event_ids: np.ndarray
    detector_ids: np.ndarray
    hit_time: np.ndarray
    hit_wavelength: np.ndarray
    source: int | None = None  # index into SourceConfig.sources (role-based layout only)
    role: str | None = None  # explicit assignment from a file-list manifest

    @property
    def n_events(self):
        return len(self.event_ids)


def nominal_center(name: str) -> np.ndarray:
    """Voxel center encoded in a simulation filename."""
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError("filename must encode nominal x/y/z coordinates")
    return np.array(match.groups(), dtype=float)


def read_arrays(h, *, require_hits: bool = True) -> dict:
    """Validated vertex and optical-hit arrays of one open LH5 file (h5py.File).

    With ``require_hits=False`` a file without the ``stp/optical`` group is accepted:
    ``labeled`` is False and the hit arrays are empty. Files with hits are always checked.
    """

    def array(key, unit=None, integer=False):
        d = h[key]
        a = d[:]
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError(f"{key}: expected finite 1-D array")
        if integer and not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"{key}: expected integer values")
        if unit and d.attrs.get("units") not in (unit, unit.encode()):
            raise ValueError(f"{key}: expected units {unit}")
        return a

    n_raw = h["number_of_simulated_events"][()]
    if np.ndim(n_raw) != 0 or not np.isfinite(n_raw) or n_raw <= 0:
        raise ValueError("invalid simulated event count")
    n = int(n_raw)
    if n != n_raw:
        raise ValueError("non-integral simulated event count")
    ev = array("vtx/evtid", integer=True)
    if len(ev) != n or len(np.unique(ev)) != n:
        raise ValueError("vertex IDs must be unique and match simulated event count")
    pos = [array("vtx/" + a + "loc", "m") for a in "xyz"]
    if all(f"vtx/p{axis}" in h for axis in "xyz"):
        # Older inputs store one momentum vector directly alongside each vertex.
        mom = [array("vtx/p" + a, "MeV") for a in "xyz"]
    elif all(f"particles/p{axis}" in h for axis in "xyz"):
        # The secondment LH5 layout stores momenta in a particle table. Map its
        # rows back to events explicitly rather than relying on table ordering.
        particle_ev = array("particles/evtid", integer=True)
        particle_mom = [array("particles/p" + a, "MeV") for a in "xyz"]
        if any(len(a) != len(particle_ev) for a in particle_mom):
            raise ValueError("particle momentum array length mismatch")
        if len(particle_ev) != n or len(np.unique(particle_ev)) != n:
            raise ValueError("expected exactly one particle momentum per event")
        if not np.array_equal(np.sort(particle_ev), np.sort(ev)):
            raise ValueError("particle IDs must match vertex event IDs")
        order = np.argsort(particle_ev)
        event_order = np.searchsorted(particle_ev[order], ev)
        mom = [a[order][event_order] for a in particle_mom]
    else:
        raise ValueError("missing momentum components in vtx/ or particles/")
    time = array("vtx/time", "ns")
    n_part = array("vtx/n_part", integer=True)
    if any(len(a) != n for a in [*pos, *mom, time, n_part]):
        raise ValueError("vertex array length mismatch")
    if np.any(n_part < 1):
        raise ValueError("n_part must be positive")
    vertex = dict(
        event_ids=ev,
        positions=np.column_stack(pos),
        momenta=np.column_stack(mom),
        vertex_time=time,
        n_part=n_part,
    )
    if "stp/optical" not in h:
        if require_hits:
            raise ValueError("no optical hit data (stp/optical)")
        empty_int, empty = np.zeros(0, dtype=np.int64), np.zeros(0)
        return dict(
            **vertex,
            labeled=False,
            hit_event_ids=empty_int,
            detector_ids=empty_int,
            hit_time=empty,
            hit_wavelength=empty,
        )
    hit_ev = array("stp/optical/evtid", integer=True)
    det = array("stp/optical/det_uid", integer=True)
    ht = array("stp/optical/time", "ns")
    wave = array("stp/optical/wavelength", "nm")
    if any(len(a) != len(hit_ev) for a in [det, ht, wave]):
        raise ValueError("optical hit array length mismatch")
    if not np.isin(hit_ev, ev).all():
        raise ValueError("hit references unknown vertex event ID")
    return dict(
        **vertex,
        labeled=True,
        hit_event_ids=hit_ev,
        detector_ids=det,
        hit_time=ht,
        hit_wavelength=wave,
    )


def read_run(path: Path, root: Path, *, fidelity: str | None = None) -> OpticalRun:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install resum-flex[optical-data] to read LH5 files") from exc
    try:
        relative = (
            path.relative_to(root).as_posix() if fidelity is None else f"{fidelity}/{path.name}"
        )
        fidelity = path.relative_to(root).parts[0] if fidelity is None else fidelity
        if fidelity not in ("hf", "lf"):
            raise ValueError("file must be inside hf/ or lf/")
        center = nominal_center(path.name)
        with h5py.File(path, "r") as h:
            a = read_arrays(h)
        positions, momenta = a["positions"].astype(float), a["momenta"].astype(float)
        with path.open("rb") as f:
            fingerprint = hashlib.file_digest(f, "sha256").hexdigest()
        return OpticalRun(
            relative,
            fidelity,
            fingerprint,
            center,
            (positions.min(0) + positions.max(0)) / 2,
            a["event_ids"],
            positions,
            momenta,
            a["vertex_time"],
            a["n_part"],
            a["hit_event_ids"],
            a["detector_ids"],
            a["hit_time"],
            a["hit_wavelength"],
        )
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class InputFile:
    path: Path
    root: Path
    fidelity: str | None = None
    source: int | None = None
    role: str | None = None
    relative_name: str | None = None
    nominal_center: tuple[float, float, float] | None = None


def list_source_files(config: SourceConfig) -> list[InputFile]:
    """Configured raw files, with optional explicit fidelity, role and CSV coordinates."""
    inputs = []
    if config.directory is not None:
        root = config.directory.resolve()
        for fidelity in ("hf", "lf"):
            inputs.extend(
                InputFile(p, root, relative_name=f"{fidelity}/{p.name}")
                for p in sorted((root / fidelity).glob("*.stp.lh5"))
            )
    elif config.sources:
        for index, source in enumerate(config.sources):
            folder = source.path.expanduser().resolve()
            if not folder.is_dir():
                raise FileNotFoundError(f"Optical input folder does not exist: {folder}")
            paths = sorted(folder.glob("*.stp.lh5"))
            if not paths:
                raise ValueError(f"No simulation files found in {folder}")
            inputs.extend(
                InputFile(p, folder, source.fidelity, source=index,
                          relative_name=f"{source.fidelity}/{p.name}")
                for p in paths
            )
    elif config.manifests:
        for index, manifest in enumerate(config.manifests):
            path = manifest.path.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"File-list manifest does not exist: {path}")
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                required = {"stp_file"}
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path}: expected a CSV with an stp_file column")
                fields = set(reader.fieldnames)
                has_center = {"x_mm", "y_mm", "z_mm"}.issubset(fields)
                partial_center = bool(fields & {"x_mm", "y_mm", "z_mm"}) and not has_center
                if partial_center:
                    raise ValueError(f"{path}: x_mm, y_mm and z_mm must be provided together")
                for line, row in enumerate(reader, start=2):
                    listed = (row.get("stp_file") or "").strip()
                    if not listed:
                        raise ValueError(f"{path}:{line}: stp_file is empty")
                    raw_path = Path(listed).expanduser()
                    file_path = raw_path if raw_path.is_absolute() else path.parent / raw_path
                    file_path = Path(os.path.abspath(file_path))
                    if not file_path.is_file():
                        raise FileNotFoundError(
                            f"{path}:{line}: listed file does not exist: {file_path}"
                        )
                    row_fidelity = (row.get("tier") or "").strip().lower()
                    fidelity = manifest.fidelity or row_fidelity or None
                    if fidelity not in ("lf", "hf"):
                        raise ValueError(f"{path}:{line}: set fidelity or provide tier: lf/hf")
                    if manifest.fidelity and row_fidelity and row_fidelity != manifest.fidelity:
                        raise ValueError(
                            f"{path}:{line}: tier {row_fidelity!r} disagrees with configured "
                            f"fidelity {manifest.fidelity!r}"
                        )
                    center = None
                    if has_center:
                        try:
                            center = tuple(
                                float(row[key]) / 1000 for key in ("x_mm", "y_mm", "z_mm")
                            )
                        except (TypeError, ValueError) as exc:
                            raise ValueError(f"{path}:{line}: invalid x_mm/y_mm/z_mm") from exc
                        if not np.isfinite(center).all():
                            raise ValueError(f"{path}:{line}: non-finite voxel center")
                    try:
                        relative = file_path.relative_to(path.parent).as_posix()
                    except ValueError:
                        relative = os.path.relpath(file_path, path.parent)
                    inputs.append(
                        InputFile(
                            file_path,
                            path.parent,
                            fidelity=fidelity,
                            source=index,
                            role=manifest.role,
                            relative_name=f"{fidelity}/{relative}",
                            nominal_center=center,
                        )
                    )
            if not any(item.source == index for item in inputs):
                raise ValueError(f"No simulation files listed in {path}")
    else:
        for fidelity, folders in sorted(config.directories.items()):
            for folder in folders:
                folder = folder.expanduser().resolve()
                if not folder.is_dir():
                    raise FileNotFoundError(f"Optical input folder does not exist: {folder}")
                paths = sorted(folder.glob("*.stp.lh5"))
                if not paths:
                    raise ValueError(f"No simulation files found in {folder}")
                inputs.extend(
                    InputFile(p, folder, fidelity,
                              relative_name=f"{fidelity}/{p.name}")
                    for p in paths
                )
    if not inputs:
        raise ValueError("No simulation files found in configured folders")
    return inputs


def _file_identity(path: str) -> str:
    info = os.stat(path)
    return f"{os.path.basename(path)}:{info.st_size}:{info.st_mtime_ns}"


def source_stamp(config: SourceConfig) -> list[list]:
    """Cheap identity of the raw inputs without opening any file.

    Folder sources hash each folder's file names, sizes and modification times. File-list
    sources also hash each CSV's contents and the listed inputs. Adding, removing, renaming
    or rewriting an input changes the stamp. Metadata is fetched in parallel because
    one-by-one stat calls are slow on network file systems.
    """
    if config.manifests:
        inputs = list_source_files(config)
        def identity(path):
            stat = path.stat()
            return stat.st_size, stat.st_mtime_ns

        with ThreadPoolExecutor(max_workers=32) as pool:
            file_ids = list(pool.map(identity, (item.path for item in inputs), chunksize=256))
        files = sorted(
            f"{item.relative_name}:{size}:{mtime}"
            for item, (size, mtime) in zip(inputs, file_ids, strict=True)
        )
        manifests = []
        for item in config.manifests:
            path = item.path.expanduser().resolve()
            stat = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifests.append([str(path), stat.st_size, stat.st_mtime_ns, digest])
        return [["manifests", manifests, hashlib.sha256("\n".join(files).encode()).hexdigest()]]
    if config.directory is not None:
        folders = [config.directory / f for f in ("hf", "lf") if (config.directory / f).is_dir()]
    elif config.sources:
        folders = [s.path for s in config.sources]
    else:
        folders = [f for _, fs in sorted(config.directories.items()) for f in fs]
    stamp = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for folder in folders:
            folder = folder.expanduser().resolve()
            paths = sorted(e.path for e in os.scandir(folder) if e.name.endswith(".stp.lh5"))
            identities = "\n".join(pool.map(_file_identity, paths))
            stamp.append([str(folder), len(paths), hashlib.sha256(identities.encode()).hexdigest()])
    return stamp


def real_paths(paths) -> list[Path]:
    """Paths with every symlink followed, like ``Path.resolve``, for many files at once.

    ``Path.resolve`` checks every path component (~20 metadata calls per file, slow on CFS).
    Here each file costs one ``readlink`` per link, run in parallel, and every folder is
    resolved only once.
    """
    folders = {}

    def folder(path: str) -> str:
        if path not in folders:
            folders[path] = os.path.realpath(path)
        return folders[path]

    def one(path) -> Path:
        path = os.path.join(folder(os.path.dirname(os.path.abspath(path))), os.path.basename(path))
        for _ in range(40):
            try:
                target = os.readlink(path)
            except OSError as exc:
                if exc.errno != errno.EINVAL:  # EINVAL: not a link
                    raise
                return Path(path)
            target = os.path.join(os.path.dirname(path), target)
            path = os.path.join(folder(os.path.dirname(target)), os.path.basename(target))
        raise OSError(f"Too many levels of symbolic links: {path}")

    with ThreadPoolExecutor(max_workers=32) as pool:
        return list(pool.map(one, paths, chunksize=256))


def select_inputs(config: SourceConfig) -> tuple[list[tuple], list[dict]]:
    """Included inputs as (InputFile, name, source index, resolved path), plus exclusions."""
    inputs = list_source_files(config)
    resolved_paths = real_paths([item.path for item in inputs])
    sources = {s.path.expanduser().resolve(): i for i, s in enumerate(config.sources or [])}
    exclusions = set(config.exclude_files) | {ANOMALOUS_FILE}
    included, excluded = [], []
    seen_paths, seen_names = {}, {}
    for item, resolved in zip(inputs, resolved_paths, strict=True):
        path, root = item.path, item.root
        name = item.relative_name or path.relative_to(root).as_posix()
        explicit_fidelity = item.fidelity
        fidelity = name.split("/", 1)[0]
        if resolved in seen_paths:
            if config.manifests:
                raise ValueError(f"File is listed more than once across manifests: {path}")
            if seen_paths[resolved] != fidelity:
                raise ValueError(f"Same source file assigned to LF and HF: {path}")
            continue
        if name in seen_names:
            raise ValueError(f"Ambiguous simulation filename {name} in multiple folders")
        seen_paths[resolved], seen_names[name] = fidelity, resolved
        exclusion_names = {name, path.name, f"{fidelity}/{path.name}"}
        if item.relative_name:
            exclusion_names.add(item.relative_name.split("/", 1)[-1])
        matched_exclusion = next((key for key in exclusions if key in exclusion_names), None)
        if matched_exclusion:
            excluded.append(
                {
                    "file": name,
                    "reason": (
                        "known anomaly"
                        if matched_exclusion == ANOMALOUS_FILE
                        else "configured exclusion"
                    ),
                }
            )
            continue
        source = (
            item.source
            if item.source is not None
            else (sources[root] if config.sources else None)
        )
        included.append((item, name, source, resolved))
    if not included:
        raise ValueError("No included simulation files")
    return included, excluded


def read_optical_runs(config: SourceConfig) -> tuple[list[OpticalRun], list[dict]]:
    included, excluded = select_inputs(config)
    runs = []
    for item, name, source, _ in included:
        run = read_run(item.path, item.root, fidelity=item.fidelity)
        run.file = name
        run.source = source
        run.role = item.role
        if item.nominal_center is not None:
            run.nominal_center = np.asarray(item.nominal_center, dtype=float)
        runs.append(run)
    return sorted(runs, key=lambda r: r.file), excluded
