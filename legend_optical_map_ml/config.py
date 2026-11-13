"""Validated configuration for fixed optical simulation datasets."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

ANOMALOUS_FILE = "lf/run067_x-0.088_y+0.484_z-1.462.stp.lh5"
# Roles of voxels in the role-based layout; see RoleSource. New roles are appended.
ROLES = (
    "cnp_train", "cnp_validation", "cnp_test", "gp_train", "gp_unlabeled", "gp_validation",
    "gp_test",
)
Role = Literal[
    "cnp_train", "cnp_validation", "cnp_test", "gp_train", "gp_unlabeled", "gp_validation",
    "gp_test",
]
# Roles whose files need no optical hits (inputs only, e.g. for event-model inference).
UNLABELED_ROLES = ("gp_unlabeled",)
# Roles that may share positions in a nested multi-fidelity GP design, for example HF training
# voxels at positions of LF inputs. Held-out CNP and GP validation roles may also overlap.
GP_TRAINING_ROLES = ("gp_train", "gp_unlabeled")
# Held-out roles that may evaluate the same position. CNP validation shared with the GP test
# leaks the checkpoint choice into the test slightly; accepted for datasets whose validation
# lists were sampled independently (e.g. LBNL secondment), avoid it in new samplings.
SHARED_VALIDATION_ROLES = ("cnp_validation", "gp_validation", "gp_test")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoleSource(StrictModel):
    """One folder whose voxels are divided among ROLES by fraction or by file count.

    Roles left out get no files. ``fractions`` must sum to 1 and are rounded per folder;
    ``counts`` must add up to the folder's file count unless one role takes ``rest``.
    An already split folder is e.g. ``fractions: {cnp_validation: 1.0}``.
    """

    path: Path
    fidelity: Literal["lf", "hf"] = "lf"
    fractions: dict[Role, float] | None = None
    counts: dict[Role, int | Literal["rest"]] | None = None

    @model_validator(mode="after")
    def check_split(self):
        if (self.fractions is None) == (self.counts is None):
            raise ValueError(f"{self.path}: give either fractions or counts")
        if self.fractions is not None:
            values = list(self.fractions.values())
            if any(not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-9:
                raise ValueError(f"{self.path}: fractions must lie in [0, 1] and sum to 1")
        else:
            values = list(self.counts.values())
            if values.count("rest") > 1:
                raise ValueError(f"{self.path}: only one role may take 'rest'")
            if any(v != "rest" and v < 0 for v in values):
                raise ValueError(f"{self.path}: counts must be nonnegative")
        return self


class ManifestSource(StrictModel):
    """A CSV file list: all rows in one role, or split across roles like a RoleSource.

    ``fractions``/``counts`` split the rows seeded by ``split.seed`` (one stream per manifest),
    keeping voxel groups together, e.g. ``fractions: {gp_validation: 0.5, gp_test: 0.5}``.
    """

    path: Path
    role: Role | None = None
    fractions: dict[Role, float] | None = None
    counts: dict[Role, int | Literal["rest"]] | None = None
    # Read from the CSV's `tier` column when present; specify this for simpler lists.
    fidelity: Literal["lf", "hf"] | None = None

    @model_validator(mode="after")
    def one_assignment(self):
        if sum(x is not None for x in (self.role, self.fractions, self.counts)) != 1:
            raise ValueError(f"{self.path}: give exactly one of role, fractions or counts")
        if self.role is None:
            RoleSource(path=self.path, fractions=self.fractions, counts=self.counts)  # same checks
        return self


class SourceConfig(StrictModel):
    directory: Path | None = None
    directories: dict[Literal["lf", "hf"], list[Path]] = Field(default_factory=dict)
    # Role-based LF layout: per-folder splits replace the global split fractions.
    sources: list[RoleSource] = Field(default_factory=list)
    # Explicit file lists (for example CSV manifests) assign every row directly to a role.
    manifests: list[ManifestSource] = Field(default_factory=list)
    exclude_files: list[str] = Field(default_factory=lambda: [ANOMALOUS_FILE])

    @model_validator(mode="after")
    def check_sources(self):
        if sum([
            self.directory is not None,
            bool(self.directories),
            bool(self.sources),
            bool(self.manifests),
        ]) != 1:
            raise ValueError(
                "Provide either directory, per-fidelity directories, role sources, "
                "or file-list manifests"
            )
        if any(not paths for paths in self.directories.values()):
            raise ValueError("Each configured fidelity needs at least one folder")
        paths = [s.path for s in self.sources]
        if len(set(paths)) != len(paths):
            raise ValueError("Each role source folder may appear only once")
        manifest_paths = [m.path for m in self.manifests]
        if len(set(manifest_paths)) != len(manifest_paths):
            raise ValueError("Each file-list manifest may appear only once")
        return self


class CylindricalConfig(StrictModel):
    include_azimuth: bool = True
    angle_encoding: Literal["sin_cos", "radians"] = "sin_cos"


class ThetaConfig(StrictModel):
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian"
    center_source: Literal["filename", "observed_midrange"] = "filename"
    cylindrical: CylindricalConfig = Field(default_factory=CylindricalConfig)


class VertexConfig(StrictModel):
    representation: Literal["offset", "absolute", "omit"] = "offset"
    coordinates: Literal["cartesian", "cylindrical"] = "cartesian"


class MomentumConfig(StrictModel):
    representation: Literal["components", "direction_and_magnitude", "omit"] = "components"


class PhiConfig(StrictModel):
    vertex: VertexConfig = Field(default_factory=VertexConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)


class TargetConfig(StrictModel):
    kind: Literal["any_channel", "single_channel"] = "any_channel"
    detector_id: int | None = None

    @model_validator(mode="after")
    def check_detector(self):
        if (self.kind == "single_channel") != (self.detector_id is not None):
            raise ValueError("detector_id is required only for single_channel")
        return self


class SplitConfig(StrictModel):
    lf_train_only: bool = False
    hf_train_count: int | None = Field(default=None, gt=0)
    train_fraction: float = Field(default=0.7, gt=0, lt=1)
    validation_fraction: float = Field(default=0.15, ge=0, lt=1)
    test_fraction: float = Field(default=0.15, ge=0, lt=1)
    seed: int = 42
    manifest: Path = Path("splits/voxel_split.json")

    @model_validator(mode="after")
    def check_fractions(self):
        if abs(sum(self.fractions) - 1) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self

    @property
    def fractions(self) -> tuple[float, float, float]:
        return self.train_fraction, self.validation_fraction, self.test_fraction


class NormalizationConfig(StrictModel):
    method: Literal["standard", "minmax", "none"] = "standard"
    fit_on: Literal["training"] = "training"
    fit_fidelities: list[Literal["lf", "hf"]] | None = Field(default=None, min_length=1)


class ViewConfig(StrictModel):
    """A named input representation of the same files, roles and split.

    Unset fields take the top-level value. Written to ``<output_directory>/views/<name>``.
    """

    theta: ThetaConfig | None = None
    phi: PhiConfig | None = None
    normalization: NormalizationConfig | None = None


ViewName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
VIEW_FIELDS = ("theta", "phi", "normalization")


def _grouping(theta: ThetaConfig) -> tuple:
    """Theta settings that decide which files form one voxel group (split.voxel_groups)."""
    azimuth_free = theta.coordinates == "cylindrical" and not theta.cylindrical.include_azimuth
    return theta.center_source, azimuth_free


class OpticalDataConfig(StrictModel):
    source: SourceConfig
    output_directory: Path = Path("outputs/optical_data")
    # Shared float32 copy of the raw files (event_cache.py); by default next to
    # output_directory, so data configs writing to the same parent folder share it.
    cache_directory: Path | None = None
    theta: ThetaConfig = Field(default_factory=ThetaConfig)
    phi: PhiConfig = Field(default_factory=PhiConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    # Further input representations of the same data; the top-level settings above are the
    # view at output_directory itself (docs/data-config.md).
    views: dict[ViewName, ViewConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_views(self):
        for name, view in self.views.items():
            if view.theta is not None and _grouping(view.theta) != _grouping(self.theta):
                raise ValueError(
                    f"view {name}: theta must group voxels as the top-level theta "
                    "(same center_source and azimuth setting), as all views share one split"
                )
        return self

    @model_validator(mode="after")
    def check_role_split(self):
        per_source = {"seed", "manifest"}
        default = SplitConfig().model_dump(exclude=per_source)
        if (self.source.sources or self.source.manifests) and self.split.model_dump(
            exclude=per_source
        ) != default:
            raise ValueError(
                "With role sources or file-list manifests, set role assignment in source; "
                "only split.seed and split.manifest apply"
            )
        return self

    @property
    def event_cache(self) -> Path:
        """Directory of the shared event cache."""
        return self.cache_directory or self.output_directory.parent / "event_cache"

    def view_directory(self, name: str | None) -> Path:
        """Prepared view ``name``; None is the top-level view at output_directory."""
        return self.output_directory if name is None else self.output_directory / "views" / name

    def view_configs(self) -> dict[str | None, "OpticalDataConfig"]:
        """Settings of every view, top-level (None) first, each without ``views``.

        A named view is the top-level config with its own theta, phi and normalization, its
        own directory and the top-level event cache; source, target and split are shared.
        """
        configs = {None: self.model_copy(update={"views": {}}, deep=True)}
        for name, view in self.views.items():
            updates = {k: getattr(view, k) for k in VIEW_FIELDS if getattr(view, k) is not None}
            updates.update(
                views={},
                output_directory=self.view_directory(name),
                cache_directory=self.event_cache,
            )
            configs[name] = self.model_copy(update=updates, deep=True)
        return configs

    def stored(self) -> dict:
        """Settings as saved with a prepared view (config.json, view.json): without views."""
        return self.model_dump(mode="json", exclude={"views"})


def resolve_cache_directory(config: OpticalDataConfig, resolve) -> None:
    if config.cache_directory is not None:
        config.cache_directory = resolve(config.cache_directory)


def load_optical_config(path: str | Path) -> OpticalDataConfig:
    """Load preparation settings with paths relative to the configuration file."""
    import yaml

    path = Path(path).resolve()
    config = OpticalDataConfig.model_validate(yaml.safe_load(path.read_text()))

    def resolve(value):
        value = value.expanduser()
        return value.resolve() if value.is_absolute() else (path.parent / value).resolve()

    if config.source.directory is not None:
        config.source.directory = resolve(config.source.directory)
    config.source.directories = {
        fidelity: [resolve(folder) for folder in folders]
        for fidelity, folders in config.source.directories.items()
    }
    for source in [*config.source.sources, *config.source.manifests]:
        source.path = resolve(source.path)
    config.output_directory = resolve(config.output_directory)
    resolve_cache_directory(config, resolve)
    config.split.manifest = resolve(config.split.manifest)
    return config


# --- Photon store: ungrouped photons (e.g. homogeneous campaigns) for event models ----------

PhotonSplitName = Literal["train", "validation", "test"]


class PhotonSource(StrictModel):
    """LH5 files ``path/pattern``; ``name`` prefixes their keys (default: the folder name).

    A file's key ``<name>/<path relative to the folder>`` decides its split, so moving the data
    keeps the split as long as name and relative path stay the same.
    """

    path: Path
    pattern: str = "*.stp.lh5"
    name: str | None = None

    @property
    def key_prefix(self) -> str:
        return self.name or self.path.name


class PhotonSplitConfig(StrictModel):
    seed: int = 0
    fractions: dict[PhotonSplitName, float] = Field(
        default_factory=lambda: {"train": 0.9, "validation": 0.05, "test": 0.05}
    )

    @model_validator(mode="after")
    def check_fractions(self):
        values = list(self.fractions.values())
        if any(not 0 <= v <= 1 for v in values) or abs(sum(values) - 1) > 1e-9:
            raise ValueError("photon split fractions must lie in [0, 1] and sum to 1")
        if self.fractions.get("train", 0) == 0:
            raise ValueError("the train split needs a nonzero fraction")
        return self


class PhotonStoreConfig(StrictModel):
    """Settings of a photon store (photon_store.py). Fixed once the store exists."""

    sources: list[PhotonSource] = Field(min_length=1)
    store: Path
    shard_photons: int = Field(default=100_000_000, gt=0)
    split: PhotonSplitConfig = Field(default_factory=PhotonSplitConfig)
    seed: int = 0  # photon permutation inside shards

    @model_validator(mode="after")
    def distinct_prefixes(self):
        prefixes = [s.key_prefix for s in self.sources]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("photon sources need distinct names (key prefixes)")
        return self


def load_photon_store_config(path: str | Path) -> PhotonStoreConfig:
    """Load photon-store settings with paths relative to the configuration file."""
    import yaml

    path = Path(path).resolve()
    config = PhotonStoreConfig.model_validate(yaml.safe_load(path.read_text()))

    def resolve(value):
        value = value.expanduser()
        return value.resolve() if value.is_absolute() else (path.parent / value).resolve()

    for source in config.sources:
        source.path = resolve(source.path)
    config.store = resolve(config.store)
    return config
