"""Feature construction and training-only affine normalization."""

from dataclasses import asdict, dataclass

import numpy as np

from legend_optical_map_ml.config import (
    CylindricalConfig,
    OpticalDataConfig,
    PhiConfig,
    StreamTransformConfig,
)
from legend_optical_map_ml.reader import OpticalRun


def center(run: OpticalRun, config: OpticalDataConfig) -> np.ndarray:
    return run.nominal_center if config.theta.center_source == "filename" else run.measured_center


def cylindrical(xyz, *, include_azimuth=True, encoding="sin_cos", symmetry=1, weighting="none"):
    """Points [..., 3] -> (r, angle features, z); the angle is symmetry * azimuth.

    weighting "radius": angle features r * sin, r * cos (continuous on the axis).
    """
    x, y, z = np.moveaxis(np.asarray(xyz), -1, 0)
    r = np.hypot(x, y)
    angle = np.where(r == 0, 0, np.arctan2(y, x))
    if symmetry > 1:
        angle = np.angle(np.exp(1j * symmetry * angle))  # wrapped to (-pi, pi]
    if not include_azimuth:
        return np.stack([r, z], axis=-1), ["r", "z"], ["m", "m"]
    if encoding == "sin_cos":
        if weighting == "radius":
            return (
                np.stack([r, r * np.sin(angle), r * np.cos(angle), z], axis=-1),
                ["r", "r_sin_azimuth", "r_cos_azimuth", "z"],
                ["m"] * 4,
            )
        return (
            np.stack([r, np.sin(angle), np.cos(angle), z], axis=-1),
            ["r", "sin_azimuth", "cos_azimuth", "z"],
            ["m", "1", "1", "m"],
        )
    return np.stack([r, angle, z], axis=-1), ["r", "azimuth", "z"], ["m", "rad", "m"]


def azimuth(xyz) -> np.ndarray:
    """Azimuth about the z axis of points [E, 3]; 0 on the axis."""
    xyz = np.asarray(xyz, dtype=float)
    return np.where(np.hypot(xyz[:, 0], xyz[:, 1]) == 0, 0, np.arctan2(xyz[:, 1], xyz[:, 0]))


def to_local_frame(vectors, angle) -> np.ndarray:
    """Vectors [E, 3] rotated into the radial / tangential / z frame at azimuth ``angle`` [E]."""
    v = np.asarray(vectors, dtype=float)
    co, si = np.cos(angle), np.sin(angle)
    return np.column_stack([v[:, 0] * co + v[:, 1] * si, -v[:, 0] * si + v[:, 1] * co, v[:, 2]])


def from_local_frame(vectors, angle) -> np.ndarray:
    """Inverse of ``to_local_frame``."""
    return to_local_frame(vectors, -np.asarray(angle))


def momentum_azimuth(positions, centers, phi: PhiConfig) -> np.ndarray:
    """Azimuth of the cylindrical momentum frame: the vertex's (absolute) or the centre's."""
    absolute = phi.vertex.representation == "absolute"
    return azimuth(positions if absolute else centers)


def cylindrical_options(angles: CylindricalConfig, *, theta=False) -> dict:
    """Keyword arguments of ``cylindrical`` for θ or an absolute vertex (always with azimuth)."""
    return dict(
        include_azimuth=angles.include_azimuth if theta else True,
        encoding=angles.angle_encoding,
        symmetry=angles.rotational_symmetry,
        weighting=angles.angle_weighting,
    )


def decode_absolute_phi(phi, config: OpticalDataConfig, near=None):
    """``decode_events`` with the view's phi and angle settings."""
    return decode_events(phi, config.phi, config.theta.cylindrical, near)


def decode_events(values, phi: PhiConfig, angles: CylindricalConfig, near=None):
    """Unnormalized event features [E, D_phi] of an absolute-vertex view -> (positions, momenta).

    Inverse of ``phi_features`` (momenta zero when omitted); used to move donor photons. With
    an n-fold ``rotational_symmetry`` the azimuth is known only up to 360/n degrees: of the n
    rotated copies the one closest to ``near`` [E, 3] (e.g. the voxel centre) is taken. Exact
    when the point lies within its sector's half-width of ``near`` (for n = 6: the voxel is
    more than about two half-diagonals from the axis); closer to the axis the copies are close
    to each other and the error stays below their spacing.
    """
    if phi.vertex.representation != "absolute":
        raise ValueError("decoding needs an absolute vertex")
    values = np.asarray(values, dtype=float)
    if phi.vertex.coordinates == "cylindrical":
        if angles.angle_encoding == "sin_cos":  # also r-weighted: atan2 is scale-free
            r, z, rest = values[:, 0], values[:, 3], values[:, 4:]
            angle = np.arctan2(values[:, 1], values[:, 2])
        else:
            r, angle, z, rest = values[:, 0], values[:, 1], values[:, 2], values[:, 3:]
        n = angles.rotational_symmetry
        if n > 1:
            if near is None:
                raise ValueError("rotational_symmetry > 1: decoding needs reference positions")
            # angle = n * azimuth (wrapped): azimuth = angle / n + k * 2 pi / n, k = 0 .. n-1
            candidates = angle[:, None] / n + np.arange(n) * (2 * np.pi / n)  # [E, n]
            near = np.asarray(near, dtype=float)
            distance = np.hypot(
                r[:, None] * np.cos(candidates) - near[:, :1],
                r[:, None] * np.sin(candidates) - near[:, 1:2],
            )
            angle = np.take_along_axis(candidates, distance.argmin(1)[:, None], 1)[:, 0]
        positions = np.column_stack([r * np.cos(angle), r * np.sin(angle), z])
    else:
        positions, rest = values[:, :3], values[:, 3:]
    momentum = phi.momentum
    if momentum.representation == "omit":
        return positions, np.zeros_like(positions)
    m = rest[:, :3]
    if momentum.representation == "direction_and_magnitude":
        m = m * rest[:, 3:4]
    if momentum.coordinates == "cylindrical":
        m = from_local_frame(m, azimuth(positions))
    return positions, m


def theta_features(centers, config: OpticalDataConfig):
    """Design features of voxel centers [B, 3] -> ([B, D_theta], names, units)."""
    centers = np.asarray(centers, dtype=float)
    if config.theta.coordinates == "cylindrical":
        return cylindrical(centers, **cylindrical_options(config.theta.cylindrical, theta=True))
    return centers.copy(), ["x", "y", "z"], ["m"] * 3


def phi_features(positions, momenta, centers, config: OpticalDataConfig):
    """Event features of a view -> ([E, D_phi] or None, names, units).

    positions, momenta: [E, 3] vertex (m) and momentum (MeV); centers: [E, 3] center of
    each event's voxel. Computed in float64 whatever the input precision.
    """
    return event_features(positions, momenta, centers, config.phi, config.theta.cylindrical)


def event_features(positions, momenta, centers, phi: PhiConfig, angles: CylindricalConfig):
    """``phi_features`` from the phi and angle settings (data views and stream transform)."""
    parts, pn, pu = [], [], []
    vertex = phi.vertex
    if vertex.representation != "omit":
        v = np.asarray(positions, dtype=float)
        names, units = ["x", "y", "z"], ["m"] * 3
        if vertex.representation == "offset":
            c = np.asarray(centers, dtype=float)
            v = v - c
            names = ["dx", "dy", "dz"]
            if vertex.coordinates == "cylindrical":
                v = to_local_frame(v, azimuth(c))
                names = ["dr_local", "dt_local", "dz"]
        elif vertex.coordinates == "cylindrical":
            v, names, units = cylindrical(v, **cylindrical_options(angles))
        parts.append(v)
        pn.extend(names)
        pu.extend(units)
    momentum = phi.momentum.representation
    if momentum != "omit":
        m = np.asarray(momenta, dtype=float)
        axes = ["x", "y", "z"]
        if phi.momentum.coordinates == "cylindrical":
            m = to_local_frame(m, momentum_azimuth(positions, centers, phi))
            axes = ["r_local", "t_local", "z"]
        names, units = [f"p{a}" if len(a) == 1 else f"p_{a}" for a in axes], ["MeV"] * 3
        if momentum == "direction_and_magnitude":
            magnitude = np.linalg.norm(m, axis=1)
            if np.any(magnitude == 0):
                raise ValueError("zero momentum cannot define direction")
            m = np.column_stack([m / magnitude[:, None], magnitude])
            names = [f"u{a}" if len(a) == 1 else f"u_{a}" for a in axes] + ["momentum_magnitude"]
            units = ["1", "1", "1", "MeV"]
        parts.append(m)
        pn.extend(names)
        pu.extend(units)
    return (np.concatenate(parts, axis=1) if parts else None), pn, pu


def stream_features(records, transform: StreamTransformConfig):
    """Stream records [M, >= 6] (x, y, z, px, py, pz first) -> (float32 [M, W], names, units).

    Reference (numpy) version of ``stream_features_torch``; stream positions are absolute.
    """
    records = np.asarray(records)
    values, names, units = event_features(
        records[:, :3], records[:, 3:6], None, transform.phi, transform.cylindrical
    )
    return values.astype(np.float32), names, units


def stream_feature_names(transform: StreamTransformConfig):
    """(names, units) of the transformed stream columns, without data."""
    return stream_features(np.zeros((0, 6), dtype=np.float32), transform)[1:]


def stream_features_torch(records, transform: StreamTransformConfig):
    """``stream_features`` in torch on the records' device: float64 inside, float32 [M, W] out."""
    import torch

    phi, angles = transform.phi, transform.cylindrical
    position = records[:, :3].to(torch.float64)
    x, y = position[:, 0], position[:, 1]
    r = torch.hypot(x, y)
    vertex_azimuth = torch.where(r == 0, torch.zeros_like(r), torch.atan2(y, x))
    parts = []
    if phi.vertex.representation == "absolute":
        if phi.vertex.coordinates == "cylindrical":
            angle = vertex_azimuth
            if angles.rotational_symmetry > 1:  # wrapped to (-pi, pi] as np.angle
                n = angles.rotational_symmetry
                angle = torch.atan2(torch.sin(n * angle), torch.cos(n * angle))
            if angles.angle_encoding == "sin_cos":
                weight = r if angles.angle_weighting == "radius" else torch.ones_like(r)
                encoded = [weight * torch.sin(angle), weight * torch.cos(angle)]
            else:
                encoded = [angle]
            parts.append(torch.stack([r, *encoded, position[:, 2]], dim=1))
        else:
            parts.append(position)
    momentum = phi.momentum
    if momentum.representation != "omit":
        m = records[:, 3:6].to(torch.float64)
        if momentum.coordinates == "cylindrical":
            co, si = torch.cos(vertex_azimuth), torch.sin(vertex_azimuth)
            m = torch.stack(
                [m[:, 0] * co + m[:, 1] * si, -m[:, 0] * si + m[:, 1] * co, m[:, 2]], dim=1
            )
        if momentum.representation == "direction_and_magnitude":
            magnitude = torch.linalg.vector_norm(m, dim=1, keepdim=True)
            m = torch.cat([m / magnitude, magnitude], dim=1)  # zero momentum: not in photon data
        parts.append(m)
    return torch.cat(parts, dim=1).to(torch.float32)


def feature_info(config: OpticalDataConfig) -> dict:
    """Feature names, units and dimensions, without data."""
    _, tn, tu = theta_features(np.zeros((0, 3)), config)
    empty = np.zeros((0, 3))
    _, pn, pu = phi_features(empty, empty, empty, config)
    return {
        "theta": {
            "names": tn, "units": tu, "dimension": len(tn), "model_input": config.theta.model_input
        },
        "phi": {"names": pn, "units": pu, "dimension": len(pn)},
    }


def make_features(run: OpticalRun, config: OpticalDataConfig):
    c = center(run, config)
    theta = theta_features(c[None], config)[0][0]
    try:
        phi = phi_features(
            run.positions, run.momenta, np.broadcast_to(c, (run.n_events, 3)), config
        )[0]
    except ValueError as exc:
        raise ValueError(f"{run.file}: {exc}") from exc
    return theta, phi, feature_info(config)


def make_labels(run: OpticalRun, config: OpticalDataConfig):
    ids = run.hit_event_ids
    if config.target.kind == "single_channel":
        ids = ids[run.detector_ids == config.target.detector_id]
    return np.isin(run.event_ids, ids).astype(np.int8)


class RunningMoments:
    """Count, mean, variance, minimum and maximum of feature rows, added in chunks.

    Chunks are merged with Chan's parallel update, so the result does not depend on how the
    rows are divided and the full training set never has to be in memory at once.
    """

    def __init__(self):
        self.n, self.mean, self.m2, self.min, self.max = 0, None, None, None, None

    def add(self, values) -> "RunningMoments":
        values = np.asarray(values, dtype=float)  # [E, D]
        n = len(values)
        if not n:
            return self
        mean = values.mean(0)
        m2 = np.square(values - mean).sum(0)
        low, high = values.min(0), values.max(0)
        if self.n == 0:
            self.n, self.mean, self.m2, self.min, self.max = n, mean, m2, low, high
            return self
        total = self.n + n
        delta = mean - self.mean
        self.mean = self.mean + delta * (n / total)
        self.m2 = self.m2 + m2 + np.square(delta) * (self.n * n / total)
        self.min, self.max = np.minimum(self.min, low), np.maximum(self.max, high)
        self.n = total
        return self

    def merge(self, other: "RunningMoments") -> "RunningMoments":
        """Add the rows summarized by ``other`` (Chan's parallel update; exact)."""
        if other.n == 0:
            return self
        if self.n == 0:
            self.n, self.mean, self.m2 = other.n, np.array(other.mean), np.array(other.m2)
            self.min, self.max = np.array(other.min), np.array(other.max)
            return self
        total = self.n + other.n
        delta = np.asarray(other.mean) - self.mean
        self.mean = self.mean + delta * (other.n / total)
        self.m2 = self.m2 + np.asarray(other.m2) + np.square(delta) * (self.n * other.n / total)
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)
        self.n = total
        return self

    def to_dict(self) -> dict:
        return dict(
            n=int(self.n),
            **{k: np.asarray(getattr(self, k)).tolist() for k in ("mean", "m2", "min", "max")},
        )

    @classmethod
    def from_dict(cls, values: dict) -> "RunningMoments":
        moments = cls()
        moments.n = int(values["n"])
        for k in ("mean", "m2", "min", "max"):
            setattr(moments, k, np.asarray(values[k], dtype=float))
        return moments


@dataclass
class AffineTransform:
    method: str
    offset: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, values, method):
        return cls.from_moments(RunningMoments().add(values), method)

    @classmethod
    def from_moments(cls, moments: RunningMoments, method):
        if not moments.n:
            raise ValueError("Cannot fit normalization without training values")
        ptp = moments.max - moments.min
        if method == "standard":
            offset, scale = moments.mean, np.sqrt(moments.m2 / moments.n)
        elif method == "minmax":
            offset, scale = moments.min, ptp
        elif method == "none":
            offset, scale = np.zeros_like(ptp), np.ones_like(ptp)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        scale = np.where(ptp == 0, 1, scale)
        return cls(method, offset.tolist(), scale.tolist())

    def transform(self, values):
        return (values - np.asarray(self.offset)) / np.asarray(self.scale)

    def to_dict(self):
        return asdict(self)
