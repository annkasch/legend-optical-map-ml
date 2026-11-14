"""Dataset provider for prepared LAr optical-map data (a view on the shared event cache).

Configs select it with ``data_provider: <package>.provider:OpticalDataset`` and may set
``data_config`` to an optical data-preparation YAML.
"""

from pathlib import Path

import numpy as np

from legend_optical_map_ml.features import decode_absolute_phi, phi_features
from legend_optical_map_ml.view import OpticalView


class OpticalDataset:
    def __init__(self, root):
        self.root = Path(root)
        self._view = None

    @property
    def view(self) -> OpticalView:
        if self._view is None:
            if not (self.root / "view.json").is_file():
                raise FileNotFoundError(f"No prepared optical data (view.json) in {self.root}")
            self._view = OpticalView(self.root)
        return self._view

    def partitions(self):
        return list(self.view.partitions)

    def load(self, partition, fidelity, *, labels=True):
        view = self.view
        batch = (
            view.load(partition, fidelity) if labels else view.load_inputs(partition, fidelity)[0]
        )
        return batch, view.provenance(partition, fidelity)

    def theta(self, partition, fidelity):
        return self.view.load_theta(partition, fidelity)

    def coordinates(self, partition, fidelity):
        """Voxel centers [B, 3] in m (filename or observed midrange, as configured)."""
        view = self.view
        selected = view.select(partition, fidelity)
        return view.centers(selected)

    def normalization(self):
        return self.view.meta["normalization"]

    def transfer_events(self, phi, source, source_rows, target):
        """Donor photon inputs moved to the target voxels (RESUM_FLEX ``predict --events-from``).

        phi [B, N, D_phi] normalized: row b holds the photons of voxel ``source_rows[b]`` of
        ``source`` = (partition, fidelity), to be placed in voxel b of ``target``. Offsets and
        momenta are relative to the voxel and stay as they are; absolute cartesian vertex
        positions are shifted by (target centre - donor centre). Cylindrical vertex or momentum
        features are decoded to cartesian, shifted and encoded again (the azimuth changes).
        """
        config = self.view.config
        vertex = config.phi.vertex
        if vertex.representation != "absolute":
            return phi
        rows = np.asarray(source_rows)
        shift = self.coordinates(*target) - self.coordinates(*source)[rows]  # [B, 3] m
        norm = self.normalization()["phi"]
        offset, scale = np.asarray(norm["offset"], float), np.asarray(norm["scale"], float)
        moved = np.array(phi, dtype=float)  # [B, N, D_phi]; padding moves too, stays masked
        if vertex.coordinates == "cartesian" and config.phi.momentum.coordinates == "cartesian":
            moved[..., :3] += (shift / scale[:3])[:, None, :]  # vertex first
            return moved.astype(np.asarray(phi).dtype)
        b, n, d = moved.shape
        donor_centres = np.repeat(self.coordinates(*source)[rows], n, axis=0)  # [B * N, 3]
        positions, momenta = decode_absolute_phi(
            (moved * scale + offset).reshape(b * n, d), config, near=donor_centres
        )
        positions = positions + np.repeat(shift, n, axis=0)
        encoded = phi_features(positions, momenta, positions, config)[0]  # centres unused
        return ((encoded - offset) / scale).reshape(b, n, d).astype(np.asarray(phi).dtype)

    @classmethod
    def prepare(cls, data_config, data_directory):
        """Rebuild ``data_directory`` from the optical data config if outdated; returns a record."""
        from legend_optical_map_ml.config import load_optical_config
        from legend_optical_map_ml.pipeline import ensure_prepared_optical_data

        preparation = load_optical_config(data_config)
        data_directory = Path(data_directory).resolve()
        directories = [v.output_directory for v in preparation.view_configs().values()]
        if data_directory not in directories:
            raise ValueError(
                f"data_directory {data_directory} is none of the views data_config writes: "
                + ", ".join(map(str, directories))
            )
        # config.json is written only by a full preparation.
        saved = data_directory / "config.json"
        before = saved.stat().st_mtime_ns if saved.exists() else None
        ensure_prepared_optical_data(preparation)
        regenerated = before != saved.stat().st_mtime_ns
        print(
            f"Prepared data {'regenerated' if regenerated else 'up to date'}: {data_directory}",
            flush=True,
        )
        return dict(
            path=str(data_config),
            regenerated=regenerated,
            settings=preparation.model_dump(mode="json"),
            view=data_directory.relative_to(preparation.output_directory).as_posix(),
        )
