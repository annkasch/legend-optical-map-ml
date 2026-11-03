"""Export all photons per voxel of prepared optical data as a countfield photon file.

    python scripts/export_photons.py DATA_DIR OUT.npz gp_unlabeled/lf gp_train/hf gp_validation/hf

Per voxel: x [V, 3] centre (m), k [V] hits, n [V] photons, group [V] = "partition/fidelity".
Per photon (CSR, voxel v owns rows photon_offsets[v]:photon_offsets[v+1]):
photon_features [P, 7] float32 = emission offset from the voxel centre (dx, dy, dz in mm),
unit direction (ux, uy, uz) and energy (eV); photon_hit [P] int8 (any channel). This is the
format of countfield.photons.save_photon_groups; its x/k/n/group part is also a count-set file.
Groups named *test* are refused unless --unblind-test is given.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from legend_optical_map_ml.provider import OpticalDataset

FEATURES = ["dx_mm", "dy_mm", "dz_mm", "ux", "uy", "uz", "energy_eV"]


def photons(provider: OpticalDataset, partition: str, fidelity: str):
    view = provider.view
    if view.config.target.kind != "any_channel":
        raise ValueError("export_photons supports target.kind any_channel only")
    selected = view.select(partition, fidelity)
    rows = view.cache.locate(view.paths(selected), selected["sha256"])
    centres = np.asarray(view.centers(selected), dtype=float)  # [V, 3] m
    features, hits, counts = [], [], []
    for row, centre in zip(rows, centres, strict=True):
        events = view.cache.events(row)  # [E, 6]: position (m), momentum (MeV)
        momentum = events[:, 3:].astype(float)
        magnitude = np.linalg.norm(momentum, axis=1)
        if np.any(magnitude == 0):
            raise ValueError(f"{view.cache.path(row)}: zero photon momentum")
        features.append(
            np.column_stack(
                [
                    (events[:, :3] - centre) * 1e3,
                    momentum / magnitude[:, None],
                    magnitude * 1e6,
                ]
            ).astype(np.float32)
        )
        labels = np.asarray(view.cache.labels(row), dtype=np.int8)
        if len(labels) != len(events):
            raise RuntimeError(
                f"{view.cache.path(row)}: {len(labels)} labels, {len(events)} events"
            )
        hits.append(labels)
        counts.append(len(events))
    return (
        centres,
        features,
        hits,
        np.asarray(counts, dtype=np.int64),
        view.provenance(partition, fidelity),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("groups", nargs="+", help="partition/fidelity, e.g. gp_train/hf")
    parser.add_argument("--unblind-test", action="store_true")
    args = parser.parse_args()
    blinded = [g for g in args.groups if "test" in g]
    if blinded and not args.unblind_test:
        parser.error(f"blinded test groups {blinded}; pass --unblind-test to export them")
    if args.output.exists():
        parser.error(f"{args.output} exists; choose a new file")

    provider = OpticalDataset(args.data_directory)
    xs, ks, ns, labels, feats, hits, summary = [], [], [], [], [], [], {}
    for group in args.groups:
        partition, fidelity = group.split("/")
        x, f, h, n, provenance = photons(provider, partition, fidelity)
        k = np.array([int(v.sum()) for v in h], dtype=np.int64)
        xs.append(x)
        ks.append(k)
        ns.append(n)
        labels.append(np.full(len(k), group))
        feats += f
        hits += h
        summary[group] = dict(
            voxels=len(k), hits=int(k.sum()), photons=int(n.sum()), data=provenance
        )
        print(f"{group}: {len(k)} voxels, {int(k.sum())} hits, {int(n.sum())} photons", flush=True)
    n_all = np.concatenate(ns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        x=np.concatenate(xs),
        k=np.concatenate(ks),
        n=n_all,
        group=np.concatenate(labels).astype(str),
        photon_offsets=np.concatenate([[0], np.cumsum(n_all)]).astype(np.int64),
        photon_features=np.concatenate(feats),
        photon_hit=np.concatenate(hits),
        feature_names=np.array(FEATURES),
        source=str(args.data_directory.resolve()),
        units="m",
    )
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
