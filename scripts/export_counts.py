"""Export per-voxel hit counts of prepared optical data as a countfield count-set file.

    python scripts/export_counts.py DATA_DIR OUT.npz gp_unlabeled/lf gp_train/hf gp_validation/hf

Writes x [P, 3] voxel centres (m), k [P] hits, n [P] events and group [P] = "partition/fidelity"
(the format of countfield.data.save_groups). Groups named *test* are refused unless
--unblind-test is given, so blinded test sets cannot reach a training run by accident.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from legend_optical_map_ml.provider import OpticalDataset


def counts(provider: OpticalDataset, partition: str, fidelity: str):
    batch, provenance = provider.load(partition, fidelity)
    hits = np.where(batch.event_mask, batch.labels, 0).sum(axis=1).astype(np.int64)  # [B]
    events = np.asarray(batch.events_per_design_point, dtype=np.int64)  # [B]
    centres = np.asarray(provider.coordinates(partition, fidelity), dtype=float)  # [B, 3] m
    if len(centres) != len(hits):
        raise RuntimeError(f"{partition}/{fidelity}: {len(centres)} centres, {len(hits)} voxels")
    return centres, hits, events, provenance


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
    xs, ks, ns, labels, summary = [], [], [], [], {}
    for group in args.groups:
        partition, fidelity = group.split("/")
        x, k, n, provenance = counts(provider, partition, fidelity)
        xs.append(x)
        ks.append(k)
        ns.append(n)
        labels.append(np.full(len(k), group))
        summary[group] = dict(voxels=len(k), hits=int(k.sum()), events=int(n.sum()),
                              data=provenance)
        print(f"{group}: {len(k)} voxels, {int(k.sum())} hits, {int(n.sum())} events", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, x=np.concatenate(xs), k=np.concatenate(ks), n=np.concatenate(ns),
        group=np.concatenate(labels).astype(str),
        source=str(args.data_directory.resolve()), units="m",
    )
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
