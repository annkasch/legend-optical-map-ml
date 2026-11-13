# Design

How the code fits together, where to make a change, and why the rules in `CLAUDE.md` exist.
Module docstrings hold the file formats.

## Data flow

```
voxel campaigns: remage *.stp.lh5 (vtx, stp/optical), one file per voxel
  reader.py       validate files, voxel centre from the file name (or CSV x_mm/y_mm/z_mm)
  event_cache.py  one shared float32 copy per file: events [n,6], labels, hits (read once)
  split.py        roles per voxel (folder fractions/counts or CSV manifests), overlap checks
  features.py     theta/phi features, streaming moments, training-only affine normalization
  view.py         per data config: view.npy + view.json; OpticalView loads partitions
  pipeline.py     prepare_optical_data, ensure_prepared_optical_data, load_partition
  provider.py     OpticalDataset -> RESUM_FLEX: core.surrogates train | predict | mfgp | binomial
homogeneous campaigns: photon_store.py -> photon_loader.PhotonStream -> core.surrogates stream
finished runs -> benchmark.py -> report.md, benchmark.{csv,json,png}
prepared view -> scripts/export_{counts,photons}.py -> countfield
BLL run + prepared view -> RESUM_FLEX score (provider.transfer_events) -> <run>/score_*/
```

## Where to change what

| Change | File |
|---|---|
| A data-YAML field or its validation | `config.py` (`OpticalDataConfig`, `PhotonStoreConfig`), then `docs/data-config.md` |
| File-name parsing, LH5 checks, file selection, source stamps | `reader.py` |
| Role assignment, manifests, allowed overlaps | `split.py`, roles in `config.py` (`ROLES`, `SHARED_VALIDATION_ROLES`) |
| Input features or labels | `features.py` (`theta_features`, `phi_features`, `make_labels`) |
| What a prepared view stores or how batches are built | `view.py` (bump `VERSION`) |
| Preparation and automatic re-preparation | `pipeline.py` |
| What RESUM_FLEX sees (partitions, unlabeled inputs) | `provider.py` |
| Cache format or update logic | `event_cache.py` (bump `VERSION`) |
| Photon store format or the training loader | `photon_store.py`, `photon_loader.py` |
| Benchmark metrics, model kinds, compute accounting | `benchmark.py`, then `docs/benchmark.md` |
| Split audit tables | `reports.py` |
| Model, training, GP or metric behaviour | RESUM_FLEX, not here |

Tests: `tests/test_<module>.py`-style files with synthetic LH5 fixtures
(`test_optical_pipeline.py` covers preparation, views and the provider).

## Why the rules exist

- **One cache, many views.** Raw files are large and on shared file systems; reading them
  once and computing features at load time means a feature change never rereads them.
- **Roles from the sampling, not from repairs.** Voxel lists come from the person who ran the
  simulation; leakage checks reject overlaps except the designed ones (nested GP design,
  shared held-out sets). Overlaps are fixed at sampling time, not by split rules.
- **Raw outputs, not maps.** Reboost maps carry the same per-voxel hit ratio; the raw
  outputs also keep per-event inputs.
- **Partly filled voxels stay.** Voxels overlapping a Ge detector are physical; filtering them
  would bias the map near detectors.
- **Same data for every rate model.** The benchmark compares models, so they must see the
  same training voxels and be scored on the same held-out voxels (`docs/benchmark.md`).
