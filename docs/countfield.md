# Exports for countfield

countfield (`../countfield`, separate repository, no RESUM_FLEX imports) holds the
neural-process and Point Transformer voxel models. It reads files exported from a prepared
view here; formats are in the script docstrings.

```bash
python scripts/export_counts.py DATA_DIR OUT.npz gp_unlabeled/lf gp_train/hf gp_validation/hf
python scripts/export_photons.py DATA_DIR OUT.npz gp_unlabeled/lf gp_train/hf gp_validation/hf
```

- `export_counts.py`: per voxel centre, hits k and events n, grouped by "partition/fidelity".
- `export_photons.py`: additionally every photon per voxel (offset from the centre in mm,
  direction, energy, hit flag) in CSR layout.
- Groups named `*test*` are refused unless `--unblind-test` is given.
- Run configs: `configs/countfield.lbnl_secondment*.yaml` (read by countfield). Its
  predictions enter the benchmark as `kind: countfield` (`docs/benchmark.md`).
