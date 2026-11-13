# Data config

A data YAML (`configs/<dataset>/data.yaml`, schema `config.py:OpticalDataConfig`)
describes the raw LH5 inputs and how they become a prepared view. Unknown keys are rejected.
Relative paths resolve from the YAML's directory, not the shell's. Example:
[`configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/data.yaml`](../configs/voxel_sets_x+365+485_y+315+435_z+297.5+452.5_5mm/data.yaml).

| Setting | Meaning |
|---|---|
| `source` | Raw LH5 files and the role of each file (below) |
| `output_directory` | Prepared view; must equal the run config's `data_directory` |
| `cache_directory` | Optional; default `<output_directory parent>/event_cache` |
| `theta` | Voxel position as model input |
| `phi` | Per-event inputs: vertex and momentum representation |
| `target` | Which hit outcome is predicted |
| `split` | `seed` and `manifest` (stored assignment, reused by later preparations) |
| `normalization` | `standard`, `minmax` or `none`, fitted on training roles only |
| `views` | Optional named input representations of the same data (below) |

## Sources and roles

Use exactly one `source` layout:

- `sources`: per folder a role assignment (the usual layout);
- `manifests`: CSV file lists, one role (or `fractions`) per CSV;
- `directories`: lists of folders per fidelity (`lf`, `hf`), split by the global `split` fields;
- `directory`: one root with `lf/` and `hf/` (old form).

Roles: `cnp_train`, `cnp_validation`, `cnp_test` (labeled LF event-model data); `gp_train`
(observed GP data, commonly HF); `gp_unlabeled` (inputs-only LF voxels, no optical hits needed,
see `docs/preparation.md`); `gp_validation`, `gp_test` (held-out GP data). Roles left out get
no files.

```yaml
source:
  sources:
  - path: /data/lf                 # fidelity: lf by default
    fractions: {cnp_train: 0.6, cnp_validation: 0.1, gp_train: 0.3}   # sum to 1
  - path: /data/lf_batch2
    counts: {cnp_train: 5000, cnp_validation: 1000, gp_train: rest}   # one role may take rest
  - path: /data/hf
    fidelity: hf
    fractions: {gp_train: 1.0}     # an already split folder is one role at 1.0
  exclude_files: [lf/run00782_....stp.lh5]   # relative to the source folder; bad files only
```

Manifests: each CSV lists files in an `stp_file` column (relative to the CSV). Fidelity comes
from a `tier` column (`lf`/`hf`) or the entry's `fidelity`. `x_mm`, `y_mm`, `z_mm`, if present,
are the exact voxel centre (else parsed from the file name). Assignments are used as given; a
CSV or a raw file may appear only once (don't add `voxels_all.csv` next to its parts).
`fractions` on a manifest splits it seeded, e.g. `{gp_validation: 0.5, gp_test: 0.5}`.

Splitting keeps whole voxels together and is seeded per folder; with `sources` or `manifests`
the role assignment is the split, so only `split.seed` and `split.manifest` apply. A voxel
position in two roles is an error, except `gp_train` + `gp_unlabeled` (nested multi-fidelity
design) and among `cnp_validation`, `gp_validation`, `gp_test` (all held out).

With `directories`, each folder must directly contain `.stp.lh5` files; repeated references
to one physical file load once; the same basename twice per fidelity, or one file in both
fidelities, is rejected. Event counts may differ between voxels (ragged partitions are
zero-padded with per-voxel `event_counts`).

## Features, target, normalization

- `theta.coordinates`: `cartesian` or `cylindrical` (angle encoding, azimuth on/off).
  `center_source: filename` (nominal centre; CSV coordinates override) or `observed_midrange`.
- `theta.model_input: false`: the voxel centre is no network input; batches are
  `EVENT_ONLY` (θ None, one row per voxel as before). θ still groups voxels for the split,
  defines offsets and stays available as design-point metadata (`provider.theta()`,
  `coordinates()`, normalization fitted as usual). Needs phi inputs. The field model needs θ:
  event-only views are for mlp / mlp_bll. Their `predict` output carries the provider's θ
  (voxel centres), so GP stages (binomial covariate, MFGP level) use it as for any model.
- `phi.vertex.representation`: `offset` from the centre, absolute, or omitted;
  `phi.momentum.representation`: `components`, direction plus magnitude, or omitted.
- `target.kind: any_channel` (an event is positive with a hit in any channel) or
  `single_channel` with `detector_id`.
- `normalization.fit_fidelities` optionally restricts which fidelities enter the statistics.
  Inference checks that the checkpoint's normalization equals the view's.

## Views

The top-level `theta`, `phi` and `normalization` define the view at `output_directory`.
`views` adds named views of the same files, roles and split, each with its own features and
normalization, written to `<output_directory>/views/<name>/`:

```yaml
views:
  absolute:                        # name: lower case, digits, underscores
    phi:
      vertex: {representation: absolute}
```

- A view sets any of `theta`, `phi`, `normalization`; unset ones take the top-level value.
  Source, target, split and cache are shared.
- A view's `theta` must group voxels like the top-level one (same `center_source`, same
  azimuth setting), because the split is assigned once for all views.
- A run picks its view with `data_directory` (`.../data` or `.../data/views/<name>`); the
  run's `experiment.json` records it (`view`).
- Adding a view or changing one rewrites only that view; the others, their provenance and the
  split stay unchanged (`docs/preparation.md`).
