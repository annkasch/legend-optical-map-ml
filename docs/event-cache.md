# Shared event cache

Raw LH5 files are read once into a cache shared by all data configs (`event_cache.py`;
layout in its docstring): by default `<output_directory parent>/event_cache`, or
`cache_directory` in the data YAML.

- Per file: float32 vertex and momentum `[n, 6]`, the any-channel label, and one hit row per
  (event, detector) with the number of hits. Features are computed at load time (≈ 0.2 µs per
  event), so feature changes never rewrite the cache.
- `index.npy`: one row per file (folder, name, size, mtime, sha256, voxel centre, event and
  hit counts, largest `n_part`). `cache.json` lists folders and shards.
- Updates read only new or changed files (size, mtime). One worker process per CPU, each
  writing one shard. A changed file goes into a new shard; its old row is marked invalid.

```bash
python -m legend_optical_map_ml.event_cache update DATA.yaml [--workers N]   # large inputs: compute node
python -m legend_optical_map_ml.event_cache compact CACHE_DIR                # merge shards, drop invalid rows
```

One writer at a time: updates take a `.lock` directory. After a crashed writer, remove the
lock by hand.
