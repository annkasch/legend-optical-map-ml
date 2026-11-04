#!/usr/bin/env bash
# Submit training -> mean prediction -> MFGP with afterok dependencies.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

train_config=${1:?usage: submit_surrogate_pipeline.sh TRAIN_CONFIG.yaml GP_CONFIG.yaml [SBATCH_OPTIONS...]}
gp_config=${2:?usage: submit_surrogate_pipeline.sh TRAIN_CONFIG.yaml GP_CONFIG.yaml [SBATCH_OPTIONS...]}
shift 2
scheduler_options=("$@")

python_exe=${RESUM_PYTHON:-.venv/bin/python}
if [[ ! -x "$python_exe" ]]; then
    echo "Python interpreter '$python_exe' not found or not executable; see README setup." >&2
    exit 2
fi
mkdir -p outputs/slurm_logs

# Resolve paths relative to each YAML file, matching RESUM_FLEX config loading.
plan=$("$python_exe" - "$train_config" "$gp_config" <<'PY'
from pathlib import Path
import sys

import yaml


def load(path):
    path = Path(path).expanduser().resolve()
    with path.open() as stream:
        return path, yaml.safe_load(stream) or {}


def resolve(config_path, value, key):
    if not value:
        raise ValueError(f"{config_path}: missing {key}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_path.parent / path).resolve()


train_path, train = load(sys.argv[1])
gp_path, gp = load(sys.argv[2])
run_dir = resolve(train_path, train.get("output_directory"), "output_directory")
train_data = resolve(train_path, train.get("data_directory"), "data_directory")
gp_data = resolve(gp_path, gp.get("data_directory"), "data_directory")
if train_data != gp_data:
    raise ValueError(f"training data directory {train_data} differs from GP data directory {gp_data}")

model_kind = train.get("model", {}).get("kind")
if model_kind not in {"mlp", "transformer", "bdt"}:
    raise ValueError("mean prediction requires a context-free model: mlp, transformer, or bdt")

model_levels = [level for level in gp.get("levels", []) if level.get("source") == "model_mean"]
if len(model_levels) != 1:
    raise ValueError("automated pipeline requires exactly one model_mean level in the GP config")
level = model_levels[0]
partition = level.get("partition")
fidelity = level.get("fidelity", "lf")
if not partition:
    raise ValueError("model_mean level must define a partition")

inference = resolve(gp_path, level.get("inference"), "model_mean inference")
checkpoint_name = inference.parent.name
checkpoint = run_dir / "checkpoints" / checkpoint_name
expected_inference = run_dir / "inference" / checkpoint_name / f"{partition}_{fidelity}.npz"
if inference != expected_inference:
    raise ValueError(
        f"GP inference path must match prediction output {expected_inference}; configured path is {inference}"
    )
if checkpoint_name not in {"best", "final"}:
    raise ValueError("model_mean inference path must use a 'best' or 'final' checkpoint")

if run_dir.exists() and any(run_dir.iterdir()):
    raise FileExistsError(f"training output directory is not empty: {run_dir}")
if inference.exists() or inference.with_suffix(".json").exists():
    raise FileExistsError(f"prediction output already exists: {inference}")

print("\t".join(map(str, (train_path, gp_path, checkpoint, gp_data, partition, fidelity))))
PY
)
IFS=$'\t' read -r train_config gp_config checkpoint data_directory partition fidelity <<< "$plan"

train_result=$(sbatch --parsable "${scheduler_options[@]}" \
    --job-name=surrogate-train \
    scripts/slurm/train_surrogate.sbatch "$train_config" train)
train_job=${train_result%%;*}

predict_threads=${PREDICT_THREADS:-8}
predict_result=$(sbatch --parsable "${scheduler_options[@]}" \
    --dependency="afterok:$train_job" \
    --job-name=surrogate-predict \
    scripts/slurm/predict_surrogate.sbatch \
    "$checkpoint" "$data_directory" "$partition" --fidelity "$fidelity" --threads "$predict_threads")
predict_job=${predict_result%%;*}

# GP fits profit from many BLAS threads; training stays single-threaded.
gp_options=()
if [[ -n "${GP_TIME:-}" ]]; then
    gp_options+=(--time="$GP_TIME")
fi
gp_result=$(BLAS_THREADS=${GP_BLAS_THREADS:-32} sbatch --parsable "${scheduler_options[@]}" \
    "${gp_options[@]}" \
    --dependency="afterok:$predict_job" \
    --job-name=surrogate-mfgp \
    scripts/slurm/train_surrogate.sbatch "$gp_config" mfgp)
gp_job=${gp_result%%;*}

printf 'Submitted training job: %s\n' "$train_job"
printf 'Submitted prediction job: %s (afterok:%s)\n' "$predict_job" "$train_job"
printf 'Submitted MFGP job: %s (afterok:%s)\n' "$gp_job" "$predict_job"
