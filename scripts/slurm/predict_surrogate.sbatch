#!/usr/bin/env bash
# Run context-free mean prediction on a Perlmutter CPU node:
#   sbatch scripts/slurm/predict_surrogate.sbatch CHECKPOINT DATA_DIR PARTITION [options]
# The environment defaults to this checkout's .venv/bin/python; set RESUM_PYTHON to override.
#SBATCH --account=m2676
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --job-name=surrogate-predict
#SBATCH --output=outputs/slurm_logs/%x-%j.out
set -euo pipefail

checkpoint=${1:?usage: predict_surrogate.sbatch CHECKPOINT DATA_DIR PARTITION [options]}
data_directory=${2:?usage: predict_surrogate.sbatch CHECKPOINT DATA_DIR PARTITION [options]}
partition=${3:?usage: predict_surrogate.sbatch CHECKPOINT DATA_DIR PARTITION [options]}
shift 3
cd "$SLURM_SUBMIT_DIR"

threads=${BLAS_THREADS:-1}
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
python_exe=${RESUM_PYTHON:-.venv/bin/python}
if [[ ! -x "$python_exe" ]]; then
    echo "Python interpreter '$python_exe' not found or not executable; see README setup." >&2
    exit 2
fi

echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), predict $partition, $threads BLAS threads"
srun --cpu-bind=none "$python_exe" -u -m core.surrogates predict \
    "$checkpoint" "$data_directory" "$partition" "$@"
