#!/bin/bash
# Run training or standalone MFGP on a Perlmutter CPU node, from the repository root:
#   sbatch scripts/slurm/train_surrogate.sbatch configs/CONFIG.yaml [train|mfgp|binomial|within_point|residual]
# Install resum-flex and this package into the same environment as described in README.md.
# The default interpreter is this checkout's .venv/bin/python; set RESUM_PYTHON to an
# executable path to override.
# BLAS threads default to 1 (training); GP fits on thousands of voxels profit from more:
#   BLAS_THREADS=32 sbatch scripts/slurm/train_surrogate.sbatch GP.yaml mfgp
# Defaults: account m2676, debug QOS, 15 min. Override on the command line, e.g.
#   sbatch -q regular -t 02:00:00 scripts/slurm/train_surrogate.sbatch CONFIG.yaml
#SBATCH --account=m2676
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --job-name=resum-train
#SBATCH --output=outputs/slurm_logs/%x-%j.out
set -euo pipefail
config=${1:?usage: sbatch scripts/slurm/train_surrogate.sbatch CONFIG.yaml [train|mfgp|binomial|within_point|residual]}
command=${2:-train}
case "$command" in
    train|mfgp|binomial|within_point|residual) ;;
    *) echo "Unknown command '$command'; choose train, mfgp, binomial, within_point or residual." >&2; exit 2 ;;
esac
cd "$SLURM_SUBMIT_DIR"
# Training is single-threaded (set in core.surrogates); evaluation_threads sets the eval pool.
threads=${BLAS_THREADS:-1}
export OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
python_exe=${RESUM_PYTHON:-.venv/bin/python}
if [[ ! -x "$python_exe" ]]; then
    echo "Python interpreter '$python_exe' not found or not executable; see the README setup instructions." >&2
    exit 2
fi
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $command $config, $threads BLAS threads"
srun --cpu-bind=none "$python_exe" -u -m core.surrogates "$command" "$config"
