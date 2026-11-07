#!/bin/bash
# Train a neural field on a photon store on Perlmutter GPU nodes, from the repository root:
#   sbatch scripts/slurm/train_photon_field.sbatch configs/photon_<volume>/field_CONFIG.yaml
# Uses the NERSC pytorch module (CUDA PyTorch) with RESUM_FLEX and this package on PYTHONPATH
# (RESUM_FLEX checkout: RESUM_FLEX_ROOT, default ../RESUM_FLEX). Several GPUs per node:
#   sbatch --gpus-per-node=4 scripts/slurm/train_photon_field.sbatch CONFIG.yaml
# runs one rank per GPU with torchrun (DistributedDataParallel).
# Defaults: account m2676, debug QOS, 15 min, 1 GPU. Real runs: -q regular -t HH:MM:SS.
#SBATCH --account=m2676
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --job-name=photon-field
#SBATCH --output=outputs/slurm_logs/%x-%j.out
set -euo pipefail
config=${1:?usage: sbatch scripts/slurm/train_photon_field.sbatch CONFIG.yaml}
cd "$SLURM_SUBMIT_DIR"
module load pytorch
resum=$(realpath "${RESUM_FLEX_ROOT:-../RESUM_FLEX}")
export PYTHONPATH="$resum:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
gpus=${SLURM_GPUS_PER_NODE:-1}
gpus=${gpus##*:}
echo "job $SLURM_JOB_ID on $(hostname), $gpus GPU(s), commits $(git rev-parse --short HEAD) / $(git -C "$resum" rev-parse --short HEAD), $config"
nvidia-smi -L
if [[ "$gpus" -gt 1 ]]; then
    srun --cpu-bind=none torchrun --standalone --nproc-per-node="$gpus" \
        -m core.surrogates stream "$config"
else
    srun --cpu-bind=none python -u -m core.surrogates stream "$config"
fi
