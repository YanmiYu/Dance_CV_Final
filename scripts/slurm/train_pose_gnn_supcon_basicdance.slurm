#!/bin/bash
# SLURM batch script for SupCon training of the Basic Dance pose embedding
# model on Oscar (Brown CCV).
#
# Submit from the repo root:
#     sbatch scripts/slurm/train_pose_gnn_supcon_basicdance.slurm
#
# Override the training config (e.g. epochs):
#     sbatch --export=ALL,TRAIN_CFG=configs/train/train_pose_gnn_supcon_basicdance.yaml,EXTRA_ARGS="--epochs 30" \
#            scripts/slurm/train_pose_gnn_supcon_basicdance.slurm

#SBATCH --job-name=cv_dance_supcon
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err
#SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your_email@brown.edu

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm data/processed/train_pose_gnn_supcon_basicdance checkpoints

ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"
module purge
module load "${ANACONDA_MODULE}"

# shellcheck disable=SC1091
source activate dance-cv

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TRAIN_CFG="${TRAIN_CFG:-configs/train/train_pose_gnn_supcon_basicdance.yaml}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=============================================================="
echo "job:        ${SLURM_JOB_NAME:-unknown} (${SLURM_JOB_ID:-?})"
echo "node:       $(hostname)"
echo "cwd:        $(pwd)"
echo "cfg:        ${TRAIN_CFG}"
echo "extra:      ${EXTRA_ARGS}"
echo "cpus/task:  ${SLURM_CPUS_PER_TASK:-?}"
echo "gpus:       ${SLURM_GPUS_ON_NODE:-?}"
echo "=============================================================="
nvidia-smi || true
python -c "import torch, sys; print('python:', sys.version.split()[0]); \
print('torch: ', torch.__version__); \
print('cuda:  ', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=============================================================="

srun --unbuffered python -m src.train.train_pose_gnn_supcon \
    --config "${TRAIN_CFG}" ${EXTRA_ARGS}
