#!/bin/bash
# SLURM batch script for training the single-person detector on Oscar (Brown CCV).
#
# Entry point: `python -m src.train.train_detector --train <cfg>`.
# The detector is trained FROM SCRATCH on AIST++ bbox labels; no pretrained
# weights anywhere (docs/project_decisions.md sections 1 and 6).
# The engine auto-selects CUDA when available (see src/train/engine.py).
#
# Submit from the repo root:
#     sbatch scripts/slurm/train_detector.sbatch
#
# Override the training config:
#     sbatch --export=ALL,TRAIN_CFG=configs/train/train_detector_xyz.yaml \
#            scripts/slurm/train_detector.sbatch
#
# Quick smoke run (tiny dataset, few steps, one epoch) before the real job:
#     sbatch --export=ALL,MAX_ITEMS=256,SMOKE_STEPS=100 \
#            scripts/slurm/train_detector.sbatch
#
# See docs/oscar_runbook.md for full upload / env-setup / monitor steps.

#SBATCH --job-name=cv_dance_detector
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
# #SBATCH --mail-user=your_email@brown.edu   # uncomment + set to get mail

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/slurm data/processed/detector

# Oscar module environment. Must match setup_env.sh's ANACONDA_MODULE so the
# `dance-cv` env created there is actually on PATH here. If `module load`
# fails because CCV rotated the build-string, run
#     module -t avail 2>&1 | grep -iE 'anaconda|miniconda'
# on a login node and update BOTH this file and scripts/slurm/setup_env.sh.
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"
module purge
module load "${ANACONDA_MODULE}"

# Activate the env created by scripts/slurm/setup_env.sh.
# shellcheck disable=SC1091
source activate dance-cv

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TRAIN_CFG="${TRAIN_CFG:-configs/train/train_detector.yaml}"

# Optional debug knobs. Leave unset for a full training run.
EXTRA_ARGS=()
if [[ -n "${MAX_ITEMS:-}" ]]; then
    EXTRA_ARGS+=(--max-items "${MAX_ITEMS}")
fi
if [[ -n "${SMOKE_STEPS:-}" ]]; then
    EXTRA_ARGS+=(--smoke-steps "${SMOKE_STEPS}")
fi

echo "=============================================================="
echo "job:        ${SLURM_JOB_NAME:-unknown} (${SLURM_JOB_ID:-?})"
echo "node:       $(hostname)"
echo "cwd:        $(pwd)"
echo "cfg:        ${TRAIN_CFG}"
echo "cpus/task:  ${SLURM_CPUS_PER_TASK:-?}"
echo "gpus:       ${SLURM_GPUS_ON_NODE:-?} (gres=${SLURM_JOB_GRES:-?})"
echo "extra:      ${EXTRA_ARGS[*]:-<none>}"
echo "=============================================================="
nvidia-smi || true
python -c "import torch, sys; print('python:', sys.version.split()[0]); \
print('torch: ', torch.__version__); \
print('cuda:  ', torch.cuda.is_available(), \
torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "=============================================================="

# ${ARR[@]+"${ARR[@]}"} is the portable way to expand an array that may be
# empty under `set -u` (bash <4.4 treats "${ARR[@]}" on an empty array as
# unbound). Oscar's login nodes still ship older bash on occasion.
srun --unbuffered python -m src.train.train_detector \
    --train "${TRAIN_CFG}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
