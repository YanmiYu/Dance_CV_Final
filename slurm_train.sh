#!/bin/bash

# ============================================================
# SimpleBaseline Training Job
# CSCI1430 - Computer Vision — Final Project
# Brown University
#
# Usage:
#   sbatch slurm_train.sh
#
# Monitor:
#   myq
#   cat slurm-<jobid>.out
#   cat slurm-<jobid>.err
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J simple_baseline_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

# Load Oscar modules
module load python/3.11.0
module load cuda/11.8.0

# Activate virtual environment
source .venv/bin/activate

# Path to COCO on Oscar — ask your TA if unsure
COCO_ROOT="/oscar/data/shared/coco"

python src/train.py \
  --coco_root  "$COCO_ROOT" \
  --out_dir    checkpoints/simple_baseline \
  --epochs     30 \
  --batch_size 32 \
  --lr         1e-3 \
  --workers    4

EXIT_CODE=$?

echo "============================================"
echo "Finished:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
