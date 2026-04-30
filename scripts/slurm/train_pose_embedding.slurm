#!/bin/bash
#SBATCH --job-name=dance-gnn
#SBATCH --output=logs/slurm/gnn_%j.out
#SBATCH --error=logs/slurm/gnn_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /users/hding33/Desktop/Dance_CV_Final

source scripts/slurm/activate_env.sh

python -m src.train.train_pose_embedding \
  --keypoints-dir data/labels/aistpp/keypoints2d_raw \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-3 \
  --negative-strategy different_video \
  --device cpu \
  --out checkpoints/pose_gnn_encoder_oscar.pt

