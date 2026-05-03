#!/bin/bash

# ============================================================
# Dance Choreography Practice Tool
# CSCI1430 - Computer Vision — Final Project
# Brown University
# SLURM job script for Oscar
#
# Usage:
#   sbatch slurm_run.sh download             # Step 1: download videos from CSV → data/videos/
#   sbatch slurm_run.sh extract_all          # Step 2: extract keypoints  → data/keypoints/
#   sbatch slurm_run.sh build_dataset        # Step 3: build train/val/test .npz pairs
#   sbatch slurm_run.sh train                # Step 4: train the LSTM TemporalErrorDetector
#   sbatch slurm_run.sh test                 # Step 5: evaluate on the test split
#   sbatch slurm_run.sh analyze   phrase_01  # Inference: analyze one phrase pair
#   sbatch slurm_run.sh batch                # Inference: analyze all phrase pairs
#
# Monitor your job:
#   myq                      # check job status
#   cat slurm-<jobid>.out    # view stdout
#   cat slurm-<jobid>.err    # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J dance_cv
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# ============================================================
# Arguments
#   $1  task name: extract_all | build_dataset | train | test | analyze | batch
#   $2  (analyze only) phrase directory name, e.g. phrase_01
# ============================================================
TASK=${1:-extract_all}
PHRASE=${2:-phrase_01}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $TASK"
echo "Phrase:    $PHRASE"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# ---- Environment setup ----
cd "$SLURM_SUBMIT_DIR"

# Load Oscar modules (use module spider python/cuda/ffmpeg to find exact names)
module load python/3.9.21s-rv63
module load cuda/12.1.1-txhkv4h          # update if your Oscar has a different cuda
module load ffmpeg                        # update with versioned name if needed

# Activate virtual environment
# Create it once with:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
source .venv/bin/activate

# ---- Dispatch ----
case "$TASK" in

  download)
    echo ">>> Downloading videos from scripts/filtered_gBR_sBM_c01.csv → data/videos/"
    bash scripts/download_data.sh scripts/filtered_gBR_sBM_c01.csv
    ;;

  extract_all)
    echo ">>> Extracting keypoints from data/videos/ → data/keypoints/"
    python main.py extract_all \
      --data    data/videos/ \
      --fps     15 \
      --backend mediapipe
    ;;

  analyze)
    echo ">>> Running full analysis on: $PHRASE"
    python main.py analyze \
      --benchmark    "data/${PHRASE}/keypoints/benchmark_kp.npy" \
      --learner      "data/${PHRASE}/keypoints/learner_kp.npy" \
      --bench-video  "data/${PHRASE}/benchmark.mp4" \
      --learner-video "data/${PHRASE}/learner.mp4" \
      --fps          15 \
      --threshold    0.25 \
      --min-duration 0.5 \
      --out          "results/${PHRASE}/"
    ;;

  build_dataset)
    echo ">>> Building train/val/test .npz pairs from data/keypoints/"
    python scripts/build_dataset.py \
      --kp-dir       data/keypoints/ \
      --out-train    data/train/ \
      --out-val      data/val/ \
      --out-test     data/test/
    ;;

  train)
    echo ">>> Training LSTM TemporalErrorDetector"
    python main.py train \
      --train-dir    data/train/ \
      --val-dir      data/val/ \
      --checkpoint   checkpoints/best_model.pt \
      --epochs       30 \
      --lr           1e-3 \
      --hidden       64 \
      --layers       2 \
      --dropout      0.3 \
      --batch-size   16 \
      --log          results/training_log.csv
    ;;

  test)
    echo ">>> Evaluating on held-out test split"
    python main.py test \
      --test-dir     data/test/ \
      --checkpoint   checkpoints/best_model.pt \
      --out          results/test_metrics.json
    ;;

  batch)
    echo ">>> Batch analyzing all phrase pairs in data/"
    python main.py batch \
      --data         data/ \
      --out          results/ \
      --fps          15 \
      --checkpoint   checkpoints/best_model.pt \
      --min-duration 0.5
    ;;

  *)
    echo "ERROR: Unknown task '$TASK'"
    echo "Valid tasks: download | extract_all | build_dataset | train | test | analyze | batch"
    exit 1
    ;;

esac

EXIT_CODE=$?

echo "============================================"
echo "Finished:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
