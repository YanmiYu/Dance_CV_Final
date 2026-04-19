#!/bin/bash

# ============================================================
# Dance Choreography Practice Tool
# CSCI1430 - Computer Vision — Final Project
# Brown University
# SLURM job script for Oscar
#
# Usage:
#   sbatch slurm_run.sh extract_all          # extract keypoints for all videos
#   sbatch slurm_run.sh analyze   phrase_01  # analyze one phrase pair
#   sbatch slurm_run.sh batch                # analyze all phrase pairs
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
#SBATCH -t 01:00:00
#SBATCH -J dance_cv
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# ============================================================
# Arguments
#   $1  task name: extract_all | analyze | batch
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

# Load Oscar modules
module load python/3.11.0
module load cuda/11.8.0
module load ffmpeg/6.0                   # needed by OpenCV for video writing

# Activate virtual environment
# Create it once with:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
source .venv/bin/activate

# ---- Dispatch ----
case "$TASK" in

  extract_all)
    echo ">>> Extracting keypoints for all videos in data/"
    python main.py extract_all \
      --data    data/ \
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

  batch)
    echo ">>> Batch analyzing all phrase pairs in data/"
    python main.py batch \
      --data         data/ \
      --out          results/ \
      --fps          15 \
      --threshold    0.25 \
      --min-duration 0.5
    ;;

  *)
    echo "ERROR: Unknown task '$TASK'"
    echo "Valid tasks: extract_all | analyze | batch"
    exit 1
    ;;

esac

EXIT_CODE=$?

echo "============================================"
echo "Finished:  $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"
exit $EXIT_CODE
