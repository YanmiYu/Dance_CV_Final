#!/bin/bash

# ============================================================
# Download AIST Dance DB test clips for d04 (benchmark) and d05 (learner)
# Genre: Break (gBR), Situation: Basic Dance (sBM),
# Camera: c01 (frontal), Music: mBR0 (80 BPM)
# Choreographies: ch01–ch05  →  data/phrase_01/ … data/phrase_05/
#
# Usage:
#   bash scripts/download_data.sh
# ============================================================

BASE_URL="https://aistdancedb.ongaaccel.jp/v1.0.0/video/2M"

GENRE="gBR"
SITUATION="sBM"
CAMERA="c01"
MUSIC="mBR0"
BENCHMARK_DANCER="d04"
LEARNER_DANCER="d05"

# Number of choreography pairs to download
N_PHRASES=5

echo "========================================"
echo "AIST Dance DB — downloading $N_PHRASES pairs"
echo "  Benchmark dancer: $BENCHMARK_DANCER"
echo "  Learner dancer:   $LEARNER_DANCER"
echo "  Genre / Music:    $GENRE / $MUSIC"
echo "  Camera:           $CAMERA"
echo "========================================"

for i in $(seq 1 $N_PHRASES); do
    CH=$(printf "ch%02d" $i)
    PHRASE=$(printf "phrase_%02d" $i)
    PHRASE_DIR="data/$PHRASE"

    mkdir -p "$PHRASE_DIR/keypoints"

    BENCH_FILE="${GENRE}_${SITUATION}_${CAMERA}_${BENCHMARK_DANCER}_${MUSIC}_${CH}.mp4"
    LEARN_FILE="${GENRE}_${SITUATION}_${CAMERA}_${LEARNER_DANCER}_${MUSIC}_${CH}.mp4"

    BENCH_URL="$BASE_URL/$BENCH_FILE"
    LEARN_URL="$BASE_URL/$LEARN_FILE"

    # ---- Download benchmark ----
    echo ""
    echo "[$PHRASE] Downloading benchmark: $BENCH_FILE"
    if [ -f "$PHRASE_DIR/benchmark.mp4" ]; then
        echo "  Already exists, skipping."
    else
        curl -L --progress-bar --fail \
             -o "$PHRASE_DIR/benchmark.mp4" \
             "$BENCH_URL"
        if [ $? -ne 0 ]; then
            echo "  ERROR: Failed to download $BENCH_URL"
        else
            echo "  Saved → $PHRASE_DIR/benchmark.mp4"
        fi
    fi

    # ---- Download learner ----
    echo "[$PHRASE] Downloading learner:    $LEARN_FILE"
    if [ -f "$PHRASE_DIR/learner.mp4" ]; then
        echo "  Already exists, skipping."
    else
        curl -L --progress-bar --fail \
             -o "$PHRASE_DIR/learner.mp4" \
             "$LEARN_URL"
        if [ $? -ne 0 ]; then
            echo "  ERROR: Failed to download $LEARN_URL"
        else
            echo "  Saved → $PHRASE_DIR/learner.mp4"
        fi
    fi
done

echo ""
echo "========================================"
echo "Done. Directory structure:"
find data/phrase_* -name "*.mp4" | sort
echo "========================================"
echo ""
echo "Next step — extract keypoints:"
echo "  python main.py extract_all --data data/ --fps 15"
