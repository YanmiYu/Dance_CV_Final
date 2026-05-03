#!/bin/bash
# ============================================================
# Download AIST Dance DB videos listed in filtered_gBR_sBM_c01.csv
# directly to data/videos/ on Oscar (or any Linux machine).
#
# Usage (from project root):
#   bash scripts/download_data.sh
#   bash scripts/download_data.sh scripts/filtered_gBR_sBM_c01.csv  # custom CSV
#
# Next step after this completes:
#   sbatch slurm_run.sh extract_all    # extract keypoints from every video
# ============================================================

CSV_FILE="${1:-scripts/filtered_gBR_sBM_c01.csv}"
OUT_DIR="data/videos"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV not found: $CSV_FILE"
    exit 1
fi

mkdir -p "$OUT_DIR"

TOTAL=$(grep -c . "$CSV_FILE")
COUNT=0
N_OK=0
N_SKIP=0
N_FAIL=0

echo "============================================"
echo "Downloading $TOTAL videos → $OUT_DIR"
echo "CSV: $CSV_FILE"
echo "============================================"

while IFS= read -r URL || [ -n "$URL" ]; do
    [ -z "$URL" ] && continue
    FILENAME=$(basename "$URL")
    DEST="$OUT_DIR/$FILENAME"
    COUNT=$((COUNT + 1))

    if [ -f "$DEST" ]; then
        echo "[$COUNT/$TOTAL] skip  $FILENAME"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    echo "[$COUNT/$TOTAL] fetch $FILENAME"
    curl --silent --show-error --fail --location \
         --retry 3 --retry-delay 2 \
         -o "$DEST" "$URL"

    if [ $? -eq 0 ]; then
        N_OK=$((N_OK + 1))
    else
        echo "  ERROR: failed → $URL"
        rm -f "$DEST"
        N_FAIL=$((N_FAIL + 1))
    fi
done < "$CSV_FILE"

echo ""
echo "============================================"
echo "Done — downloaded: $N_OK  skipped: $N_SKIP  failed: $N_FAIL"
echo "Videos in $OUT_DIR: $(ls "$OUT_DIR" | wc -l)"
echo "============================================"
echo ""
echo "Next: sbatch slurm_run.sh extract_all"
