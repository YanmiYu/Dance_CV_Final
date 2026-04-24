#!/usr/bin/env bash
# Push data/raw_backgrounds/ from this laptop to Oscar.
#
# Companion to scripts/push_to_oscar.sh: that one handles the raw videos +
# AIST++ keypoints, this one handles the external background-texture library
# used by the detector's noisy-background augmentation
# (see configs/data/detector_train.yaml, docs/project_decisions.md section 6).
#
# Sidesteps three things the naive `rsync -av --mkpath ...` hits on macOS:
#   1) Remote parent dir does not exist; Apple's rsync 2.6.9 lacks --mkpath.
#   2) Every SSH to Oscar needs Duo 2FA -> we open one ControlMaster socket
#      and reuse it, so it's ONE Duo for the whole upload.
#   3) ~36k tiny JPEGs over SSH is slow and flaky; --partial + -W resumes a
#      dropped transfer without re-hashing already-uploaded files.
#
# Overrides (all optional):
#   OSCAR_USER         (default: mwang264)
#   OSCAR_HOST         (default: ssh.ccv.brown.edu)
#   OSCAR_REMOTE_ROOT  (default: ~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice)
#
# Usage:
#   bash scripts/push_backgrounds_to_oscar.sh
set -euo pipefail

OSCAR_USER="${OSCAR_USER:-mwang264}"
OSCAR_HOST="${OSCAR_HOST:-ssh.ccv.brown.edu}"
OSCAR_REMOTE_ROOT="${OSCAR_REMOTE_ROOT:-~/scratch/projects/CV_Tool_for_Dance_Choreography_Practice}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BG_SRC="data/raw_backgrounds"
if [[ ! -d "$BG_SRC" ]]; then
    echo "ERROR: local source directory missing: $BG_SRC" >&2
    echo "       Place the Places365-val (or equivalent) images under $BG_SRC" >&2
    echo "       before running this script. See configs/data/detector_train.yaml." >&2
    exit 1
fi

# One shared ControlMaster socket => one Duo prompt for the whole script.
# macOS caps Unix-socket paths at 104 bytes, so we use ~/.ssh with the short
# %C hash (16 chars) instead of a long /var/folders tmpdir.
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
CTRL_PATH="$HOME/.ssh/cm-%C"
SSH_OPTS=(
    -o "ControlMaster=auto"
    -o "ControlPath=$CTRL_PATH"
    -o "ControlPersist=60m"
    -o "ServerAliveInterval=30"
    -o "ServerAliveCountMax=6"
)

cleanup() {
    ssh "${SSH_OPTS[@]}" -O exit "$OSCAR_USER@$OSCAR_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "==> Opening SSH master connection to $OSCAR_USER@$OSCAR_HOST"
echo "    (expect Brown login banner + ONE Duo prompt; later steps reuse this socket)"
ssh "${SSH_OPTS[@]}" "$OSCAR_USER@$OSCAR_HOST" \
    "echo 'connected as '\$(whoami)' on '\$(hostname)"

REMOTE_BG="$OSCAR_REMOTE_ROOT/data/raw_backgrounds"

echo "==> Creating remote directory on Oscar"
ssh "${SSH_OPTS[@]}" "$OSCAR_USER@$OSCAR_HOST" \
    "mkdir -p $REMOTE_BG && echo 'remote dir ready'"

# -a   archive (recursive + preserve perms/times/links)
# -v   verbose
# -h   human-readable sizes
# -W   whole-file (skip delta-xfer; pointless on tiny immutable JPEGs)
# --info=progress2  single aggregate progress line instead of per-file spam
# --partial         keep partials so a re-run resumes a dropped transfer
# --stats           summary at the end
# -e "ssh ..."      route rsync's ssh through the shared ControlMaster
RSYNC_SSH="ssh ${SSH_OPTS[*]}"
RSYNC_OPTS=(-avhW --info=progress2 --partial --stats -e "$RSYNC_SSH")

N_FILES="$(find "$BG_SRC" -type f | wc -l | tr -d ' ')"
echo "==> Transferring external backgrounds ($(du -sh "$BG_SRC" | awk '{print $1}'), ${N_FILES} files)"
rsync "${RSYNC_OPTS[@]}" "$BG_SRC/" \
    "$OSCAR_USER@$OSCAR_HOST:$REMOTE_BG/"

echo "==> Verifying remote file count"
ssh "${SSH_OPTS[@]}" "$OSCAR_USER@$OSCAR_HOST" bash -s <<EOF
set -e
echo "backgrounds: \$(find $REMOTE_BG -type f | wc -l) files, \$(du -sh $REMOTE_BG | awk '{print \$1}')"
EOF

echo "==> Done."
