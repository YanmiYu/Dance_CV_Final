#!/bin/bash
# Activate the `dance-cv` conda env on Oscar (Brown CCV).
#
# IMPORTANT: source this script, do NOT execute it, otherwise the activation
# will not survive in your current shell.
#
#     source scripts/slurm/activate_env.sh
#
# After sourcing, your prompt should start with `(dance-cv)` and `python`,
# `pip`, `torch` etc. will resolve to the env at
# /users/<you>/.conda/envs/dance-cv.
#
# If `module load anaconda/...` fails because CCV rotated the build-string,
# run `module avail anaconda` and update ANACONDA_MODULE below (or export it
# before sourcing). Keep this value in sync with scripts/slurm/setup_env.sh.

ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"
ENV_NAME="${ENV_NAME:-dance-cv}"

# Warn if the user ran `bash activate_env.sh` instead of `source ...`.
# $0 is the script path when executed, but "bash" / "-bash" when sourced.
case "${0##*/}" in
    activate_env.sh)
        echo "error: this script must be sourced, not executed." >&2
        echo "  run:  source scripts/slurm/activate_env.sh" >&2
        exit 1
        ;;
esac

module purge
module load "${ANACONDA_MODULE}"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "${ENV_NAME}"

echo "activated conda env: ${ENV_NAME}"
echo "python: $(command -v python)"
