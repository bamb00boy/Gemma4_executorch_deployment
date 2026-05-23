#!/usr/bin/env bash
# Activate the `gemma4` conda env and run a script with all caches
# pointed at <repo>/cache/ so nothing leaks into ~/.cache.
#
# Usage:
#   scripts/run.sh scripts/01_smoke_test.py
#   scripts/run.sh -m pip install foo
#   scripts/run.sh hf login

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="$REPO_ROOT/cache"
MODELS_DIR="$REPO_ROOT/models"

mkdir -p \
    "$CACHE_DIR/huggingface/hub" \
    "$CACHE_DIR/huggingface/datasets" \
    "$CACHE_DIR/huggingface/transformers" \
    "$CACHE_DIR/torch" \
    "$CACHE_DIR/xdg" \
    "$CACHE_DIR/triton" \
    "$CACHE_DIR/pip" \
    "$MODELS_DIR"

export HF_HOME="$CACHE_DIR/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$CACHE_DIR/torch"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export PIP_CACHE_DIR="$CACHE_DIR/pip"

# Source <repo>/.env if present so secrets (HF_TOKEN, etc.) load from a
# single file instead of disk-resident token caches. The file is gitignored.
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

# Activate the gemma4 conda env. Works with Miniforge or Anaconda.
if [[ -z "${CONDA_EXE:-}" ]]; then
    for cand in \
        "$HOME/miniforge3/bin/conda" \
        "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda"; do
        if [[ -x "$cand" ]]; then
            export CONDA_EXE="$cand"
            break
        fi
    done
fi
if [[ -z "${CONDA_EXE:-}" ]]; then
    echo "error: could not find conda. Install Miniforge or set CONDA_EXE." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$("$CONDA_EXE" info --base)/etc/profile.d/conda.sh"
conda activate gemma4

cd "$REPO_ROOT"

# If the first arg looks like a python script or a python flag (-c, -m,
# -u, etc.), run it through python; otherwise exec as-is so commands like
# `hf login` or `pip install ...` work.
if [[ $# -gt 0 && ( "${1##*.}" == "py" || "$1" == -* ) ]]; then
    exec python "$@"
else
    exec "$@"
fi
