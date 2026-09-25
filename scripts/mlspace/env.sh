#!/usr/bin/env bash
# Source from either the CPU notebook terminal or a GPU job.
LESA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LESA_REPO
export LESA_STORAGE="${LESA_STORAGE:-$LESA_REPO/.runtime}"
export HF_HOME="$LESA_STORAGE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE
export PIP_CACHE_DIR="$LESA_STORAGE/pip-cache"
export PYTHONPATH="$LESA_REPO/LESA_FLUX.1-dev_FLUX.1-schnell:$LESA_REPO/LESA_FLUX.1-dev_FLUX.1-schnell/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
