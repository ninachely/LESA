#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LESA_STORAGE"
if [[ ! -x "$LESA_STORAGE/env/bin/python" ]]; then
    command -v conda >/dev/null || {
        echo 'conda is required to create the shared Python 3.11 environment.' >&2
        exit 1
    }
    conda create -y -p "$LESA_STORAGE/env" python=3.11 pip libstdcxx-ng
fi
PY="$LESA_STORAGE/env/bin/python"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -r "$LESA_REPO/scripts/mlspace/requirements.txt"
"$PY" -m pip install --no-deps invisible-watermark==0.2.0
"$PY" -c 'import torch, transformers, cv2, imwatermark; from flux.sampling import denoise_cache; print("Imports OK:", torch.__version__, transformers.__version__)'
echo "Environment ready: $LESA_STORAGE/env"
