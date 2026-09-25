#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
test -f "$LESA_STORAGE/model_paths.json" || {
    echo 'Download models on the CPU server first; see scripts/mlspace/README_RU.md.' >&2
    exit 1
}
nvidia-smi
PY="$LESA_STORAGE/env/bin/python"
test -x "$PY" || { echo 'Run setup.sh on the CPU server first.' >&2; exit 1; }
export LD_LIBRARY_PATH="$LESA_STORAGE/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -f "$LESA_STORAGE/env/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$LESA_STORAGE/env/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PY" "$LESA_REPO/scripts/mlspace/smoke_flux.py" "$@"
