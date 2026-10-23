#!/usr/bin/env bash
# Fetch the weights from the private repo (when configured), then serve. Without them the API reports `unavailable`.
set -euo pipefail
if [ -n "${AT_WEIGHTS_REPO:-}" ]; then
  python - <<'PY' || echo "weights download failed; serving without a model"
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["AT_WEIGHTS_REPO"], local_dir="/app", token=os.environ.get("HF_TOKEN"))
PY
fi
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
