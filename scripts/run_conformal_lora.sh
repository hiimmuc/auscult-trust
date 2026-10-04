#!/bin/bash
# AuscultTrust: conformal evaluation of the finished cells, then the LoRA rung (CV + test). Resumable: re-run to continue.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ID=${RUN_ID:-ladder-ast-v1} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python; C=configs/ladder_ast.yaml
echo "[$(date +%T)] 1/4 conformal, finished cells";  $PY -m src.auscult_trust final-conformal $C --cells last-mean,last-attn,ft-k2-attn
echo "[$(date +%T)] 2/4 LoRA CV";                    $PY -m src.auscult_trust cv-ft $C --pool attn --cells lora-r8-attn
echo "[$(date +%T)] 3/4 LoRA test + conformal";      $PY -m src.auscult_trust final $C --cells lora-r8-attn; $PY -m src.auscult_trust final-conformal $C --cells lora-r8-attn
echo "[$(date +%T)] 4/4 done"
