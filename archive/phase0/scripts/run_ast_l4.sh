#!/bin/bash
# R3 retry (option 1): layer-4 token cache -> CV of ft-k4/k8 attn with and without +mix -> CV table -> final test runs.
#   setsid nohup bash scripts/run_ast_l4.sh > logs/ast_l4.log 2>&1 < /dev/null &      (resumable, same command)
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_ID=${RUN_ID:-ladder-ast_l4-v1} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python; C=configs/ladder_ast_l4.yaml; cells=ft-k4-attn,ft-k8-attn,ft-k4-attn+mix,ft-k8-attn+mix
echo "[$(date +%T)] 1/4 cache features";  $PY -m src.features configs/extract_ast_l4.yaml
echo "[$(date +%T)] 2/4 CV of $cells";    $PY -m src.auscult_trust cv-ft $C --pool attn --cells $cells
echo "[$(date +%T)] 3/4 CV table";        $PY -m src.auscult_trust select $C
echo "[$(date +%T)] 4/4 final test runs"; $PY -m src.auscult_trust final $C --cells $cells
echo "[$(date +%T)] done. Results: outputs/ladder_ast_l4/$RUN_ID/"
