#!/bin/bash
# R3: Patch-Mix CL recipe flags on the best fine-tuned rung of a finished ladder (same RUN_ID, so cells sit next to the baseline).
# Grouped CV over train+val patients picks the cell; the official test is evaluated once per cell after CV.
#   bash scripts/run_improve.sh ast [cell_base]     # cell_base default ft-k4-attn; flags +mix, +ema, +mix+ema are appended
# Resumable: re-run the same command after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
enc=${1:?usage: run_improve.sh ast|hear [cell_base]}; base=${2:-ft-k4-attn}
export RUN_ID=${RUN_ID:-ladder-$enc-v1} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python; C=configs/ladder_$enc.yaml; cells=$base+mix,$base+ema,$base+mix+ema
echo "[$(date +%T)] 1/3 CV of $cells";   $PY -m src.auscult_trust cv-ft $C --cells $cells
echo "[$(date +%T)] 2/3 CV table";       $PY -m src.auscult_trust select $C
echo "[$(date +%T)] 3/3 final test runs"; $PY -m src.auscult_trust final $C --cells $base,$cells
echo "[$(date +%T)] done. Results: outputs/ladder_$enc/$RUN_ID/"
