#!/bin/bash
# Resumable ladder pipeline for one encoder: cache features -> L0 probe -> L1 head CV -> (ViT only) L2/L3 fine-tuning CV -> test.
# Every stage skips finished work, so after an interruption (reboot, OOM, Ctrl-C) just run the same command again.
#   bash scripts/run_ladder.sh opera_ct|opera_ce|clap|ast|hear [frozen|all]
# Run it detached and watch the log:
#   setsid nohup bash scripts/run_ladder.sh ast > logs/ladder_ast.log 2>&1 < /dev/null &   ;   tail -f logs/ladder_ast.log
# `all` (default) also runs the fine-tuned rungs L2/L3 for ast and hear; the other encoders have no token cache, so they
# stop after L1. `frozen` stops after L1 for every encoder; re-run with `all` and the same RUN_ID to add L2/L3.
set -euo pipefail
cd "$(dirname "$0")/.."
enc=${1:?usage: run_ladder.sh opera_ct|opera_ce|clap|ast|hear [frozen|all]}; stage=${2:-all}
export RUN_ID=${RUN_ID:-ladder-$enc-v1}      # fixed id = fixed output folder = resumable
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=.venv/bin/python; C=configs/ladder_$enc.yaml
echo "[$(date +%T)] 1/6 cache features (resumable)";   $PY -m src.features configs/extract_$enc.yaml
echo "[$(date +%T)] 2/6 L0 probe";                     $PY -m src.auscult_trust probe $C
echo "[$(date +%T)] 3/6 L1 head variants, grouped CV"; $PY -m src.auscult_trust cv-head $C
case $enc/$stage in
  ast/all|hear/all) echo "[$(date +%T)] 4/6 L2/L3 fine-tuning, grouped CV"; $PY -m src.auscult_trust cv-ft $C ;;
  *) echo "[$(date +%T)] 4/6 skipped: frozen rungs only" ;;
esac
echo "[$(date +%T)] 5/6 CV table";                     $PY -m src.auscult_trust select $C
echo "[$(date +%T)] 6/6 final test runs (baseline, best head, best fine-tuned)"; $PY -m src.auscult_trust final $C --cells report
echo "[$(date +%T)] done. Results: outputs/ladder_$enc/$RUN_ID/  (cv_summary.json, final-*/seed*.json, summary.txt)"
