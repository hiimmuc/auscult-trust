#!/bin/bash
# Option 1 -> 2 -> 3, resumable (every stage skips finished work; re-run the same command after an interruption):
#   1  scripts/run_ast_l4.sh (AST layer-4 token cache, lr_block 5e-5, 30 epochs, +mix)
#   2  Part II: KAUH E2 and ICBHI device-held-out E1, rungs L0/L1/L2, 5 encoders (src/shift_study.py, RUN_ID shift-v1)
#   3  Patch-Mix CL seeds 2-5 (baselines/patchmix_cl)
#   setsid nohup bash scripts/run_chain.sh > logs/chain.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/huggingface} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
echo "[$(date +%T)] 1/3 AST layer-4 retry"
while pgrep -f "^bash scripts/run_ast_l4.sh" > /dev/null; do sleep 30; done   # already running from an earlier launch
bash scripts/run_ast_l4.sh >> logs/ast_l4.log 2>&1 || echo "option 1 FAILED"
echo "[$(date +%T)] 2/3 Part II"
for enc in ast hear clap opera_ct opera_ce; do
  for e in e2 e1; do
    echo "[$(date +%T)] shift_study $e $enc"
    RUN_ID=shift-v1 .venv/bin/python -m src.shift_study $e $enc --seeds 5 >> logs/shift_${enc}_$e.log 2>&1 || echo "shift_study $e $enc FAILED"
  done
done
RUN_ID=shift-v1 .venv/bin/python -m src.shift_study table > outputs/shift_table_v1.txt 2>&1
echo "[$(date +%T)] 3/3 Patch-Mix CL seeds 2-5"
cd baselines/patchmix_cl
for s in 2 3 4 5; do ./run_until_done.sh $s; done
echo "[$(date +%T)] chain done"
