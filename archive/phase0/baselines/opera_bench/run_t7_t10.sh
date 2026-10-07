#!/bin/bash
# Reproduce OPERA paper Tab. 4 T7 (ICBHI COPD) and T10 (KAUH obstructive) linear-probe AUROC for OPERA-CT / OPERA-CE
# with the unmodified OPERA benchmark code (a copy; only data paths are symlinked). Needs matplotlib and seaborn in .venv.
# Note: scripts/eval_all.sh of the OPERA repo calls `--task icbhi`, but linear_eval.py only knows `icbhidisease`.
set -euo pipefail
cd "$(dirname "$0")"
# Working copy of the OPERA benchmark (untracked, recreated here): code from ../../../repos/OPERA, data and checkpoints by symlink.
R=../../../repos/OPERA; D=../../../data
mkdir -p datasets/icbhi datasets/KAUH cks/model logs
[ -d src ] || cp -r $R/src src
cp -n $R/datasets/icbhi/*.txt datasets/icbhi/
ln -sfn "$(realpath $D/raw/icbhi/ICBHI_final_database)" datasets/icbhi/ICBHI_final_database
ln -sfn "$(realpath "$D/raw/kauh/Audio Files")" datasets/KAUH/AudioFiles
ln -sf "$(realpath $R/cks/model/encoder-operaCT.ckpt)" cks/model/encoder-operaCT.ckpt
ln -sf "$(realpath $D/models/opera/encoder-operaCE.ckpt)" cks/model/encoder-operaCE.ckpt
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 PYTHONPATH=.
PY=../../.venv/bin/python
for m in "operaCT 768" "operaCE 1280"; do
  set -- $m
  $PY -u src/benchmark/processing/kauh_processing.py --pretrain $1 --dim $2
  $PY src/benchmark/linear_eval.py --task kauh --pretrain $1 --dim $2 2>&1 | grep -E "Five times|^\[" > logs/t10_$1.log
  $PY -u src/benchmark/processing/icbhi_processing.py --pretrain $1 --dim $2
  $PY src/benchmark/linear_eval.py --task icbhidisease --pretrain $1 --dim $2 2>&1 | grep -E "Five times|^\[" > logs/t7_$1.log
done
