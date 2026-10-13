#!/bin/bash
# Job lists of the campaigns after the 2026-10-09 report (docs/prereg-v9-amendment-2.md). Writes logs/campaigns/<name>.txt,
# one training command per line with its own RUN_ID, for scripts/sweep.py. Nothing is started here.
#   bash scripts/campaigns.sh c2_ast c2_dass c2_htsat c2f c3 c4_screen c4_final c5 c6 smoke
# c4_final needs the screen summary of screen-ast-cv5 (CV epoch of sc_cons) and the bound chosen in screen-ast-c0.
cd "$(dirname "$0")/.." || exit 1
C=experiments/train; DEVS="meditron littc2se litt3200 akgc417l"
PY=.venv-train/bin/python
DASS_PY="LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:\$HOME/miniconda3/envs/cuda128/lib .venv-dass/bin/python"
mkdir -p logs/campaigns
cfg() { local c=$C/base.yaml; for p in ${1//+/ }; do c=$c,$C/$p.yaml; done; echo $c; }
line() { echo "RUN_ID=$1 ${3:-$PY} -m src.training.patchmix_cl.main --config $(cfg $2) --cell ${4:-$2} ${@:5}"; }
gain_cell() { $PY -c "import json; print(json.load(open('outputs/train/screen-ast-c0/screen.json'))['arms']['sc_gain']['cell'].replace('sc_gain', 'sc_gain_v2'))"; }
lodo50() {  # lodo50 <run> <cell prefix> <config parts> [python]
  for d in $DEVS; do for s in 0 1 2 3 4; do
    line $1 "$3+lodo_$d" "${4:-$PY}" "$2+lodo_$d" --selection fixed --report_epochs 50 --seed $s
  done; done
}
for name in "$@"; do
  out=logs/campaigns/$name.txt
  case $name in
    c2_ast) lodo50 lodo-bench ast_baseline baseline ;;
    c2_dass) lodo50 lodo-bench dass_baseline encoder_dass+baseline "$DASS_PY" ;;
    c2_htsat) lodo50 lodo-bench htsat_baseline encoder_htsat+baseline ;;
    c2f) lodo50 lodo-bench ast_frozen baseline+freeze_encoder; lodo50 lodo-bench opera_ct_frozen encoder_opera_ct+baseline+freeze_encoder ;;
    c3) for s in 0 1 2 3 4; do line beats encoder_beats+beats_anchor "$PY" beats_anchor --selection test --seed $s; done
        for s in 0 1 2 3 4; do line beats encoder_beats+baseline "$PY" beats_patchmix_cl --selection test --seed $s; done
        lodo50 lodo-bench beats_baseline encoder_beats+baseline ;;
    c4_screen) g=$(gain_cell)
        for k in 0 1 2 3 4; do
          line screen-ast-cv5 sc_cons "$PY" sc_cons --cv_folds 5 --cv_fold $k --seed $k --stop_epoch 20
          line screen-ast-cv5 $g "$PY" $g --cv_folds 5 --cv_fold $k --seed $k --stop_epoch 20
        done ;;
    c4_final) ep=$($PY -c "import json; print(json.load(open('outputs/train/screen-ast-cv5/screen.json'))['arms']['sc_cons']['epoch'])")
        for s in 0 1 2 3 4; do line screen-ast sc_cons "$PY" sc_cons --selection fixed --report_epochs $ep --stop_epoch $ep --seed $s; done
        for d in $DEVS; do for s in 0 1 2; do
          line lodo-ast "sc_cons+lodo_$d" "$PY" "sc_cons+lodo_$d" --selection fixed --report_epochs $ep --seed $s
        done; done ;;
    c5) for lr in "" "+lr1e-4"; do for arm in baseline sc+sc_static; do for k in 0 1 2 3 4; do
          line bs32 "$arm+bs32$lr" "$PY" "$arm+bs32$lr" --cv_folds 5 --cv_fold $k --seed $k --stop_epoch 20
        done; done; done ;;
    c6) for s in 0 1 2 3 4; do line pafa pafa_ast "$PY" pafa_ast --selection test --seed $s; done
        lodo50 lodo-bench pafa_ast pafa_ast ;;
    smoke) line smoke-b sc_cons "$PY" sc_cons --cv_folds 5 --cv_fold 0 --seed 0 --stop_epoch 1
        line smoke-b encoder_beats+baseline "$PY" beats_patchmix_cl --cv_folds 5 --cv_fold 0 --seed 0 --stop_epoch 1
        line smoke-b encoder_beats+beats_anchor "$PY" beats_anchor --cv_folds 5 --cv_fold 0 --seed 0 --stop_epoch 1
        line smoke-b pafa_ast "$PY" pafa_ast --cv_folds 5 --cv_fold 0 --seed 0 --stop_epoch 1
        line smoke-b sc+sc_static+bs32 "$PY" sc_bs32 --cv_folds 5 --cv_fold 0 --seed 0 --stop_epoch 1 ;;
    *) echo "unknown campaign $name"; exit 1 ;;
  esac > $out
  echo "$out: $(wc -l < $out) jobs"
done
