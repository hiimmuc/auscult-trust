#!/bin/bash
# Stage 1 of proposal v9, registered protocol (docs/prereg-v8.md section 3 + docs/prereg-v9-amendment.md).
# Every campaign lives in outputs/stage1/$RUN_ID and checkpoints/stage1/$RUN_ID; nothing is overwritten.
# Finished units are skipped and interrupted ones resume from last.pth when RUN_ID is the same.
#
#   bash scripts/run_stage1.sh screen [arm ...]   3-fold patient-grouped CV (fold k uses seed k), test not evaluated;
#                                                 SC arms run with both sc_mode values (dynamic, +sc_static)
#   bash scripts/run_stage1.sh summary            ranking, CV epochs, AuscultTrust -> outputs/stage1/$RUN_ID/stage1_screen.json
#   bash scripts/run_stage1.sh final              refit every arm at its CV epoch on all training patients;
#                                                 seeds 0-9 for P0 and AuscultTrust, 0-4 for the others
#   bash scripts/run_stage1.sh report             final table + gate G -> outputs/stage1/$RUN_ID/stage1_final.md
#
# Environment: PY (training python, timm==0.4.5; default .venv-train/bin/python), RUN_ID, FOLDS (3), EXTRA_ARGS.
cd "$(dirname "$0")/.." || exit 1
export RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-train/bin/python}
CONF=experiments/stage1
FOLDS=${FOLDS:-3}
SC_ARMS="P1_a1_input P1P3_sc_gain"
mode=$1; shift
echo "run: $RUN_ID  (resume or continue this campaign with RUN_ID=$RUN_ID)"

train() {  # train <cell> <extra main.py args...>; cell parts joined by '+' map to experiments/stage1/<part>.yaml
  local cell=$1; shift
  local cfg=$CONF/base.yaml
  for part in ${cell//+/ }; do cfg=$cfg,$CONF/$part.yaml; done
  $PY -m src.training.patchmix_cl.main --config "$cfg" --cell "$cell" $EXTRA_ARGS "$@" || echo "FAILED: $cell $*"
}

case $mode in
  screen)
    ARMS=${@:-"P0_baseline P1_a1_input P1P3_sc_gain P4_freq_mixstyle"}
    for arm in $ARMS; do
      cells=$arm
      [[ " $SC_ARMS " == *" $arm "* ]] && cells="$arm $arm+sc_static"
      for cell in $cells; do
        for k in $(seq 0 $((FOLDS - 1))); do
          train "$cell" --cv_folds "$FOLDS" --cv_fold "$k" --seed "$k"
        done
      done
    done ;;
  summary) $PY -m src.training.patchmix_cl.summary screen --run "$RUN_ID" --folds "$FOLDS" ;;
  final)
    plan=outputs/stage1/$RUN_ID/stage1_final_plan.txt  # one line per arm: <cell> <report epoch> <seed list>
    $PY - "$RUN_ID" > $plan <<'PYEOF' || exit 1
import json, sys
s = json.load(open('outputs/stage1/{}/stage1_screen.json'.format(sys.argv[1])))
for arm, r in s['arms'].items():
    ten = arm == 'P0_baseline' or arm == s['auscult_trust']
    print(r['cell'], r['epoch'], ','.join(str(i) for i in range(10 if ten else 5)))
PYEOF
    cat $plan
    while read cell epoch seeds; do
      for s in ${seeds//,/ }; do
        train "$cell" --selection fixed --report_epochs "$epoch" --seed "$s"
      done
    done < $plan ;;
  report) $PY -m src.training.patchmix_cl.summary final --run "$RUN_ID" ;;
  *) echo "usage: $0 screen [arm ...] | summary | final | report"; exit 1 ;;
esac
