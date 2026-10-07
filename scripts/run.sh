#!/bin/bash
# Train and compare the lung-sound classifier arms (see experiments/train/README.md for the arms).
# Every campaign is one RUN_ID: results go to outputs/train/<RUN_ID>/ and weights to checkpoints/train/<RUN_ID>/.
# Nothing is overwritten. Finished runs are skipped and interrupted ones resume (last.pth) when RUN_ID is the same.
#
#   bash scripts/run.sh screen [arm ...]   cross-validate every arm: 3 folds, each fold trained on part of the training
#                                          patients and scored on the rest (test set untouched); spectrum-correction
#                                          arms run with both coefficient bounds (dynamic, +sc_static)
#   bash scripts/run.sh summary            rank the arms, pick the epoch and coefficient bound of each, pick the
#                                          AuscultTrust arm -> outputs/train/RUN_ID/screen.json
#   bash scripts/run.sh final              retrain every arm on all training patients at its chosen epoch and score on
#                                          the test set: seeds 0-9 for baseline and AuscultTrust, 0-4 for the others
#   bash scripts/run.sh report             final table and non-inferiority check -> outputs/train/RUN_ID/final.md
#
# Environment variables: PY (python of the training env, default .venv-train/bin/python), RUN_ID (default: timestamp),
# FOLDS (default 3), EXTRA_ARGS (appended to every training call).
cd "$(dirname "$0")/.." || exit 1
export RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-train/bin/python}
CONF=experiments/train
FOLDS=${FOLDS:-3}
SC_ARMS="sc sc_gain"
mode=$1; shift
echo "run: $RUN_ID  (continue or resume this campaign with RUN_ID=$RUN_ID)"

train() {  # train <cell> <extra args...>; cell parts joined by '+' map to experiments/train/<part>.yaml
  local cell=$1; shift
  local cfg=$CONF/base.yaml
  for part in ${cell//+/ }; do cfg=$cfg,$CONF/$part.yaml; done
  $PY -m src.training.patchmix_cl.main --config "$cfg" --cell "$cell" $EXTRA_ARGS "$@" || echo "FAILED: $cell $*"
}

case $mode in
  screen)
    ARMS=${@:-"baseline sc sc_gain freq_mixstyle"}
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
    plan=outputs/train/$RUN_ID/final_plan.txt  # one line per arm: <cell> <epoch> <seed list>
    $PY - "$RUN_ID" > $plan <<'PYEOF' || exit 1
import json, sys
s = json.load(open('outputs/train/{}/screen.json'.format(sys.argv[1])))
for arm, r in s['arms'].items():
    ten = arm == 'baseline' or arm == s['auscult_trust']
    print(r['cell'], r['epoch'], ','.join(str(i) for i in range(10 if ten else 5)))
PYEOF
    cat $plan
    while read cell epoch seeds; do
      for s in ${seeds//,/ }; do
        train "$cell" --selection fixed --report_epochs "$epoch" --seed "$s"
      done
    done < $plan ;;
  report) $PY -m src.training.patchmix_cl.summary final --run "$RUN_ID" ;;
  *) echo "usage: bash scripts/run.sh screen [arm ...] | summary | final | report  (details in the file header)"; exit 1 ;;
esac
