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
#   RUN_ID=<id> PARALLEL=3 bash scripts/run.sh resume   after a crash or reboot: runs the same jobs again, skipping finished units
#   bash scripts/run.sh reproduce [cell ...]   published protocol for comparison with the papers: official split, 5 seeds
#                                          (0-4), the epoch with the best test Score is kept (optimistic, as in the papers);
#                                          a cell is arm files joined by '+', e.g. baseline, baseline+patchmix,
#                                          baseline+freeze_encoder, sc_gain (default: baseline)
#
# PARALLEL=N runs N jobs at the same time on the GPU (scripts/sweep.py: a job starts only when MEM_GB of GPU memory is free,
# default 8). Without it the jobs run one after the other.
# Environment variables: PY (python of the training env, default .venv-train/bin/python; .venv-dass/bin/python for DASS), RUN_ID (default: timestamp),
# FOLDS (default 3), EXTRA_ARGS (appended to every training call).
cd "$(dirname "$0")/.." || exit 1
export RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-train/bin/python}
[[ "$PY" == *venv-dass* ]] && export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$HOME/miniconda3/envs/cuda128/lib:${LD_LIBRARY_PATH:-}  # DASS kernel runtime libraries (CUDA 13.0 or the conda 12.8 build)
CONF=experiments/train
FOLDS=${FOLDS:-3}
mkdir -p logs/$RUN_ID
JOBS=logs/$RUN_ID/jobs-${1:-all}.txt  # kept on disk: `resume` runs every jobs-*.txt of this RUN_ID again
[ "$1" = resume ] || : > "$JOBS"
SC_ARMS="sc sc_gain"
mode=$1; shift
echo "run: $RUN_ID  (continue or resume this campaign with RUN_ID=$RUN_ID)"

train() {  # train <cell> <extra args...>; cell parts joined by '+' map to experiments/train/<part>.yaml
  local cell=$1; shift
  local cfg=$CONF/base.yaml
  for part in ${cell//+/ }; do cfg=$cfg,$CONF/$part.yaml; done
  local cmd="$PY -m src.training.patchmix_cl.main --config $cfg --cell $cell $EXTRA_ARGS $*"
  if [ -n "$PARALLEL" ]; then echo "$cmd" >> "$JOBS"; else $cmd || echo "FAILED: $cell $*"; fi
}

run_queued() {  # in PARALLEL mode: run the collected jobs concurrently
  [ -n "$PARALLEL" ] && $PY scripts/sweep.py "$JOBS" --workers "$PARALLEL" --mem-gb "${MEM_GB:-8}" --ram-gb "${RAM_GB:-6}"
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
    done
    run_queued ;;
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
    done < $plan
    run_queued ;;
  reproduce)
    for cell in ${@:-baseline}; do
      for s in 0 1 2 3 4; do
        train "$cell" --selection test --seed "$s"
      done
    done
    run_queued ;;
  resume)  # run the jobs of every earlier parallel invocation of this RUN_ID again; finished units are skipped
    cat logs/$RUN_ID/jobs-*.txt | grep -v '^$' | awk '!seen[$0]++' > logs/$RUN_ID/jobs-resume.txt
    JOBS=logs/$RUN_ID/jobs-resume.txt; PARALLEL=${PARALLEL:-3}; run_queued ;;
  report) $PY -m src.training.patchmix_cl.summary final --run "$RUN_ID" ;;
  *) echo "usage: bash scripts/run.sh screen [arm ...] | summary | final | report | reproduce [cell ...] | resume  (details in the file header)"; exit 1 ;;
esac
