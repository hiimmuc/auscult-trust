#!/bin/bash
# Stage 1 of proposal v8, registered protocol (docs/prereg-v8.md section 3). Resumable: finished runs are skipped,
# interrupted runs resume from last.pth.
#
#   bash scripts/run_stage1.sh screen [variant ...]   3-fold patient-grouped CV, fold k uses seed k, test not evaluated
#   bash scripts/run_stage1.sh p9                     screen the combination of the two best single variants
#   bash scripts/run_stage1.sh final                  refit finalists on all training patients, 5 seeds, CV epoch fixed
#
# After `screen` (and `p9`): python stage1_summary.py screen  -> save/stage1_screen.json (epochs, finalists)
# After `final`:             python stage1_summary.py final   -> save/stage1_final.md (primary + literature column, G-base)
# P8 is a selection rule: it reuses the P0 runs, so it is never trained on its own.
cd "$(dirname "$0")/../baselines/patchmix_cl" || exit 1
CONF=../../configs/stage1_variants
FOLDS=${FOLDS:-3}
SEEDS=${SEEDS:-"0 1 2 3 4"}
PY=${PY:-.venv/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mode=$1; shift

run() {  # run <config list> <tag> <extra args...>
  local cfg=$1 tag=$2; shift 2
  local dir=save/icbhi_ast_patchmix_cl_${tag}
  [ -f $dir/stage1_report.json ] && { echo "done: $tag"; return; }
  $PY main.py --config $cfg --tag $tag --resume $dir/last.pth "$@" || echo "FAILED: $tag"
}

case $mode in
  screen)
    VARIANTS=${@:-$(cd $CONF && ls *.json | sed 's/\.json$//' | grep -v '^P8_')}
    for v in $VARIANTS; do
      for k in $(seq 0 $((FOLDS - 1))); do
        run $CONF/$v.json ${v}_cv${k} --cv_folds $FOLDS --cv_fold $k --seed $k
      done
    done ;;
  p9)
    pair=$($PY -c "import json; print(' '.join(json.load(open('save/stage1_screen.json'))['p9_candidates']))")
    set -- $pair
    [ $# -eq 2 ] || { echo "need two screened single variants in save/stage1_screen.json"; exit 1; }
    for k in $(seq 0 $((FOLDS - 1))); do
      run $CONF/$1.json,$CONF/$2.json ${1}+${2}_cv${k} --cv_folds $FOLDS --cv_fold $k --seed $k
    done ;;
  final)
    # one line per finalist: <variant> <config list> <report epochs>; P0 also carries the P8 epoch
    $PY - "$CONF" > save/stage1_final_plan.txt <<'EOF' || exit 1
import json, sys
conf = sys.argv[1]
s = json.load(open('save/stage1_screen.json'))
fin, ep = s['finalists'], s['epochs']
if 'P0_baseline' not in fin:
    fin.append('P0_baseline')
for v in fin:
    if v == 'P8_worst_device':
        continue  # read from the P0 runs
    cfg = ','.join('{}/{}.json'.format(conf, x) for x in v.split('+'))
    eps = [ep[v]] + ([ep['P8_worst_device']] if v == 'P0_baseline' and 'P8_worst_device' in ep else [])
    print(v, cfg, ','.join(str(e) for e in dict.fromkeys(eps)))
EOF
    cat save/stage1_final_plan.txt
    while read v cfg eps; do
      for s in $SEEDS; do
        run $cfg ${v}_fix_seed${s} --selection fixed --report_epochs $eps --seed $s
      done
    done < save/stage1_final_plan.txt ;;
  *) echo "usage: $0 screen|p9|final [variant ...]"; exit 1 ;;
esac
