#!/bin/bash
# Run campaigns one after the other on this machine (each a scripts/sweep.py over logs/campaigns/<name>.txt).
#   nohup bash scripts/queue.sh wait:<pattern> smoke c4_screen c4_gate c4_final c2_ast c2_dass c3 c5 c6 > logs/queue.log 2>&1 &
# wait:<pattern>   wait until no process matches the pattern (e.g. an earlier scheduler); waitpid:<pid> waits for a process
# smoke[:<list>]   1-epoch run of every cell of the job list (default smoke); logs memory, loss_cons and agreement; a failed
#                  cell is listed in logs/campaigns/smoke_failed.txt and its campaigns are skipped (only `smoke` clears the list)
# c4_gate          5-fold screen summary of screen-ast-cv5 and the registered stop rule of sc_cons; c4_final runs only if it passes
# Finished units are skipped (same RUN_ID), so the queue can be restarted at any point.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv-train/bin/python
WORKERS=${WORKERS:-4}
declare -A NEEDS=([c7_screen]=sc_noise [c7_final]=sc_noise [c4_screen]=sc_cons [c4_final]=sc_cons [c3]="beats_patchmix_cl beats_anchor" [c6]=pafa_ast [c5]=sc_bs32 [c8]="lung_srad lung_srad+sc+sc_static")
sweep() {  # sweep <name> <workers> <mem-gb>
  bash scripts/campaigns.sh $1 || return 1
  for round in 1 2 3; do  # units lost to a full GPU after all retries run again; finished units are skipped
    SWEEP_TAG=-$1 RUN_ID=queue-$1 $PY scripts/sweep.py logs/campaigns/$1.txt --workers $2 --mem-gb $3 --ram-gb 8 > logs/campaigns/$1.sweep.log 2>&1; tail -n 1 logs/campaigns/$1.sweep.log; tail -n 1 logs/campaigns/$1.sweep.log | grep -q "0 failed: \[\]" && return 0
  done
  return 1
}
for stage in "$@"; do
  echo "=== $(date '+%F %T') $stage"
  for c in ${NEEDS[$stage]}; do
    if grep -qx "$c" logs/campaigns/smoke_failed.txt 2>/dev/null; then echo "skip $stage: smoke failed for $c"; continue 2; fi
  done
  case $stage in
    waitpid:*) while kill -0 "${stage#waitpid:}" 2>/dev/null; do sleep 120; done ;;
    wait:*) while pgrep -af "${stage#wait:}" | grep -v "queue.sh" | grep -q .; do sleep 120; done ;;  # other queues name the pattern too
    smoke|smoke:*)
      list=smoke; [[ $stage == smoke:* ]] && list=${stage#smoke:}
      bash scripts/campaigns.sh $list; [ $list = smoke ] && : > logs/campaigns/smoke_failed.txt
      while read -r cmd; do
        cell=$(echo "$cmd" | sed -E 's/.*--cell ([^ ]+).*/\1/')
        log=logs/campaigns/smoke_$cell.log
        for try in $(seq 30); do  # a GPU filled by other jobs is not a failure of the cell: wait and try again
          ( eval "$cmd --num_workers 4" > $log 2>&1 ) &
          pid=$!; peak=0
          while kill -0 $pid 2>/dev/null; do
            m=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p=$(pgrep -P $pid -f patchmix_cl.main | head -1) '$1==p+0 {print $2}' | tr -d ' ,')
            [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
            sleep 5
          done
          wait $pid; rc=$?
          grep -q "out of memory" $log || break
          sleep 300
        done
        echo "smoke $cell: exit $rc, peak GPU memory ${peak} MiB, $(grep -o 'loss_cons [0-9.]* ([0-9.]*)[^w]*agree [0-9.]*' $log | head -1) | $(grep -o 'val Score [0-9.]*' $log | tail -1)"
        if [ $rc -ne 0 ] || ! grep -q "val Score [0-9]" $log; then echo $cell >> logs/campaigns/smoke_failed.txt; fi
        if [ "$cell" = sc_cons ] && ! grep -q "loss_cons" $log; then echo sc_cons >> logs/campaigns/smoke_failed.txt; fi
      done < logs/campaigns/$list.txt ;;
    kauh:*)  # kauh:<condition>:<cell>:<epoch|auto>[:sc|:scown|:none][:a2|:ttaK]  frozen refit seeds 0-4 of screen-ast on KAUH windows
      IFS=: read -r _ cond cell ep mode more <<< "$stage"
      SC_UNIT=outputs/train/screen-ast/sc+sc_static/cv0
      for s in 0 1 2 3 4; do
        u=outputs/train/screen-ast/$cell/seed$s; o=outputs/kauh/final/$cond/seed$s.npz
        [ -f $o ] && continue
        [ "$ep" = auto ] && ep=$(ls checkpoints/train/screen-ast/$cell/seed$s/report_epoch_*.pth 2>/dev/null | head -1 | sed -E 's/.*report_epoch_([0-9]+).pth/\1/')  # fixed-selection refit: one epoch
        [ -f checkpoints/train/screen-ast/$cell/seed$s/report_epoch_$ep.pth ] || { echo "no checkpoint $cell seed$s"; continue; }
        case $mode in
          sc) extra="--sc --sc_state $SC_UNIT/sc_state.npz --sc_args $SC_UNIT/train_args.json" ;;
          scown) extra="--sc" ;;
          *) extra="" ;;
        esac
        case $more in a2) extra="$extra --a2" ;; tta*) extra="$extra --tta_eq ${more#tta}" ;; esac
        $PY -m src.evaluation.kauh_infer $u --ckpt report_epoch_$ep.pth --out $o $extra
      done ;;
    export:*)  # export:<cell>:<epoch>[:source]  softmax export of refit seeds 0-4 (source: inductive SC test coefficients)
      IFS=: read -r _ cell ep prof <<< "$stage"
      for s in 0 1 2 3 4; do
        u=outputs/train/screen-ast/$cell/seed$s
        [ -f $u/export${prof:+_$prof}/model.sha256 ] && continue
        $PY -m src.training.patchmix_cl.export_probs $u --ckpt report_epoch_$ep.pth ${prof:+--sc_test_profile $prof}
      done ;;
    screen:*)  # screen:<condition>:<cell>:<epoch>[:sc]  ICBHI official-test windows for the screening view
      IFS=: read -r _ cond cell ep mode <<< "$stage"
      for s in 0 1 2 3 4; do
        o=outputs/screening/$cond/seed$s.npz; [ -f $o ] && continue
        [ -f checkpoints/train/screen-ast/$cell/seed$s/report_epoch_$ep.pth ] || { echo "no checkpoint $cell seed$s"; continue; }
        $PY -m src.evaluation.screening infer outputs/train/screen-ast/$cell/seed$s --ckpt report_epoch_$ep.pth --out $o ${mode:+--sc}
      done ;;
    snr:*)  # snr:<arm>:<cell>[:test]  noise stress on refit seeds 0-4 (official test); test: ESC-50 folds 4-5 only
      IFS=: read -r _ arm cell folds <<< "$stage"
      $PY -m src.evaluation.snr_stress infer outputs/train/screen-ast/$cell/seed{0,1,2,3,4} \
        ${folds:+--test_folds} --out outputs/snr${folds:+_test_folds}/$arm ;;
    c7_gate)  # CV epoch of sc_noise from the 5-fold screen
      $PY -m src.training.patchmix_cl.summary screen --run screen-ast-cv5 --folds 5 > logs/campaigns/c7_screen_summary.txt 2>&1
      cat logs/campaigns/c7_screen_summary.txt ;;
    c4_screen) sweep c4_screen $WORKERS 7 ;;
    c4_gate)
      [ -f outputs/train/screen-ast-cv5/screen_before_c4.json ] || cp outputs/train/screen-ast-cv5/screen.json outputs/train/screen-ast-cv5/screen_before_c4.json
      $PY -m src.training.patchmix_cl.summary screen --run screen-ast-cv5 --folds 5 > logs/campaigns/c4_screen_summary.txt 2>&1
      $PY scripts/c4_gate.py >> logs/campaigns/c4_screen_summary.txt 2>&1 || echo "STOP sc_cons: gate could not be computed (incomplete screen)" >> logs/campaigns/c4_screen_summary.txt
      cat logs/campaigns/c4_screen_summary.txt
      grep -q "^STOP" logs/campaigns/c4_screen_summary.txt && echo sc_cons >> logs/campaigns/smoke_failed.txt ;;
    c4_final) sweep c4_final $WORKERS 7 ;;
    c7_screen) sweep c7_screen $WORKERS 7 ;;
    c7_final) sweep c7_final $WORKERS 7 ;;
    c2_ast|c2_htsat|c2f|c3) sweep $stage $WORKERS 7 ;;
    c2_dass) sweep c2_dass $WORKERS 12 ;;  # a DASS unit peaks near 11 GB
    c5) sweep c5 1 28 ;;
    c8) sweep c8 2 16 ;;
    c6) sweep c6 2 14 ;;
    *) echo "unknown stage $stage" ;;
  esac
done
echo "=== $(date '+%F %T') queue done"
