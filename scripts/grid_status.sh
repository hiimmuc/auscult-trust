#!/bin/bash
# Status of the running sweeps on every machine (read-only): per job list units done / total and scheduler alive, units
# training now, failures, out-of-memory requeues, NaN or collapsed job logs, GPU, disk; plus the share dispatchers on A.
#   bash scripts/grid_status.sh
# A list `<name>` whose `<name>2` exists was re-split by scripts/lodo_move.py: its scheduler was stopped on purpose.
cd "$(dirname "$0")/.." || exit 1
REMOTE=Workspace/Personal/KHKT-CVA-2026/auscult-trust
# machine | ssh host (empty = this machine) | job lists (logs/campaigns/<list>.txt; missing ones are skipped)
MACHINES="A||lodo-v1-A lodo-v1-A2
B|namdp15@10.148.183.8|lodo-v1-B lodo-v1-B2
C|namdp15@10.148.183.119|lodo-v1-C lodo-v1-C2
D|namdp15@10.148.183.171|pcmcl-v1-D lodo-v1-D2
E|namdp15@10.148.182.179|lodo-v1-E2"

status() {  # $1 machine, $2 lists; prints a script to run in the repo root of that machine
  echo "M=$1; LISTS=\"$2\""
  cat <<'EOF'
echo "== machine $M   GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)   disk free: $(df -h . | tail -1 | awk '{print $4}')"
echo "training now: $(ps -eo args | grep 'src.training.patchmix_cl.main' | grep -v grep | grep -v '^/bin/sh' | sed -E 's/.*--cell (\S+) .*--seed ([0-9]+).*/\1:\2/' | sort -u | tr '\n' ' ')"
for N in $LISTS; do
  L=logs/campaigns/$N.txt
  [ -f $L ] || continue
  run=${N%-*}; total=$(grep -c . $L); done=0
  while read -r cell seed; do [ -f outputs/train/$run/$cell/seed$seed/report.json ] && done=$((done+1)); done < <(sed -E 's/.*--cell (\S+) .*--seed ([0-9]+).*/\1 \2/' $L)
  if [ -f logs/campaigns/${N}2.txt ]; then echo "-- $N: done $done / $total; superseded by ${N}2 (scheduler stopped on purpose)"; continue; fi
  if [ ! -f logs/$N.log ]; then echo "-- $N: done $done / $total; scheduler not started yet (the dispatcher starts it when this machine's old jobs end)"; continue; fi
  sweep=$(ps -eo pid,args | grep "scripts/sweep.py $L " | grep -v grep | awk '{print $1}' | head -1)
  echo "-- $N: done $done / $total, scheduler ${sweep:-NOT RUNNING}; last line: $(tail -1 logs/$N.log | cut -c1-110)"
  echo "   FAILED: $(grep -c FAILED logs/$N.log)   out-of-memory requeues: $(grep -c 'out of memory' logs/$N.log)"
  grep FAILED logs/$N.log 2>/dev/null | tail -3
  nan=$(grep -l -E "loss nan|Score nan" logs/$N/*.log 2>/dev/null | tr '\n' ' '); echo "   job logs with NaN loss or Score: ${nan:-none}"
  col=$(for f in logs/$N/*.log; do [ -f "$f" ] || continue; [ $(grep -c '^ \* test' $f) -ge 3 ] && grep '^ \* test' $f | tail -1 | grep -q ' Se 0\.' && echo -n "$f "; done)
  echo "   job logs collapsed (test Se below 1% from epoch 3): ${col:-none}"
done
EOF
}

while IFS='|' read -r m host lists; do
  if [ -z "$host" ]; then status "$m" "$lists" | bash
  else status "$m" "$lists" | timeout 120 ssh -o BatchMode=yes "$host" "cd $REMOTE && bash -s" || echo "== machine $m: ssh failed"; fi
done <<< "$MACHINES"
echo "== share dispatchers on A (scripts/lodo_move.py): $(ps -eo pid,args | grep '[l]odo_move.py' | sed -E 's/^ *([0-9]+).*--to ([A-E ]+).*/\1 (to \2)/' | tr '\n' ';')"
for f in logs/lodo-v1-share-*.log; do l=$(grep . $f | tail -1 | cut -c1-150); echo "   $f: ${l:-no lines yet (waiting)}"; done
