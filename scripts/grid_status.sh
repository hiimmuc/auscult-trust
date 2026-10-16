#!/bin/bash
# Status of the grid-v1 sweeps on both machines (read-only): sweep alive, units done / running, failures, NaN losses, GPU, disk.
#   bash scripts/grid_status.sh
cd "$(dirname "$0")/.." || exit 1
B=namdp15@10.148.183.8
status() {  # $1 = machine letter; runs in the repo root of that machine
  cat <<'EOF' | sed "s/@M@/$1/g"
M=@M@; L=logs/campaigns/grid-v1-$M.txt
total=$(grep -c . $L)
done=0; while read -r cell seed; do [ -f outputs/train/grid-v1/$cell/seed$seed/report.json ] && done=$((done+1)); done < <(sed -E 's/.*--cell (\S+) .*--seed ([0-9]+).*/\1 \2/' $L)
sweep=$(ps -eo pid,args | grep "scripts/sweep.py $L" | grep -v grep | awk '{print $1}' | head -1)
echo "== machine $M: units done $done / $total, sweep pid ${sweep:-NOT RUNNING}"
echo "running: $(ps -eo args | grep 'src.training.patchmix_cl.main' | grep -v grep | grep -v '^/bin/sh' | sed -E 's/.*--cell (\S+) .*--seed ([0-9]+).*/\1:\2/' | sort -u | tr '\n' ' ')"
echo "failed (sweep gave up): $(grep -c FAILED logs/grid-v1-$M.log)   out-of-memory requeues: $(grep -c 'out of memory' logs/grid-v1-$M.log)"
grep FAILED logs/grid-v1-$M.log | tail -5
nan=$(grep -l -E "loss nan|Score nan" logs/grid-v1-$M/*.log 2>/dev/null | tr '\n' ' '); echo "job logs with NaN loss or Score: ${nan:-none}"
col=$(for f in logs/grid-v1-$M/*.log; do [ $(grep -c '^ \* test' $f) -ge 3 ] && grep '^ \* test' $f | tail -1 | grep -q ' Se 0\.' && echo -n "$f "; done); echo "job logs collapsed (test Se below 1% from epoch 3): ${col:-none}"
echo "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader)   disk free: $(df -h . | tail -1 | awk '{print $4}')"
echo "last sweep lines:"; tail -3 logs/grid-v1-$M.log
EOF
}
bash -c "$(status A)"
timeout 60 ssh "$B" "cd ~/Workspace/Personal/KHKT-CVA-2026/auscult-trust && bash -c '$(status B | sed "s/'/'\\\\''/g")'" || echo "== machine B: ssh failed"
