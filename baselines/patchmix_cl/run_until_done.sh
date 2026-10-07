#!/bin/bash
# Re-run run_repro.sh (which resumes from last.pth) until the seed's results.json entry exists.
s=$1
until grep -q "seed${s}_best_param" save/results.json 2>/dev/null; do
  ./run_repro.sh $s >> logs/seed${s}.log 2>&1
  sleep 20
done
