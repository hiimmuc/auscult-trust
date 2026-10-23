#!/bin/bash
# H2 tables per source cell: base = the no-SC cell, arms = test-time SC (static, dynamic), A2, TTA-EQ, and trained SC
# (static, dynamic) when run; BEATs PAFA also gets PAFA + PC-MCL with trained SC static and dynamic.
cd "$(dirname "$0")/.." || exit 1
SC="ast+ft+patchmix_cl ast+ft+pafa htsat+ft+patchmix_cl htsat+ft+sg_scl beats+ft+patchmix_cl beats+ft+pafa beats+ft+sg_scl"
DYN="beats+ft+patchmix_cl beats+ft+pafa beats+ft+sg_scl"
for c in ast+ft+ce ast+ft+pafa ast+ft+patchmix_cl htsat+ft+ce htsat+ft+patchmix_cl htsat+ft+sg_scl beats+ft+ce beats+ft+patchmix_cl beats+ft+sg_scl beats+ft+pafa beats+ft+qlung; do
  arms="$c+ttsc_static $c+ttsc_dynamic $c+a2 $c+tta8"
  echo " $SC " | grep -q " $c " && arms="$arms $c+sc+sc_static"
  echo " $DYN " | grep -q " $c " && arms="$arms $c+sc"
  [ "$c" = beats+ft+pafa ] && arms="$arms beats+ft+pafa_pcmcl+sc+sc_static beats+ft+pafa_pcmcl+sc"
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv-train/bin/python -m src.evaluation.lodo --run lodo-v1 --base $c --arms $arms --out h2_$c > logs/h2_$c.log 2>&1 &
done
wait
echo "all done"
