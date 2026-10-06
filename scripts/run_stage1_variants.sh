#!/bin/bash
# Stage 1 variants of proposal v8: every variant runs the same seed list (paired comparison). Resumable per run.
# Usage: bash scripts/run_stage1_variants.sh [variant ...]   (default: all configs in configs/stage1_variants)
cd "$(dirname "$0")/../baselines/patchmix_cl" || exit 1
CONF=../../configs/stage1_variants
SEEDS=${SEEDS:-"0 1 2 3 4"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VARIANTS=${@:-$(cd $CONF && ls *.json | sed 's/\.json$//')}
for v in $VARIANTS; do
  for s in $SEEDS; do
    tag=${v}_seed${s}
    dir=save/icbhi_ast_patchmix_cl_${tag}
    [ -f $dir/stage1_report.json ] && continue
    .venv/bin/python main.py --config $CONF/$v.json --tag $tag --seed $s --resume $dir/last.pth
  done
done
