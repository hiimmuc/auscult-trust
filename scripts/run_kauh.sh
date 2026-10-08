#!/bin/bash
# KAUH positive controls with the frozen refit models (seeds 0-4): window softmax for three conditions, then the analysis.
#   baseline weights, no SC            -> outputs/kauh/<OUT>/p0
#   baseline weights + test-time SC    -> outputs/kauh/<OUT>/p0_sc   (isolation contrast)
#   AuscultTrust weights + SC          -> outputs/kauh/<OUT>/auscult
# Units are taken from RUN_ID (default screen-ast) when their checkpoint exists on this machine; run it on every machine that
# holds refit checkpoints, copy the outputs/kauh/<OUT> folders together, then `bash scripts/run_kauh.sh analyse`.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-train/bin/python}
RUN_ID=${RUN_ID:-screen-ast}; OUT=${OUT:-final}
SCREEN=outputs/train/$RUN_ID/screen.json
SC_UNIT=${SC_UNIT:-outputs/train/$RUN_ID/sc+sc_static/cv0}   # any unit of the SC arm: s_ref and the coefficient bound
[ "$1" = analyse ] && exec $PY -m src.evaluation.kauh --dir outputs/kauh/$OUT --base p0 --arms p0_sc auscult
read E_BASE E_SC < <($PY -c "
import json; s = json.load(open('$SCREEN')); print(s['arms']['baseline']['epoch'], s['arms'][s['auscult_trust']]['epoch'])")
SC_CELL=$($PY -c "import json; s = json.load(open('$SCREEN')); print(s['arms'][s['auscult_trust']]['cell'])")
for s in 0 1 2 3 4; do
  B=outputs/train/$RUN_ID/baseline/seed$s; A=outputs/train/$RUN_ID/$SC_CELL/seed$s
  ck() { [ -f "${1/outputs/checkpoints}/$2" ]; }
  if ck $B report_epoch_$E_BASE.pth; then
    [ -f outputs/kauh/$OUT/p0/seed$s.npz ] || $PY -m src.evaluation.kauh_infer $B --ckpt report_epoch_$E_BASE.pth --out outputs/kauh/$OUT/p0/seed$s.npz
    [ -f outputs/kauh/$OUT/p0_sc/seed$s.npz ] || $PY -m src.evaluation.kauh_infer $B --ckpt report_epoch_$E_BASE.pth --sc --sc_state $SC_UNIT/sc_state.npz --sc_args $SC_UNIT/train_args.json --out outputs/kauh/$OUT/p0_sc/seed$s.npz
  fi
  if ck $A report_epoch_$E_SC.pth; then
    [ -f outputs/kauh/$OUT/auscult/seed$s.npz ] || $PY -m src.evaluation.kauh_infer $A --ckpt report_epoch_$E_SC.pth --sc --out outputs/kauh/$OUT/auscult/seed$s.npz
  fi
done
