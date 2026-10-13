#!/bin/bash
# Inference-only LODO arms on the trained LODO units (src/evaluation/lodo_ttsc.py), then the H2 tables.
#   baseline weights: plain (embeddings), test-time SC on the held-out device only (primary) and on every device,
#                     A2, TTA-EQ (K = 8)
#   AuscultTrust weights: plain (embeddings), TTA-EQ, A2 after SC, target-sample sweep (N held-out patients for the coefficient)
# Resumable: finished arms are skipped.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-train/bin/python}; RUN=${RUN:-lodo-ast}; SC_CELL=${SC_CELL:-sc+sc_static}
$PY -m src.evaluation.lodo_ttsc outputs/train/$RUN/baseline+lodo_*/seed? --run $RUN \
  --arm baseline_plain:none --arm baseline_ttsc:target:static --arm baseline_ttsc_all:all:static \
  --arm baseline_a2:none:a2 --arm baseline_ttsc_a2:target:static:a2 --arm baseline_tta8:none:tta8
$PY -m src.evaluation.lodo_ttsc outputs/train/$RUN/$SC_CELL+lodo_*/seed? --run $RUN \
  --arm sc_plain:all --arm sc_a2:all:a2 --arm sc_tta8:all:tta8
# target-sample sweep: R = 50 draws on Litt3200, 10 on the other folds (docs/prereg-v9-amendment-2.md)
$PY -m src.evaluation.lodo_ttsc outputs/train/$RUN/$SC_CELL+lodo_litt3200/seed? --run $RUN --target_n 1,2,3,5,10,all --target_rep 50
$PY -m src.evaluation.lodo_ttsc outputs/train/$RUN/$SC_CELL+lodo_{meditron,littc2se,akgc417l}/seed? --run $RUN \
  --target_n 1,2,3,5,10,all --target_rep 10
echo LODO_INFER_DONE
