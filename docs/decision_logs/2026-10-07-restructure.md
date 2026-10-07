# 2026-10-07: repository restructure

[Fact] Layout now follows the conventional research layout: `src/{processing,training,evaluation}`, `experiments/`, `notebooks/`,
`data/` (README only), `checkpoints/`, `docs/`; reference repos, proposals and Phase 0 code live in the wrapper directory
(`../repos`, `../docs`, `../archive`). Git tag `pre-restructure` holds the old tree.

- Runs: `outputs|checkpoints/<exp>/<run_id>/<cell>/<unit>/`, new `run_id` per campaign, no overwrite, `meta.json` (config, git commit),
  `sc_state.npz` (s_ref, per-device spectra and coefficients, sc_mode) and its hash for every SC run.
- Patch-Mix CL folded into `src/training/patchmix_cl`; one SC implementation (`src/processing/correction.py`), one metric implementation
  (`src/evaluation/metrics.py:by_device`, percent conversion in `reporting.py`), one split implementation (`src/processing/splits.py`).
- Removed (v8 variants, not v9 candidates): P2 bin_norm, P6 bin_affine, P7 hc_fuse, P8 worst-device selection, P9 combination;
  other backbones (CNN6, ResNet, EfficientNet, SSAST), raw-audio augmentation, weighted loss/sampler, RespireNet 80/20 folds, diagnosis labels.
- [Interpretation] Cycles are now sorted by file name (the old `set` order changed with the hash seed), so a seed gives the same batches in
  every process; v9 runs are not bitwise comparable with v8 runs.
- Stage 1 summary rewritten for v9: no P8, finalists = every screened arm, `auscult_trust` = better mean CV Score of P1 / P1+P3
  (tie: P1), `sc_mode` chosen per SC arm by mean CV Score (tie: dynamic), gate G over seeds 0-9 with margin -1.5.
- Training env stays separate (`timm==0.4.5`, Python 3.10): `requirements-train.txt`.
