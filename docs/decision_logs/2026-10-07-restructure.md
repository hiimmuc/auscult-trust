# 2026-10-07: repository restructure

[Fact] Layout now follows the conventional research layout: `src/{processing,training,evaluation}`, `experiments/`, `notebooks/`,
`data/` (README only), `checkpoints/`, `docs/`; reference repos, proposals and Phase 0 code live in the wrapper directory
(`../repos`, `../docs`, `../archive`). Git tag `pre-restructure` holds the old tree.

- Runs: `outputs|checkpoints/<exp>/<run_id>/<cell>/<unit>/`, new `run_id` per campaign, no overwrite, `meta.json` (config, git commit),
  `sc_state.npz` (s_ref, per-device spectra and coefficients, sc_mode) and its hash for every SC run.
- Patch-Mix CL: the model and loss are loaded from the read-only reference repository (`src/training/patchmix_cl/upstream.py`, identical results to the former patched copy: same loss and validation Score after one epoch); dataset, loop and reports live in `src/training/patchmix_cl`; one SC implementation (`src/processing/correction.py`), one metric implementation
  (`src/evaluation/metrics.py:by_device`, percent conversion in `reporting.py`), one split implementation (`src/processing/splits.py`).
- Removed (v8 variants, not v9 candidates): P2 bin_norm, P6 bin_affine, P7 hc_fuse, P8 worst-device selection, P9 combination;
  other backbones (CNN6, ResNet, EfficientNet, SSAST), raw-audio augmentation, weighted loss/sampler, RespireNet 80/20 folds, diagnosis labels.
- [Interpretation] Cycles are now sorted by file name (the old `set` order changed with the hash seed), so a seed gives the same batches in
  every process; v9 runs are not bitwise comparable with v8 runs.
- Summary rewritten: finalists = every screened arm, `auscult_trust` = better mean CV Score of `sc` / `sc_gain` (tie: `sc`), `sc_mode` chosen per SC arm by mean CV Score (tie: dynamic), non-inferiority check over seeds 0-9 with margin -1.5.
- Training env stays separate (`timm==0.4.5`, Python 3.10): `requirements-train.txt`.

- Naming: experiment `train` (was `stage1`), arms `baseline`, `sc`, `sc_gain`, `freq_mixstyle` (were P0, P1, P1+P3, P4), `run.sh` (was `run_stage1.sh`), files `report.json`, `screen.json`, `final.md`. Legacy runs sit in `outputs|checkpoints|logs/train_legacy` (v8 recipe, not comparable). Data scripts: `scripts/download_data.sh`, `scripts/prepare_data.py`.
