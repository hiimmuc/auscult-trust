# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status
Main line: proposal A+B v7 (`../docs/research-proposal-AB-gain-decomposition-v7-en.md`). Fallback: proposal H v6 (`../docs/research-proposal-H-fm-device-benchmark-v6-en.md`), public data only. v4 (`research-proposal-lung-sound-v4-en.md`) is superseded; its phase and short-window experiments are archived as reported negative or dataset-specific results.
Two parts, no calendar limits (order by dependency): Part I base model (encoder ladder L0-L4, ICBHI official split, fair val-selected protocol), Part II device shift (KAUH, ICBHI device-held-out, phantom, conformal coverage). Model name: **AuscultTrust**.
Data in `../data/raw/`: ICBHI (920 wav, 6898 cycles, 126 patients), HF_Lung_V1 (9765 wav), KAUH (336 wav = 112 patients x 3 filters). Reference repos in `../repos` (read-only): OPERA, patch-mix_contrastive_learning, SG-SCL.
Primary split: `../data/splits/icbhi_official_v1.json` (val carved from official train; 2 patients on both sides kept in test, their train recordings dropped). Results so far: `report.md`.
TODO (R-plan: R0 clean rerun, R1 encoder ladders L0-L1 on 5 encoders vs `../docs/papers/anchors.md`, R2 fine-tune ladder L2-L3, R3 improvements, R4 Part II, R5 Patch-Mix CL): KAUH E2 runs, E1 device-held-out runs (`device_holdout_official`), decodability and coverage-deficit tables per rung, A1/A2 wired into extraction, A3 (Tent on encoder norms, needs re-extraction), gain augmentation on a rung that sees the input, Patch-Mix CL seeds 2-5 (stopped by user after seed 1; partial seed 2 folder in `baselines/patchmix_cl/save/`).

## Commands
Env uses `uv`: `uv venv .venv && uv pip install -r requirements.txt`.
- Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests` (single: append `tests/test_core.py::test_icbhi_score`)
- AuscultTrust, resumable full pipeline per encoder (features -> L0 probe -> L1 head CV -> L2/L3 fine-tune CV (ast, hear) -> test): `setsid nohup bash scripts/run_ladder.sh ast > logs/ladder_ast.log 2>&1 < /dev/null &` (`opera_ct`, `opera_ce`, `clap`, `ast`, `hear`). Re-run the same command to resume. Keep `RUN_ID` fixed.
- Single steps: `.venv/bin/python -m src.auscult_trust cv-head|cv-ft|select|final|final-conformal|probe configs/ladder_ast.yaml [--cells a,b|report] [--alpha 0.1]`. Cells: `<last|concat|scalar>-<mean|max|meanmax|attn>`, `ft-k<k>-<pool>`, `lora-r<r>-<pool>`. `probe` is the L0 rung (logistic regression). `scripts/run_conformal_lora.sh` = conformal for finished cells + LoRA rung (AST).
- Feature cache for any encoder: `.venv/bin/python -m src.features configs/extract_<ast|hear|opera_ct|opera_ce|clap>.yaml [--limit N]` (KAUH windows: `extract_ast_kauh.yaml`). Split file: `python -m src.make_split configs/split_icbhi_official.yaml`.
- Legacy v4 stack (branch/phase, MLP head, Tent, conformal V1-V5 on it): `python -m src.legacy.<train|rq3|compare_encoders|ablate|hf_ablate|phase> configs/legacy/<config>.yaml`. Kept for the reported results in `report.md`, not extended; reads the old npz caches, no extractor.
- CPU-heavy extraction (`src.features`): prefix `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`, otherwise BLAS threads oversubscribe all cores and extraction runs ~20x slower.

## Code map
Run from repo root, imports are `src.*`. Flow: `data/icbhi.py` cycle table -> `data/splits.py` patient-disjoint split file -> `encoders/` (one `Encoder` interface; each keeps its own native preprocessing: OPERA 64-mel 16 kHz, AST 128-mel fbank, CLAP 48 kHz 10 s, HeAR 2 s PCEN clips; all give (32, D) frames) -> `features.py` writes `../data/cache/<encoder>_feat/` (`layers.npy` (N, L, 32, D), `tokens.npy` for ViT encoders only, `meta.json`, `keys.json`, `done.npy`) -> `auscult_trust/` (`data.py` FeatureData and grouped folds, `model.py` Head/Tail/AuscultTrust, `fit.py`, `probe.py` L0, `conformal.py` ConformalLayer, `__main__.py` CLI) -> `runlog.py` writes `outputs/<exp>/<run_id>/<cell>/seed<N>.json` (`test_pred` inside allows paired CIs offline).
Part II: `data/kauh.py` (loader, 6 ordered filter pairs, `partitions`), `data/splits.py` (`device_holdout_official`, `patient_device_crosstab`, `with_official`), `eval/shift.py` (decodability AUC + permutation test, coverage deficit, closed fraction, set stats, Spearman block bootstrap), `shift/correction.py` (A1 spectrum correction, A2 ISA moment matching, random bin gain), `conformal/conformal.py` (V1-V5). Archived: `src/legacy/` (branch/phase stack, MLP head, Tent, npz-cache reader), `configs/legacy/`. Code docstrings: Google style.

## Project
Lung-sound device-shift and reliability study. Full spec: proposal A+B v7.
Question: what fraction of stethoscope-induced shift in log-mel is a per-frequency gain, and does correcting it restore conformal coverage?
- H1: on the phantom, a gain predicted from measured H(f) explains >= 0.8 of the paired log-mel shift (relative to the placement-noise ceiling).
- H2: under device shift, split-conformal coverage falls below nominal; gain-based correction (A1, A2) closes part of the deficit, in proportion to the H1 explained fraction.
- H3: stethoscope-matched IR augmentation closes more deficit than generic microphone IRs (needs >= 3 distinct hardware).
- H4: fine-tuning raises ICBHI Score but also device decodability and coverage deficit, versus the frozen encoder.
- Part I criterion: best rung within 1 SD of or above Patch-Mix CL (62.37 +- 0.61) under the val-selected protocol; also report best-epoch-on-test for comparability, labelled.
Not claimed: a new correction method; diagnosis; guarantees on humans; vitals or clinician-agreement results (dropped from v4).

## Data tiers (hard rules)
| Tier | Data | Use |
|---|---|---|
| 0 | ICBHI, HF_Lung_V1, KAUH | Train, validate, calibrate, shift experiments |
| 1 | Lung phantom recordings | Evaluate only. Model, head and thresholds frozen and hashed before any phantom recording. No model selection. No conformal calibration for humans |
| 2 | Hospital patients | Test only. Locked. Opened once, per pre-registration, after ethics approval |

- Never read, load, or inspect Tier 2 data unless the task explicitly says the pre-registered analysis is running.
- Splits are patient-disjoint. Always.
- KAUH: one recording exists in 3 filter renderings (Bell/Diaphragm/Extended); unit = whole recording, label = recording-level sound label (`Bronchial`-only recordings excluded). All renderings of a patient go in the same split.
- ICBHI: device is confounded with class prior and site. 4 patients (112, 158, 218, 226) recorded on two devices; `device_holdout_official` drops them from the non-held-out parts. Device-held-out otherwise keeps the official split.
- Never join datasets across different people (e.g., ICBHI audio + MIMIC vitals).

## Model (AuscultTrust)
- Encoder: AST (AudioSet) is primary (frozen head-only Score 54.7 on ICBHI official test; report sections 8, 16). Comparators: HeAR, CLAP, OPERA-CT (last, 41.7). Non-OPERA results are primary for device claims (OPERA saw ICBHI-train and HF_Lung-train). Encoder input must match its pretraining preprocessing; bump the cache name when preprocessing changes.
- Ladder: L0 logistic regression (`probe`), L1 best frozen head by grouped CV, L2 last-k blocks tuned (`ft-k*`), L3 LoRA (`lora-r*`, on the blocks after cached layer 8 only), L4 Patch-Mix CL (`baselines/`). Same split, same grouped-CV epoch selection, no test-set selection.
- Head pools over real frames only (mask from cycle length; ~66% of an 8 s input is repeat padding). Class-weighted loss: config `class_weight: balanced` (new exp name).
- Conformal layer (`ConformalLayer`): LAC score, V1 split and V2 Mondrian calibrated on held-out dev patients (`final-conformal`); V3 weighted, V4 label shift, V5 k-shot in `src/conformal/conformal.py`.
- Corrections: A1 spectrum correction (device id needed), A2 ISA (log-mel moment matching), A3 Tent on encoder norms (deferred). After any adaptation, recompute calibration scores with the adapted model; calibration clips use source statistics, test clips target statistics.
- Dropped from the model: short-window branch (no gain), hard phase pooling (helped AST crackle F1 on HF_Lung only). Code archived in `src/legacy/`.

## Experiment rules
- Variance: >= 5 seeds for trained rungs; for deterministic parts (L0, frozen features) use >= 5 patient-disjoint partitions, or the fixed official split plus patient-level bootstrap. Report mean +- SD. A gain smaller than seed SD is not a result.
- Report per-class F1/recall, ICBHI Score, ECE, marginal and per-class coverage, coverage deficit, closed fraction, set size, singleton and empty-set rate. Compare coverage with the binomial CI of nominal for the actual test size.
- Patient-level bootstrap for CIs on Tier 0 and Tier 2. Never cycle-level.
- Log config, seed, data hash, and encoder for every run.
- Tune hyperparameters on val or grouped CV over train patients. Never on test.
- Negative results are kept and reported.

## Repo layout
```
data/        raw/ (read-only), cache/ (embeddings), splits/ (versioned split files)
src/         encoders/, features.py, auscult_trust/, data/, conformal/, eval/, shift/, legacy/
configs/     one file per experiment (extract_<enc>, ladder_<enc>, split_*); legacy/ for the archived v4 runs
outputs/     results: <exp>/<run_id>/<cell>/seed<N>.json, meta.json, summary.txt (gitignored)
checkpoints/ trained heads: <exp>/<run_id>/<cell>/seed<N>.pt (gitignored)
baselines/   patched copies of reference repos that need running; ../repos is read-only reference
```

## Working conventions
- `../repos/*` is read-only reference. Never edit, build, or write results there. Patched copies go in `baselines/`.
- Results: `<exp>` = config stem, `<cell>` = variant, `run_id` = `YYYYMMDD-HHMMSS[_$RUN_TAG]`, or a fixed `RUN_ID` for resumable pipelines; do not hand-name run folders otherwise.
- Install a missing package before reaching for a workaround.
- Simplest change that works. No speculative abstractions.
- Do not modify `data/raw/` or committed split files.
- Do not invent dataset statistics, citations, or results. Mark unknowns as TODO.
- Claims in docs use tags: [Fact], [Interpretation], [Hypothesis], [Assumption].
- Code comments and docs in English.

## Open items
See proposal A+B v7 Appendix B (B1-B11). Recount ICBHI per-device patients and cycles from filenames before using any device counts (done for patients: 32 AKGC417L, 11 Litt3200, 23 LittC2SE, 64 Meditron; 4 patients span two devices).
