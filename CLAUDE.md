# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Main line: **proposal v9** (`docs/research-proposal-v9-en.md`, 07/10/2026). Registered title (fixed): *Nghiên cứu giải pháp cải thiện độ tin cậy của mô hình phân loại âm thanh phổi dưới sự dịch chuyển thiết bị bằng hiệu chỉnh phổ và dự đoán Conformal cho hỗ trợ sàng lọc bệnh đường hô hấp*. v8 and earlier (A+B, H, v4) are superseded; their results stay in `report.md` as Phase 0.
Pre-registration: `docs/prereg-v8.md`, amended 07/10/2026 (`docs/prereg-v9-amendment.md`, appended to the end of `prereg-v8.md`; registered, no Stage-1 run preceded it).
Data in `../data/raw/`: ICBHI (920 wav, 6898 cycles, 126 patients), HF_Lung_V1 (9765 wav, not used in v9 core), KAUH (336 wav = 112 patients x 3 filters). Reference repos in `../repos` (read-only): OPERA, patch-mix_contrastive_learning, SG-SCL.
Model name: **AuscultTrust** = SC front end + AST (Patch-Mix CL recipe) + split-conformal layer (cycle 4-class view + recording-level screening view).
Dates: school round 17/10/2026 (10-min talk + poster, results report required); research cut-off 31/01/2027.

## Project (v9)

One question: does spectrum correction (SC) let a benchmark-level lung-sound classifier keep its predictions and its conformal coverage when the stethoscope changes, without labels from the new stethoscope?

- Arms: "SC pipeline" = AuscultTrust weights + SC at test; "baseline pipeline" = P0 weights, no SC; isolation contrast = P0 weights with vs without test-time SC (reported, no verdict).
- **G (Stage 1 gate):** ΔScore (AuscultTrust - P0) over 10 seeds (0-9), CV epoch, 95% t-interval lower bound > -1.5 points (margin = about half the published AST-FT to Patch-Mix CL gap). Seeds 0-4 are the benchmark report. If G fails: Stage 2 uses P0 + test-time SC.
- **H1 (invariance):** SC lowers the prediction flip rate between devices on the same sound. Primary: phantom transducer pairs [pending]. Cycle 4-class argmax, bootstrap by source patient.
- **H2 (coverage):** SC lowers |Δ| (Δ = prior-matched coverage - 0.90) on an unseen device, set size not larger. Primary: ICBHI LODO, D = mean over tested folds of |Δ_base| - |Δ_SC|, two-level bootstrap (seeds, patients) with recalibration in every draw; co-primary phantom. Null gate: a fold is tested only if |Δ_base| > 95th percentile of |Δ| over 200 exchangeable re-splits of source-device patients with the same sizes.
- **H3 (mechanism, pending):** explained fraction R²_gain / R²_ceiling of the paired log-mel shift, median over transducer pairs >= 0.8 (ceiling from a remove-and-replace repeat of the same transducer).
- **Positive controls (must pass, no claim):** KAUH-A flip rate drops with SC; KAUH-B2 |Δ| drops with SC on pairs that pass the null gate. KAUH filters are vendor software and near-linear, so SC should succeed by construction; a failure means broken SC code or a non-linear filter.
- **Findings, no direction:** F1 per-device Sp/Se/Score/HS; F2 LODO benchmark (P0, P4, AuscultTrust; optional AST-CE, SG-SCL) and class-mix vs device split of coverage error; F3 screening view (referral rate, screening Se/Sp; B1 ICBHI-calibrated threshold on KAUH); F4 SC practicality (n unlabelled target recordings, class-mix sensitivity, arithmetic vs geometric reference); F5 comparators, each with and without SC: A2 test-time statistics, TTA-EQ (cold start, no device info: mean softmax over K = 8 random smooth per-bin gain views from the P3 distribution, 6 dB SD, fixed seeds 1000-1007; calibration uses the same ensemble; view disagreement reported as a label-free shift score), k-shot recalibration. Comparators get no verdict; H2 is SC vs baseline. Multi-rate resampling consensus was rejected (low-pass only, no-op on 4 kHz data).
- Prior work [46] (Koo et al., arXiv 2605.29862v2): federated device leave-out on ICBHI + SPRSound, Score only, 3 seeds. Its deterministic device-mean removal failed on AKGC417L (Sp 11.49) and Yunting: treat SC risk R4 as high. Its stochastic spectral perturbation helped most (supports P1P3).
- Not claimed: a new algorithm; SOTA Score; disease diagnosis (screening endpoint = abnormal lung sound); guarantees on patients; KAUH software filters as real hardware.

### Stage 1 (model)

- Official 60/40 split, 4-class primary, 2-class view of the same predictions, Sp/Se/Score/HS per device and pooled, 5 seeds (0-4, same in every arm).
- Arms: P0 Patch-Mix CL; P1 = P0 + SC (`--a1_input`); P1+P3 = SC + random per-bin gain (6 dB SD); P4 Freq-MixStyle p = 0.5 (competitor). AuscultTrust = P1 or P1+P3 by mean 3-fold CV Score (tie: P1). v8 variants P2, P6, P7, P8, P9 are not candidates; report them in an appendix if run.
- SC: STFT n_fft 1024, hop 512, 16 kHz; reference = arithmetic mean of training-device mean spectra; `sc_mode` (`--sc_mode`, `src/shift/correction.py:limit_coefficients`): `dynamic` (default) clips every bin to +-20 dB (4 kHz devices have no energy above 2 kHz); `static` leaves 50-2000 Hz unclipped, no-op outside. Not a reported ablation: at CV screening run both, keep the higher mean CV Score per SC arm, tie -> `dynamic`. Device from file name; target-device spectrum from unlabelled clips only.
- Primary result: refit all four arms at their CV epoch (`--selection fixed`), seeds 0-4, test once; seeds 5-9 for P0 and AuscultTrust (gate G). Literature column: best epoch on test, labelled optimistic. Stage-1 SC estimates test-device spectra from unlabelled official-test clips (transductive, as in [3]).
- Hand-off to Stage 2: frozen checkpoint + SHA-256, saved s_ref and coefficients, exported softmax (`export_probs.py`).

### Stage 2 (experiments; conformal score LAC, alpha 0.1)

- Unit follows labels: ICBHI and phantom = cycle, 4-class; KAUH = recording, normal vs abnormal (screening view: 8 s windows, recording p(abnormal) = mean over windows of 1 - p(normal); max as sensitivity). Triage: {normal} no referral, {abnormal} refer, {both} uncertain, empty re-record.
- E0 reference: official-test patients split into calibration/evaluation halves, stratified by device, 20 re-splits (`calibration_eval_splits`).
- Test 1 LODO: hold out all recordings of one device (Meditron, LittC2SE, Litt3200, AKGC417L); train on the rest minus 20% calibration patients (stratified by device); a patient with any recording on the held-out device leaves training and calibration (their held-out-device recordings stay in test); recipe and epoch count frozen from Stage 1; 5 seeds. AKGC417L fold is class-mix dominated (training keeps at most 321 of 1,864 crackle cycles): report, flag, add a macro mean without it. Arms: P0, P4, AuscultTrust (trained); P0 + test-time SC, TTA-EQ on P0 and AuscultTrust (inference only). Secondary protocol matching [46] (reference only): every multi-device patient removed; folds AKGC417L, Meditron, pooled Littmann.
- Test 2 KAUH, frozen Stage-1 model, no KAUH training. A: paired invariance across Bell/Diaphragm/Extended (flip rate, TV distance, set change) = positive control. B1: ICBHI threshold (E0 calibration halves) applied per filter (deployment check, F3). B2: 5-fold patient rotation, calibrate on source-filter recordings of the other 4 groups, test target filter of the held-out group, 6 ordered pairs = positive control. C: A and B again with SC (target spectrum per filter from unlabelled recordings) and with A2.
- Test 3 phantom [pending]: 1,200 ICBHI official-test cycles replayed per transducer; calibration/test halves and bootstrap by source patient; sweep ratio Y_B/Y_A (no reference sensor); 100-clip remove-and-replace repeat per transducer for the ceiling; conformal calibrated on transducer A clips (phantom-internal, allowed), tested on B clips.
- Stage 2 excludes the 2 patients that the published official split puts on both sides. ICBHI recording label for the screening view: abnormal if any cycle has a crackle or wheeze.

## Commands

Env uses `uv`: `uv venv .venv && uv pip install -r requirements.txt`.

- Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests` (single: append `tests/test_core.py::test_icbhi_score`). `test_registry_lists_all_encoders...` needs `pytorch_lightning` (OPERA).
- Stage 1 (v9 core): `bash scripts/run_stage1.sh screen P0_baseline P1_a1_input P1P3_sc_gain P4_freq_mixstyle` then `cd baselines/patchmix_cl && .venv/bin/python stage1_summary.py screen`; `bash scripts/run_stage1.sh final`; `.venv/bin/python stage1_summary.py final` -> `save/stage1_final.md`. Resumable; finished runs are skipped. Single run: `baselines/patchmix_cl/main.py --config <json[,json]> --tag <tag> [--cv_folds 3 --cv_fold k | --selection fixed --report_epochs E]`.
- Freeze/export: `baselines/patchmix_cl/export_probs.py <run_dir> [--ckpt report_epoch_<E>.pth]` -> softmax, embeddings, `model.sha256`.
- Phase 0 tools (kept for `report.md`, not extended): encoder ladder `scripts/run_ladder.sh <enc>`, `python -m src.auscult_trust ...`, feature cache `python -m src.features configs/extract_<enc>.yaml`, shift study `src/shift_study.py`, legacy v4 stack `python -m src.legacy.*`.
- CPU-heavy extraction: prefix `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`.

## Code map

Run from repo root, imports are `src.*`.

- Stage 1: `baselines/patchmix_cl/` (patched copy of Patch-Mix CL): `main.py` (selection cv/test/fixed, `--cv_folds`, variant flags), `util/icbhi_dataset.py` (P1 SC on the waveform, P2, P7), `util/stage1.py` (per-device metrics, folds, `a1_coefficients`, `pick_epochs`, `cv_epoch`), `stage1_summary.py` (screen ranking, final table, paired CIs, expected-max optimism), `export_probs.py`. Variant configs: `configs/stage1_variants/`.
- Stage 2: `src/shift/correction.py` (A1 SC, A2 ISA, random bin gain), `src/conformal/conformal.py` (V1, V5 in patients, V4-oracle; V2/V3/V4 kept, not core), `src/eval/shift.py` (coverage deficit, prior-matched coverage, set stats, binomial CI), `src/eval/metrics.py` (Sp/Se/HS/macro-F1 per device), `src/data/kauh.py` (loader, ordered filter pairs, `rotation_partitions`, `windows`), `src/data/splits.py` (`device_holdout_official`, `calibration_eval_splits`).
- Phase 0 only: `src/encoders/`, `src/features.py`, `src/auscult_trust/` (ladder), `src/features_handcrafted.py`, decodability/MMD in `src/eval/shift.py`, `src/legacy/`.
- Docstrings: Google style.

## Data tiers (hard rules)

| Tier | Data | Use |
| --- | --- | --- |
| 0 | ICBHI, KAUH (HF_Lung_V1 Phase 0 only) | Train, select, calibrate, shift experiments |
| 1 | Lung phantom recordings | Evaluate only. Weights and s_ref frozen and hashed before any phantom recording. No model selection. Phantom-internal conformal calibration (transducer A clips) is allowed for phantom tests only |
| 2 | Hospital patients | Out of scope for v9. Never read or load |

- Splits are patient-disjoint. Always. Calibration patients are never used for training, selection or evaluation.
- KAUH: 3 filter renderings of one recording (Bell/Diaphragm/Extended) share a patient and a split; unit = whole recording; label = recording-level sound label mapped to normal vs abnormal; `Bronchial`-only recording excluded. KAUH device = Littmann 3200 (same model as ICBHI Litt3200).
- ICBHI: device is confounded with class mix and site. 4 patients (112, 158, 218, 226) recorded on two devices.
- Target labels are used only for evaluation, V5 k-shot and V4-oracle rows. SC and A2 use unlabelled target audio only.
- Never join datasets across different people.

## Experiment rules
>
- >= 5 seeds for trained arms, same seeds across arms (paired). Report mean +- SD. A gain smaller than seed SD is not a result.
- Report Sp, Se, Score, HS per device and pooled; coverage, |coverage - nominal|, set size, singleton and empty-set rate; flip rate and TV distance for paired-device data. Compare coverage with the binomial CI of nominal for the actual test size.
- Patient-level bootstrap for CIs (1,000), paired across SC/no-SC on the same resampled patients. Never cycle-level. Phantom: clip bootstrap.
- Log config, seed, data hash, checkpoint hash, s_ref hash for every run.
- Tune on grouped CV over training patients. Never on test. Test-selected numbers only in the labelled literature column.
- Negative results are kept and reported.

## Working conventions

- `../repos/*` is read-only reference. Never edit, build, or write results there. Patched copies go in `baselines/`.
- Results: `<exp>` = config stem, `<cell>` = variant, `run_id` = `YYYYMMDD-HHMMSS[_$RUN_TAG]`, or a fixed `RUN_ID` for resumable pipelines.
- Install a missing package before reaching for a workaround.
- Simplest change that works. No speculative abstractions.
- Do not modify `data/raw/` or committed split files.
- Do not invent dataset statistics, citations, or results. Mark unknowns as TODO.
- Claims in docs use tags: [Fact], [Interpretation], [Hypothesis], [Assumption], [Pending].
- Code comments and docs in English.

## TODO (v9, in dependency order)

1. ~~Register: choose clip vs common band and the TTA-EQ seed list, then date, append and commit~~ — done (`docs/prereg-v9-amendment.md`, 07/10/2026; `sc_mode` decided by CV, not by hand). Still open: recount ICBHI cycles per device from filenames ([8] and [14] disagree on Litt3200/LittC2SE labels).
2. SC safety: done — `sc_mode` (`dynamic` clip, `static` band) in `util/stage1.py:a1_coefficients` / `limit_coefficients` and `src/shift/correction.py:limit_coefficients`, unit tested (`tests/test_correction.py`). Still open: save s_ref, per-device coefficients and the screened `sc_mode` with every P1 run; add `--sc_mode` to the Stage-1 screen so both values are run per SC arm before picking one (`stage1_summary.py`, TODO 3).
3. Stage 1 screen + final for P0, P1, P1P3 (`P1P3_sc_gain.json`), P4. Needed first: `stage1_summary.py screen` auto-adds P8 (from the P0 folds) and keeps only top 3 + P0, so P8 can push P4 or an SC arm out. Add a v9 mode: no P8, finalists = every screened arm, write `auscult_trust` = better mean CV Score of P1 / P1P3 (tie: P1) to `stage1_screen.json`.
4. External-audio inference for the frozen Patch-Mix model: KAUH 8 s windows and phantom clips -> softmax npz, with optional SC from saved s_ref + target spectrum from unlabelled clips (`--sc_n` to cap the number of target recordings, F4).
5. Paired invariance metrics in `src/eval/shift.py`: flip rate, TV distance, set change rate, patient-paired bootstrap.
6. Screening view: window -> recording aggregation (mean, max), recording-level labels for ICBHI and KAUH, 2-class conformal, triage counts.
7. LODO split over all recordings (`device_holdout_all` in `src/data/splits.py` and a `--holdout_device` path in `baselines/patchmix_cl`): a patient with any held-out-device recording leaves training and calibration; calibration carve-out 20% stratified by device. H2 statistic: prior-matched |Δ|, two-level bootstrap with recalibration, null gate from 200 exchangeable re-splits.
8. KAUH B2 with the frozen model: calibrate on the other 4 rotation groups (current `rotation_partitions` gives a single group as `cal`, about 22 patients: add a variant), test target filter of the held-out group.
9. A2 at test time on fbank for the frozen model (per-mel-bin mean/SD of target mapped to training statistics). TTA-EQ at test time: reuse `random_bin_gain` (P3) to draw K views, average softmax, store per-view probs for the disagreement score; same ensemble for calibration and test. LODO secondary split matching [46] (`device_holdout_all(..., drop_multi_device=True)`, pooled-Littmann fold).
10. Phantom capture and analysis scripts (sweep ratio, clip-half gain, ceiling, explained fraction) once hardware is fixed.
