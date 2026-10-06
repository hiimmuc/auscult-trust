# Pre-registration, proposal v8

Date: 2026-10-06. Written and committed before any Stage-1 screening run (no variant P1-P9 has been trained at this date; P0 is the existing Patch-Mix CL reproduction, 5 seeds, report.md). Source: `../docs/research-proposal-v8-en.md` sections 4.1, 5.2, 5.6. Any change after the first screening run is a dated amendment at the end of this file, never an edit of this text.

## 1. Metrics

- Stage 1 (per device and pooled): Sp, Se, ICBHI Score = (Sp + Se) / 2, HS = harmonic mean of Sp and Se, per-class F1, macro-F1; 4-class and the 2-class view of the same 4-class predictions (not a separately trained model). Code: `src/eval/metrics.py` (`by_device`), `baselines/patchmix_cl/util/stage1.py`.
- Stage 2: marginal and per-class coverage, deficit Delta = (1 - alpha) - coverage, closed fraction phi = 1 - Delta_corrected / Delta_none (NaN unless Delta_none exceeds the binomial 95% half-width at the test size: `closed_fraction_checked`), set size, singleton and empty-set rate, ECE. alpha = 0.1.
- Prior-matched coverage (F1): importance weights make the test class mix equal the calibration mix, evaluation only (`prior_matched_coverage`). V4-oracle uses true class frequencies (`oracle_label_shift_threshold`).
- Decodability (H-diag): logistic probe on the first 8 principal components, learning curve n in {20, 50, 100, 200} rows per domain, MMD with patient-level permutation (`decodability_curve`, `mmd_permutation`). The full-dimensional AUC is reported but not interpreted.
- Uncertainty: patient-level bootstrap (1000) for E0, E1; patient-rotation bootstrap for E2. Coverage is judged against the binomial half-width at the actual test size.

## 2. Variant pool (every variant tried is reported, failures included)

P0 baseline. P1 A1 spectrum correction on the waveform (device from file name; reference = arithmetic mean of train devices, geometric as sensitivity). P2 per-device, per-mel-bin standardisation. P3 `random_bin_gain`, 6 dB SD. P4 Freq-MixStyle, p swept in {0.25, 0.5, 0.75}. P5 microphone IR augmentation: **dropped**, no IR set with a verified licence is available; reinstated by amendment if one is found. P6 per-mel-bin affine layer before the encoder. P7 fusion with the 20 frame-SDs of the MFCC (gain-invariant). P8 worst-device selection criterion. P9 combination of the two best single variants. Configs: `configs/stage1_variants/`. All variants use the same seed list (0-4) for paired comparison.

## 3. Selection rule

1. Screening: 3-fold patient-grouped CV on the 60% training patients, one seed per fold, every variant. Criterion: mean CV Score; ties by worst-device CV Score.
2. Finalists: top 3 by CV plus P0. Train on all training patients with the CV-chosen epoch count (validation-selected, `--selection cv`), 5 seeds, test once. This is the primary result.
3. Literature column: best epoch on test for every finalist, labelled "optimistic; for comparison only", with the expected-optimism term s * E[max of K normals] (K = 5: 1.16 s; K = 10: 1.54 s; s = seed SD / sqrt(5)) next to it.
4. Improvement claims use paired seeds: Delta Score with a CI from the five paired differences.
5. Gate G-base: Stage 2 uses the CV-selected finalist if its paired Delta Score over P0 has a CI above 0, otherwise P0 (last-epoch protocol). Stage 2 never uses a test-selected checkpoint.

## 4. Hypotheses and rejection thresholds

| H | Rejected if |
|---|---|
| H1 | V1 coverage deficit within the binomial 95% half-width on at least 2 of the 3 usable ICBHI devices (AKGC417L, Litt3200, Meditron) |
| H-diag | handcrafted (74-dim) decodability >= embedding decodability, CIs overlapping or above, same non-saturating probe |
| H3 | phi CI includes 0, or mean set size > V3, on at least 2 of the shift conditions that show a deficit; tested only where Delta(V1) exceeds its half-width |
| H-inv | paired Delta Score CI includes 0, or decodability or Delta rises |
| H2, H3-link, H3-IR | pending hardware; thresholds registered before the first phantom recording (H2: explained fraction < 0.8) |
| F4 (finding) | not a hypothesis: fine-tuning raising decodability and coverage deficit is reported as a negative finding |

## 5. Stage 2 design fixed here

Calibration patients: official test patients of the non-held-out devices, split into calibration and evaluation groups stratified by device, 20 re-splits (`calibration_eval_splits`). Held-out-device patients are never calibration or training patients; the 4 patients on two devices (112, 158, 218, 226) are dropped from the non-held-out parts (`device_holdout_official`). KAUH: 5-fold patient rotation (`rotation_partitions`), all filter renderings of a patient in the same part. V5 k counts patients, k in {5, 10}. The model is frozen and its weights hashed before calibration.
