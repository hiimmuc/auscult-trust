# Amendment 1 to `prereg-v8.md` (proposal v9)

**Status: registered 07/10/2026.** Appended to the end of `prereg-v8.md`; committed before any Stage-1 run.
- **Earlier runs:** none. No Stage-1 run (v8 or v9) had started before this date (`save/` and `baselines/patchmix_cl` carry no run output).

Date of registration: 07/10/2026. Source: `docs/research-proposal-v9-en.md`.

This amendment replaces prereg-v8 sections 1–5 for every primary claim. v8 text stays as history.

## 1. Reason

The registered title (fixed) commits the study to spectrum correction (SC) and conformal prediction. v9 fixes the SC family as the proposed model and reduces the hypotheses to one chain, H3 → H1 → H2, plus a Stage-1 gate and two positive controls.

## 2. Metrics (replaces v8 §1)

**Stage 1.**
- Sp, Se, Score, HS, per-class F1, macro-F1, per device and pooled.
- 4-class, plus the 2-class view of the same predictions.

**Stage 2.**
- Coverage and Δ = coverage − 0.90; \|Δ\|.
- Set size; singleton and empty rates.
- Flip rate and TV distance on paired units.
- Screening view: referral rate, screening Se and Sp.
- α = 0.1; LAC score.

**Prior-matched coverage.** Test units are importance-weighted to the calibration class mix. This is used for evaluation only.

**Uncertainty.**
- 1,000 bootstrap draws by patient (by source patient on the phantom), paired across arms.
- Seeds are resampled for trained arms.
- Conformal calibration is redone inside every draw whenever coverage is the statistic.

## 3. Stage 1 (replaces v8 §2–3)

**Arms.** Same seeds in every arm.

| Arm | Config |
|---|---|
| P0 | `P0_baseline` |
| P1 | `P1_a1_input` |
| P1P3 | `P1P3_sc_gain` (random per-bin gain, 6 dB SD) |
| P4 | `P4_freq_mixstyle` (p = 0.5) |

**SC.**
- Reference: arithmetic mean of the training-device mean spectra.
- STFT: n_fft 1024, hop 512, 16 kHz.
- Coefficient bound, `sc_mode` arg (`src/shift/correction.py:limit_coefficients`, mirrored in `util/stage1.py:a1_coefficients`): `dynamic` clips every bin to ±`limit_freq_diff` dB (default 20); `static` leaves `[limit_freq_low, limit_freq_high]` Hz (default 50–2,000) unclipped and sets the correction to 1 (no-op) outside it. Not a reported ablation: at CV screening (3-fold, same as P1 vs P1P3), run both `sc_mode` values for every SC arm and keep whichever gives the higher mean CV Score as a fixed implementation choice; tie goes to `dynamic` (default; simpler, no band edge).
- Device taken from the file name.
- s_ref and the coefficients are saved with each checkpoint.

**Screening.** 3-fold patient-grouped CV (`--cv_folds 3`), one seed per fold, all four arms.

**AuscultTrust.**
- P1 or P1P3, whichever has the higher mean CV Score; ties go to P1.
- If P1P3 is not screened by 10/10, AuscultTrust = P1 and P1P3 is an ablation.

**Refit.**
- All four arms at their CV epoch (`--selection fixed`), seeds 0–4, test once. This is the primary benchmark result.
- The literature column (best epoch on test) is labelled optimistic.

**Gate G.**
- P0 and AuscultTrust get 5 more seeds (5–9).
- Statistic: ΔScore (AuscultTrust − P0) over 10 seeds; 95% t-interval.
- Pass if the lower bound is > −1.5 points.
- If G fails, Stage 2 uses P0 with test-time SC as the SC arm.

**v8 variants P2, P6, P7, P8, P9.** Not candidates; reported in an appendix if run.

**TTA-EQ (cold-start comparator, F5; inference only).**
- Mean softmax over K = 8 views. Each view applies a random smooth per-bin gain drawn from the P3 distribution (6 dB SD).
- Fixed seed list for the views: 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007 (offset from the training/gate seeds 0–9 so a global-RNG seed call cannot correlate a view with a training run).
- Conformal calibration scores use the same K-view ensemble.
- K ∈ {1, 4, 8, 16} is reported only as a cost curve. The verdict-free F5 comparison uses K = 8.
- Rejected before any run: consensus over several resampling rates (low-pass only; in-band shift untouched).

## 4. Hypotheses, controls and rejection rules (replaces v8 §4)

| ID | Primary test | Rejected if |
|---|---|---|
| H1 | **Phantom**, ≥ 3 distinct transducers, all pairs. Flip rate of cycle 4-class argmax, SC pipeline (AuscultTrust + SC) minus baseline pipeline (P0, no SC), pooled over pairs; bootstrap by source patient | 95% CI includes 0 or lies above 0 |
| H2 | **ICBHI LODO**, 4 folds, co-primary with **phantom** pairs. D = mean over tested folds of (\|Δ_base\| − \|Δ_SC\|), with Δ from prior-matched coverage; seeds 0–4 per arm. Two-level bootstrap (seeds; calibration and evaluation patients), recalibrating in each draw. Mean set-size change reported with its CI | CI of D includes 0, or the set-size change CI lies above 0 |
| H3 | **Phantom.** Explained fraction = R²_gain / R²_ceiling per transducer pair (proposal §5.6) | Median over pairs < 0.8 |

**H2 null gate.** A fold (or phantom pair) is tested only if \|Δ_base\| exceeds the 95th percentile of \|Δ\| from 200 exchangeable re-splits of the source-device patients, with the same calibration and evaluation sizes. If no LODO fold passes, report "not detectable on ICBHI" and decide H2 on the phantom.

**Positive controls.** Must pass; they support no claim. A failure stops interpretation until the SC code and filter linearity are checked.
- **KAUH-A:** the flip rate across the 3 unordered filter pairs (window 4-class argmax) drops with SC. Paired bootstrap; CI below 0.
- **KAUH-B2:** \|Δ\| drops with SC on the ordered filter pairs that pass the same null gate. Calibration uses the source-filter recordings of the other 4 patient groups in a 5-fold rotation.

**Isolation contrast** (P0 weights with vs without test-time SC): reported for H1 and H2. No verdict.

**Comparator arms** (A2, TTA-EQ, SC + TTA-EQ, P4): reported with the same statistics. No verdict. H2 compares SC with the baseline, not with the best arm.

## 5. Stage 2 design (replaces v8 §5)

**LODO.**
- Hold out all recordings of one device: Meditron, LittC2SE, Litt3200, AKGC417L.
- A patient with any recording on the held-out device is removed from training and calibration. Their held-out-device recordings stay in test.
- Calibration: 20% of the remaining patients, stratified by device, seed 0.
- Recipe and epoch count frozen from Stage 1; seeds 0–4.
- Arms:
  - trained: P0, P4, AuscultTrust;
  - inference only: P0 + test-time SC; TTA-EQ on P0 and on AuscultTrust.
- Secondary protocol matching [46] (arXiv 2605.29862v2), reference only, no verdict:
  - every patient on > 1 device removed;
  - folds AKGC417L, Meditron, pooled Littmann (LittC2SE + Litt3200);
  - same arms, seeds 0–4.

**E0.**
- Official-test patients split into calibration and evaluation halves, stratified by device, 20 re-splits.
- The 2 official-split patients on both sides are excluded.

**Screening view.**
- 8-s windows over the whole recording.
- Recording p(abnormal) = mean over windows of 1 − p(normal); max as sensitivity.
- ICBHI recording label: abnormal if any cycle has a crackle or a wheeze.
- B1 threshold: E0 calibration halves.

**KAUH.**
- The one bronchial-only recording is excluded.
- All filter renderings of a patient stay in the same part.
- The SC target spectrum is estimated per filter from unlabelled recordings.

**Phantom.**
- 1,200 ICBHI official-test cycles.
- Calibration and test halves split by source patient, 20 re-partitions.
- Thresholds are calibrated on transducer A clips only; never used for model selection.

**Freezing.** Weights, s_ref and coefficients are hashed before any Stage-2 calibration and before the first phantom recording.

## 6. Findings registered without direction

F1–F5 as in proposal v9 §4.3.
