# Amendment 2 to `prereg-v8.md` (proposal v9): exploratory arms and controls after the 2026-10-09 report

**Status: registered 14/10/2026, committed before the first run of every arm listed in section 3.** Runs that started earlier on this day are listed in section 1 with their status.

Nothing here changes a registered rule of Amendment 1. Gate G, H1, H2, H3 and the positive controls keep their registered arms, recipe, epochs and statistics. Every item below is **exploratory: reported with the same statistics, no verdict**, unless it is called a sensitivity analysis of a registered result, in which case it is reported next to that result and does not replace it.

## 1. Already running when this was written

| Campaign | RUN_ID / cell | Started | Status |
|---|---|---|---|
| `sc_gain` rerun after the `db_scale` fix (`docs/decision_logs/2026-10-14-db-scale.md`) | `screen-ast-c0` / `sc_gain`, `sc_gain+sc_static` | 14/10 03:35 (+0700), commit `a442947` | 3-fold screen, registered rule (both bounds, higher mean CV Score, tie dynamic); refit seeds 0-4 at the chosen epoch follows |
| Epoch control, official split | `screen-ast` / `baseline_e14` | 14/10 04:00, commit `cc56136` | baseline recipe, seeds 0-9, scored at epoch 14 (the AuscultTrust CV epoch) |
| Epoch control, LODO | `lodo-ast` / `baseline_e14+lodo_<dev>` | 14/10 04:01, commit `cc56136` | 4 devices x seeds 0-4 at epoch 14 |

**Reading of the epoch control, fixed before any of its numbers are seen.**
- KAUH A: if `p0_e14_sc` (baseline weights at epoch 14 with test-time SC) keeps a flip rate near 0.10, the missing invariance of the AuscultTrust weights comes from training with SC; if it rises towards 0.17, it comes from training longer.
- LODO Litt3200: D(SC vs baseline at epoch 2) is compared with D(SC vs baseline at epoch 14). A large drop means part of the H2 effect is the epoch.
- Gate G: ΔScore of AuscultTrust against the baseline at epoch 14, ten seeds, paired t-interval and patient bootstrap. Sensitivity analysis; the registered gate stays as computed.

## 2. Inference-only arms on the trained LODO units (`src/evaluation/lodo_ttsc.py`)

Same calibration and test cycles as the trained unit; weights unchanged (checkpoint hash stored per arm).

| Cell | Weights | Change at test time |
|---|---|---|
| `baseline_ttsc` (primary isolation contrast for H2) | baseline | SC on the held-out device only, towards the mean spectrum of the source devices, static bound |
| `baseline_ttsc_all` | baseline | SC on every device, static bound |
| `baseline_a2`, `baseline_ttsc_a2` | baseline | A2 (per-bin moments of the unlabelled held-out images mapped onto the training images), without / with target SC |
| `baseline_tta8`, `sc_tta8` | baseline / AuscultTrust | TTA-EQ as registered in Amendment 1 (K = 8, 6 dB SD after the `db_scale` fix, seeds 1000-1007, same views for calibration and test) |
| `sc_a2` | AuscultTrust | A2 after SC |
| `baseline_plain`, `sc_plain` | baseline / AuscultTrust | none (reproduces the trained unit; saves the embeddings for the probe) |

H2 statistics are reported for each cell against `baseline`. The epoch-matched D (`--base baseline_e14 --arms sc+sc_static`) is a sensitivity analysis.

**Target-sample sweep (F4).** AuscultTrust weights. The held-out coefficient is estimated from N held-out patients, N in {1, 2, 3, 5, 10, all but 5}, capped at (held-out patients - 5); the model is evaluated on the other held-out patients, next to the full-estimate model on the same patients. R = 50 draws per N and seed (one draw for "all but 5") on the Litt3200 fold; R = 10 on the other three folds (cost: every draw re-runs the model on the evaluation patients; fixed on 14/10 before any F4 run). Reported: |Δ|, Score, RMS dB distance of the coefficient to the full estimate.

**Inductive SC on the official split (A5).** AuscultTrust refit seeds 0-4 exported with `--sc_test_profile source` (each test device uses its training-clip coefficient). Score reported next to the registered transductive number.

## 3. New trained arms

### 3.1 `sc_cons` (SC + consistency)
- Config `experiments/train/sc_cons.yaml`: SC with the static bound; Jensen-Shannon consistency loss between the softmax of each training image and of the same image plus a smooth random per-bin gain (6 knots, linear, one curve per sample); weight λ = 1.0 ramped linearly over 5 epochs; gain SD 2 dB = σ_N10 (2.23 dB, median SD of a device coefficient estimated from 10 patients, `outputs/sc_diag/summary.json`) rounded; no random-gain augmentation.
- Comparator, paired seeds and folds: `sc_gain_v2` (SC + random gain after the `db_scale` fix, with the bound chosen by `screen-ast-c0`).
- Protocol: RUN_ID `screen-ast-cv5`, 5-fold CV, `--stop_epoch 20`, CV epoch by the registered rule; refit seeds 0-4 at that epoch (`screen-ast`); LODO 4 devices x seeds 0-2 at that epoch (`lodo-ast`).
- **Stop rule (decided now):** stop `sc_cons` after the screen if its mean Se over epochs 10-20 is more than 3 points below `sc_gain_v2`, or its worst-device mean Score over the same epochs is more than 3 points below. Reported as a negative result.
- Key reading: KAUH A flip rate of `sc_cons` (with test-time SC) against `p0_sc` (0.104). No verdict.

### 3.2 LODO benchmark at epoch 50 (C2)
Fine-tuned AST, HTS-AT, DASS and frozen AST, OPERA-CT; baseline recipe of each encoder; all four folds; seeds 0-4; scored at epoch 50 for every encoder (no encoder has CV epochs except AST, and epoch 50 needs no test selection). RUN_ID `lodo-bench`. Benchmark table only; the registered H2 result (CV epochs) stays separate.

### 3.3 BEATs (C3)
`BEATs_iter3_plus_AS2M.pt`, preprocessing as `BEATs.preprocess` (Kaldi fbank of wave · 2^15, 128 bins, 25/10 ms, normalised with 15.41663 and 6.55582), Patch-Mix on the patch embeddings, mean-pooled encoder output. (i) Anchor: cross-entropy, batch 32, lr 5e-5, 100 epochs, best epoch on test (optimistic, as PAFA reports 63.49), seeds 0-4. (ii) Patch-Mix CL with our recipe, best-on-test and epoch 50, seeds 0-4. (iii) LODO baseline at epoch 50, 4 x 5. RUN_ID `beats`.

### 3.4 PAFA arm (C6)
Cross-entropy + PAFA (PCSL λ 5.0, GPAL λ 0.005) on the projector output, AST, batch 32, lr 5e-5, 50 epochs, as in the PAFA AST script. Official split best-on-test (anchor), 5 seeds; LODO at epoch 50, 4 x 5. RUN_ID `lodo-bench` / `pafa_ast`.

### 3.5 Batch 32 (C5)
`baseline` and `sc+sc_static` at batch 32, lr 5e-5 and 1e-4, 5-fold CV, `--stop_epoch 20`. Adopted for a later recipe only if the CV epoch moves later **and** the fold SD drops, for both arms; even then, not before the phantom.

## 4. Analyses

- **Device probe (B2).** Logistic regression on the pooled embeddings (`emb.npz` of the LODO units), patient-grouped 5-fold, subsample balanced on device x class, devices with at least 10 patients. Balanced accuracy and log-loss of device; class probe on the same embeddings. References: chance and a probe fed only the class label.
- **Device-robust threshold (B4).** q_robust = max over source devices with at least 30 calibration cycles of the per-device LAC quantile. Coverage and set size next to split conformal, every LODO arm.
- **Noise stress (S1).** Stationary ESC-50 noise (rain, wind, sea_waves, engine, vacuum_cleaner, washing_machine, airplane, train; impulsive classes excluded) mixed into the test waveform before SC at SNR ∞, 30, 20, 10, 5 dB by cycle power, one fixed seed per cycle. Calibration half clean, evaluation half noisy (E0 split). Score, coverage, set size, and the change of the SC coefficient.
- **SC diagnostics (N3/B3).** `scripts/sc_diagnostics.py`: coefficient curves, sampling noise by number of patients, class-mix drift, between-device distance.

## 5. Conditional items (only if the condition is met)

- Noise-trained arm: only if S1 shows that coverage drops, or that SC makes it worse, at SNR ≥ 10 dB.
- Class-conditional device alignment: only if the device probe on AuscultTrust and `sc_cons` embeddings gives device balanced accuracy clearly above the label-only probe.
- SPRSound as a fifth fold: only if time is left after the 15/11 report.
- Not planned: PCSL with device ids; a combined patient + device loss before 15/11; KAUH filters as extra devices; training on KAUH.
