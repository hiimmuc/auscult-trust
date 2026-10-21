# LODO v1 results: leave-one-device-out, inference arms and trained SC (21/10/2026)

Registration: `docs/updates/prereg-v9-amendment-4.md` (exploratory, no verdict; the registered AST results in
`docs/reports/report_2026-10-09.md` stay the registered outcomes). Run `lodo-v1`; plan `docs/updates/lodo-plan-v1.md`.
Tables: `scripts/lodo_h2_all.sh` (null gate and H2 per cell, `outputs/train/lodo-v1/h2_<cell>.{md,json}`) and
`scripts/lodo_tables.py`. Tags: [Fact], [Interpretation].

## What ran

- No SC, 11 cells x 4 held-out devices x 5 seeds = 220 units (machines A, B, E; finished 21/10 00:17; 0 failed). Fixed
  last epoch of the grid recipe (50; BEATs PAFA 100). Test = every cycle of the held-out device; 20% of each source
  device's patients are calibration patients (same patients for every cell of one seed).
- Inference arms on all 220 units (`src.evaluation.lodo_ttsc`, 21/10): test-time SC with static and with dynamic bound
  (held-out device corrected towards the mean spectrum of the source devices, from its unlabelled cycles), A2 (per-bin
  mean/SD of the held-out spectrograms mapped onto the training statistics), TTA-EQ (mean softmax over 8 random smooth
  per-bin gain views, seeds 1000-1007). 880 arm units. The 20 QLung units failed at first (missing `class_nums` at
  model build in inference) and were rerun after a one-line fix.
- Trained SC with static bound, 7 cells x (official split + 4 folds) x 5 seeds = 175 units (machines C, D, A, B, E;
  finished 21/10 13:25; 0 failed; 9 units cut off by kills on C were rerun from epoch 0).
- Conformal: LAC at 90%, calibrated on the calibration patients. Delta = 0.90 minus prior-matched coverage (positive =
  under-coverage). Null gate: a fold is tested only if the baseline |Delta| exceeds the 95th percentile of |Delta| over
  200 exchangeable re-splits of the calibration patients.

## Summary

1. [Fact] **H2 cannot be tested in LODO v1.** For all 11 cells and all 4 folds the baseline |Delta| stays below the null
   95th percentile (closest: BEATs Patch-Mix CL on Litt3200, 0.090 vs 0.091). At the fixed last epoch, conformal
   coverage on an unseen stethoscope is within chance once the class mix is matched. The registered AST LODO (stopped
   at epoch 2) had under-coverage on Litt3200 (0.670); the converged models do not.
2. [Fact] Coverage stays near nominal in different ways per fold: on Litt3200 most cells over-cover (89-99.5%) with large
   sets (2.6-3.5 of 4 labels: little information); on AKGC417L raw coverage is 74-85% but the prior-matched |Delta| is
   small (0.03-0.06), so the shortfall there is class mix, not device.
3. [Fact] Encoders keep their order under device shift: macro Score over the 4 folds BEATs 58.8-60.1, AST 56.3-57.0,
   HTS-AT 56.1-56.9. Litt3200 is the hardest fold (Score 47.3-59.7); BEATs PAFA is best there (59.7, Sp 88.5%).
4. [Fact] Methods vs CE on the same encoder are within seed SD in LODO (macro -0.3 to +1.0). BEATs PAFA gains on
   Litt3200 (+3.3 ± 2.9) but not in the macro (-0.3).
5. [Fact] **Test-time SC helps on the hardest fold, for every cell:** Litt3200 Score +3.4 to +5.3 (AST), +1.9 to +3.7
   (HTS-AT), +0.6 to +2.4 (BEATs); macro -1.4 to +1.1; coverage and set size almost unchanged. Static and dynamic bound
   are close.
6. [Fact] **Trained SC (static bound) improves LODO Score for 6 of 7 cells** (macro +1.2 to +1.5 for AST PAFA, HTS-AT
   both, BEATs PAFA and SG-SCL; +0.3 AST Patch-Mix CL; -0.0 BEATs Patch-Mix CL) and costs nothing on the official split
   (-0.6 to +0.7 Score). BEATs PAFA + trained SC reaches 62.3 ± 0.9 on the official split at the last epoch (best so
   far; 61.7 without SC) and about 60.3 macro in LODO (61.6 on Litt3200).
7. [Fact] A2 hurts every BEATs cell (macro -4.0 to -6.1; Litt3200 -4.8 to -11.6) and AST PAFA (-2.6), and helps HTS-AT
   on Litt3200 (+5). [Interpretation] Matching whole-device statistics removes the held-out device's class mix along
   with its spectrum.
8. [Fact] TTA-EQ changes Score by 0.0 to +0.5 and shrinks sets by 0.07-0.32 labels, lowering Litt3200 over-coverage by 1-5
   points (towards nominal).

[Interpretation] The device-shift problem that the registered AST study found is mostly absent in converged models at the
level of coverage; SC's value in LODO v1 shows in Score on the hardest device (test-time) and in macro Score (trained),
not in coverage. Coverage-based claims (H2) need either earlier-epoch models, the phantom data or another dataset.

## Null gate (baseline |Delta| / null 95th percentile; * = tested)

| Cell | Meditron | LittC2SE | Litt3200 | AKGC417L |
|---|---|---|---|---|
| ast+ft+ce | 0.022 / 0.118 | 0.039 / 0.120 | 0.050 / 0.092 | 0.027 / 0.090 |
| ast+ft+pafa | 0.032 / 0.100 | 0.062 / 0.125 | 0.059 / 0.104 | 0.041 / 0.097 |
| ast+ft+patchmix_cl | 0.033 / 0.124 | 0.048 / 0.126 | 0.069 / 0.105 | 0.026 / 0.099 |
| htsat+ft+ce | 0.021 / 0.141 | 0.027 / 0.130 | 0.084 / 0.093 | 0.032 / 0.135 |
| htsat+ft+patchmix_cl | 0.022 / 0.122 | 0.034 / 0.130 | 0.058 / 0.102 | 0.057 / 0.115 |
| htsat+ft+sg_scl | 0.026 / 0.123 | 0.030 / 0.110 | 0.035 / 0.090 | 0.059 / 0.127 |
| beats+ft+ce | 0.028 / 0.142 | 0.044 / 0.136 | 0.090 / 0.107 | 0.033 / 0.136 |
| beats+ft+patchmix_cl | 0.041 / 0.143 | 0.044 / 0.157 | 0.090 / 0.091 | 0.027 / 0.105 |
| beats+ft+sg_scl | 0.041 / 0.112 | 0.056 / 0.105 | 0.060 / 0.098 | 0.027 / 0.123 |
| beats+ft+pafa | 0.035 / 0.113 | 0.118 / 0.150 | 0.044 / 0.115 | 0.032 / 0.115 |
| beats+ft+qlung | 0.041 / 0.156 | 0.039 / 0.143 | 0.083 / 0.107 | 0.027 / 0.105 |

## Score and coverage per held-out device (no SC, 5 seeds)

| Cell | Meditron Score | LittC2SE Score | Litt3200 Score | AKGC417L Score | Macro Score | Macro w/o AKGC417L | Meditron Cov / set | LittC2SE Cov / set | Litt3200 Cov / set | AKGC417L Cov / set |
|---|---|---|---|---|---|---|---|---|---|---|
| ast+ft+ce | 56.2 | 65.1 | 50.2 | 53.7 | 56.3 ± 1.7 | 57.2 ± 2.5 | 94.4 / 2.62 | 93.3 / 2.41 | 96.3 / 3.04 | 78.1 / 2.40 |
| ast+ft+pafa | 57.0 | 64.5 | 50.5 | 56.1 | 57.0 ± 1.0 | 57.3 ± 1.2 | 92.0 / 2.21 | 91.5 / 2.22 | 97.6 / 3.22 | 74.5 / 1.92 |
| ast+ft+patchmix_cl | 55.6 | 63.0 | 52.8 | 54.0 | 56.3 ± 1.2 | 57.1 ± 1.9 | 94.4 / 2.59 | 92.6 / 2.49 | 95.3 / 2.97 | 85.2 / 2.81 |
| htsat+ft+ce | 58.1 | 63.0 | 47.3 | 55.9 | 56.1 ± 1.7 | 56.1 ± 1.9 | 93.7 / 2.21 | 92.8 / 2.22 | 88.9 / 2.67 | 77.9 / 2.13 |
| htsat+ft+patchmix_cl | 58.3 | 63.4 | 49.6 | 56.1 | 56.9 ± 2.0 | 57.1 ± 2.2 | 95.0 / 2.32 | 93.7 / 2.26 | 91.2 / 2.60 | 74.1 / 1.90 |
| htsat+ft+sg_scl | 58.2 | 63.7 | 48.8 | 55.7 | 56.6 ± 1.5 | 56.9 ± 1.7 | 93.9 / 2.47 | 93.5 / 2.39 | 89.7 / 2.87 | 74.6 / 1.89 |
| beats+ft+ce | 57.6 | 64.4 | 56.3 | 57.9 | 59.1 ± 1.3 | 59.5 ± 1.8 | 95.1 / 2.74 | 92.6 / 2.34 | 99.5 / 3.54 | 81.1 / 2.35 |
| beats+ft+patchmix_cl | 57.7 | 66.4 | 57.5 | 58.8 | 60.1 ± 1.7 | 60.5 ± 2.6 | 95.2 / 2.67 | 94.6 / 2.50 | 99.4 / 3.54 | 84.0 / 2.36 |
| beats+ft+sg_scl | 58.4 | 64.8 | 54.9 | 57.9 | 59.0 ± 1.7 | 59.4 ± 2.3 | 95.6 / 2.87 | 95.3 / 2.55 | 94.9 / 2.80 | 79.3 / 2.05 |
| beats+ft+pafa | 57.2 | 62.0 | 59.7 | 56.4 | 58.8 ± 1.4 | 59.6 ± 1.6 | 91.8 / 1.94 | 85.7 / 1.95 | 96.6 / 2.76 | 80.9 / 2.26 |
| beats+ft+qlung | 58.5 | 65.1 | 57.2 | 57.8 | 59.6 ± 0.8 | 60.3 ± 0.6 | 94.8 / 2.65 | 92.4 / 2.32 | 98.8 / 3.37 | 85.2 / 2.46 |

## Sp / Se / HS per held-out device (no SC)

| Cell | Meditron Sp / Se / HS | LittC2SE Sp / Se / HS | Litt3200 Sp / Se / HS | AKGC417L Sp / Se / HS |
|---|---|---|---|---|
| ast+ft+ce | 79.6 / 32.8 / 46.1 | 94.0 / 36.2 / 52.2 | 49.0 / 51.5 / 47.1 | 88.1 / 19.4 / 31.7 |
| ast+ft+pafa | 81.8 / 32.1 / 45.9 | 91.5 / 37.5 / 53.0 | 51.0 / 50.1 / 49.0 | 87.6 / 24.7 / 38.5 |
| ast+ft+patchmix_cl | 79.7 / 31.6 / 45.2 | 94.7 / 31.3 / 46.7 | 53.6 / 52.1 / 51.1 | 86.3 / 21.6 / 34.4 |
| htsat+ft+ce | 84.4 / 31.8 / 46.0 | 90.0 / 36.0 / 51.3 | 33.0 / 61.6 / 40.8 | 85.4 / 26.5 / 40.1 |
| htsat+ft+patchmix_cl | 81.3 / 35.3 / 48.9 | 89.6 / 37.3 / 52.5 | 39.6 / 59.6 / 44.5 | 87.3 / 25.0 / 38.6 |
| htsat+ft+sg_scl | 82.7 / 33.8 / 47.6 | 90.9 / 36.5 / 51.8 | 41.6 / 56.0 / 44.6 | 87.8 / 23.5 / 37.0 |
| beats+ft+ce | 78.7 / 36.5 / 49.8 | 91.3 / 37.6 / 53.1 | 59.5 / 53.2 / 54.9 | 83.4 / 32.5 / 46.7 |
| beats+ft+patchmix_cl | 78.9 / 36.4 / 49.7 | 90.4 / 42.5 / 57.7 | 56.8 / 58.3 / 53.4 | 79.9 / 37.8 / 51.0 |
| beats+ft+sg_scl | 84.9 / 32.0 / 45.9 | 92.2 / 37.4 / 53.2 | 60.6 / 49.3 / 51.2 | 85.3 / 30.5 / 44.8 |
| beats+ft+pafa | 85.6 / 28.7 / 43.0 | 90.8 / 33.2 / 48.5 | 88.5 / 30.9 / 44.5 | 84.7 / 28.0 / 41.8 |
| beats+ft+qlung | 82.5 / 34.5 / 48.4 | 92.0 / 38.3 / 53.9 | 63.1 / 51.3 / 54.6 | 78.2 / 37.4 / 49.8 |

## Inference arms and trained SC vs the source cell (paired by seed)

| Cell | Arm | dScore macro | dScore Litt3200 | dCov Litt3200 | d|Delta| macro | dSet size macro | Set size Litt3200 (arm) |
|---|---|---|---|---|---|---|---|
| ast+ft+ce | TT-SC static | +0.4 ± 0.9 | +3.5 ± 2.0 | +0.5 | +0.001 | -0.01 | 3.02 |
| ast+ft+ce | TT-SC dynamic | +0.0 ± 0.4 | +4.1 ± 1.7 | +1.1 | +0.002 | -0.01 | 3.03 |
| ast+ft+ce | A2 | -1.1 ± 1.5 | -0.3 ± 4.2 | +1.4 | +0.006 | +0.21 | 3.51 |
| ast+ft+ce | TTA-EQ | +0.4 ± 0.4 | +0.6 ± 1.2 | -4.6 | +0.005 | -0.19 | 2.69 |
| ast+ft+pafa | TT-SC static | +0.7 ± 0.8 | +3.4 ± 1.0 | +0.1 | +0.003 | +0.09 | 3.23 |
| ast+ft+pafa | TT-SC dynamic | +0.1 ± 0.7 | +4.4 ± 1.5 | +0.5 | -0.003 | +0.19 | 3.29 |
| ast+ft+pafa | A2 | -2.6 ± 1.9 | -8.8 ± 7.8 | -0.3 | -0.000 | +0.35 | 3.64 |
| ast+ft+pafa | TTA-EQ | +0.5 ± 0.2 | +0.5 ± 0.7 | -3.8 | -0.010 | -0.18 | 2.79 |
| ast+ft+pafa | trained SC static | +1.4 ± 0.7 | +2.8 ± 2.7 | -0.5 | +0.007 | -0.17 | 3.03 |
| ast+ft+patchmix_cl | TT-SC static | +0.8 ± 0.8 | +4.6 ± 3.3 | +1.6 | -0.005 | -0.05 | 3.00 |
| ast+ft+patchmix_cl | TT-SC dynamic | +1.1 ± 0.4 | +5.3 ± 1.7 | +2.0 | -0.003 | -0.02 | 2.97 |
| ast+ft+patchmix_cl | A2 | -0.4 ± 1.1 | -0.6 ± 4.1 | -0.0 | -0.000 | +0.16 | 3.34 |
| ast+ft+patchmix_cl | TTA-EQ | +0.2 ± 0.3 | +0.0 ± 0.7 | -3.0 | +0.004 | -0.30 | 2.54 |
| ast+ft+patchmix_cl | trained SC static | +0.3 ± 1.5 | +2.7 ± 3.5 | +1.6 | -0.003 | -0.29 | 2.76 |
| htsat+ft+ce | TT-SC static | +0.8 ± 1.0 | +2.1 ± 2.0 | +1.5 | -0.007 | +0.00 | 2.65 |
| htsat+ft+ce | TT-SC dynamic | +0.0 ± 1.1 | +1.9 ± 2.3 | +2.9 | -0.006 | -0.00 | 2.68 |
| htsat+ft+ce | A2 | +0.8 ± 1.3 | +5.2 ± 3.0 | +5.2 | -0.012 | +0.07 | 2.76 |
| htsat+ft+ce | TTA-EQ | +0.3 ± 0.3 | -0.0 ± 1.0 | -3.0 | +0.009 | -0.07 | 2.50 |
| htsat+ft+patchmix_cl | TT-SC static | +1.1 ± 1.3 | +2.6 ± 3.9 | +2.5 | -0.002 | -0.06 | 2.62 |
| htsat+ft+patchmix_cl | TT-SC dynamic | +0.8 ± 0.9 | +3.7 ± 3.9 | +2.5 | -0.001 | -0.05 | 2.59 |
| htsat+ft+patchmix_cl | A2 | +0.7 ± 1.3 | +5.2 ± 4.6 | +3.3 | -0.009 | +0.01 | 2.58 |
| htsat+ft+patchmix_cl | TTA-EQ | +0.3 ± 0.6 | +0.0 ± 2.1 | -3.6 | +0.006 | -0.11 | 2.38 |
| htsat+ft+patchmix_cl | trained SC static | +1.2 ± 1.5 | -0.4 ± 3.2 | +0.6 | -0.007 | -0.12 | 2.57 |
| htsat+ft+sg_scl | TT-SC static | +1.0 ± 0.8 | +2.8 ± 2.3 | +1.0 | +0.000 | -0.05 | 2.78 |
| htsat+ft+sg_scl | TT-SC dynamic | +0.7 ± 0.7 | +3.6 ± 1.0 | +1.7 | +0.001 | -0.05 | 2.84 |
| htsat+ft+sg_scl | A2 | +0.7 ± 1.1 | +5.0 ± 4.1 | +3.7 | -0.004 | -0.07 | 2.73 |
| htsat+ft+sg_scl | TTA-EQ | +0.1 ± 0.4 | -0.4 ± 1.5 | -2.7 | +0.008 | -0.12 | 2.62 |
| htsat+ft+sg_scl | trained SC static | +1.5 ± 1.4 | +1.5 ± 3.8 | -1.4 | +0.013 | -0.11 | 2.62 |
| beats+ft+ce | TT-SC static | -0.1 ± 0.9 | +1.0 ± 1.2 | +0.2 | -0.001 | +0.12 | 3.62 |
| beats+ft+ce | TT-SC dynamic | -0.3 ± 0.5 | +1.1 ± 1.3 | +0.0 | -0.001 | +0.04 | 3.53 |
| beats+ft+ce | A2 | -4.6 ± 1.6 | -6.9 ± 4.7 | -0.8 | -0.003 | +0.18 | 3.64 |
| beats+ft+ce | TTA-EQ | +0.3 ± 0.1 | -0.2 ± 0.4 | -1.2 | -0.012 | -0.24 | 3.10 |
| beats+ft+patchmix_cl | TT-SC static | +0.1 ± 0.6 | +1.1 ± 1.2 | +0.4 | -0.001 | +0.04 | 3.72 |
| beats+ft+patchmix_cl | TT-SC dynamic | -0.7 ± 0.6 | +1.0 ± 0.8 | +0.4 | +0.004 | -0.03 | 3.69 |
| beats+ft+patchmix_cl | A2 | -4.4 ± 2.1 | -7.4 ± 5.0 | -0.0 | +0.004 | +0.13 | 3.75 |
| beats+ft+patchmix_cl | TTA-EQ | +0.3 ± 0.4 | +0.3 ± 1.0 | -2.4 | -0.011 | -0.30 | 3.07 |
| beats+ft+patchmix_cl | trained SC static | -0.0 ± 1.4 | -0.3 ± 3.8 | +0.2 | -0.003 | -0.00 | 3.75 |
| beats+ft+sg_scl | TT-SC static | -0.2 ± 1.0 | +0.6 ± 1.3 | +0.1 | +0.005 | -0.02 | 2.84 |
| beats+ft+sg_scl | TT-SC dynamic | -0.2 ± 0.6 | +1.2 ± 2.1 | +0.1 | -0.002 | -0.05 | 2.79 |
| beats+ft+sg_scl | A2 | -4.0 ± 1.7 | -4.8 ± 3.7 | -2.7 | -0.008 | -0.00 | 2.86 |
| beats+ft+sg_scl | TTA-EQ | +0.3 ± 0.3 | +0.3 ± 0.6 | -1.2 | -0.005 | -0.32 | 2.51 |
| beats+ft+sg_scl | trained SC static | +1.5 ± 1.1 | +2.9 ± 4.7 | +1.7 | -0.002 | -0.07 | 2.66 |
| beats+ft+pafa | TT-SC static | -0.3 ± 0.5 | +2.4 ± 0.9 | +0.2 | -0.011 | +0.07 | 2.82 |
| beats+ft+pafa | TT-SC dynamic | -0.8 ± 0.4 | +2.2 ± 1.3 | +0.2 | -0.017 | +0.11 | 2.78 |
| beats+ft+pafa | A2 | -6.1 ± 1.7 | -11.6 ± 5.1 | -0.2 | -0.019 | +0.29 | 3.15 |
| beats+ft+pafa | TTA-EQ | +0.2 ± 0.2 | -0.0 ± 0.3 | -1.1 | -0.012 | -0.10 | 2.49 |
| beats+ft+pafa | trained SC static | +1.5 ± 1.2 | +1.9 ± 3.3 | +0.9 | +0.012 | -0.10 | 3.01 |
| beats+ft+qlung | TT-SC static | -0.8 ± 1.4 | +0.7 ± 3.0 | +0.3 | -0.004 | +0.11 | 3.41 |
| beats+ft+qlung | TT-SC dynamic | -1.4 ± 1.1 | +0.7 ± 3.5 | +0.3 | +0.000 | +0.02 | 3.39 |
| beats+ft+qlung | A2 | -5.1 ± 1.3 | -7.0 ± 5.3 | -0.7 | -0.001 | +0.22 | 3.60 |
| beats+ft+qlung | TTA-EQ | +0.0 ± 0.3 | +0.2 ± 0.8 | -2.2 | -0.015 | -0.28 | 2.92 |

## Method vs CE on the same encoder (no SC)

| Encoder | Method | dScore macro | dScore w/o AKGC417L | dScore Litt3200 |
|---|---|---|---|---|
| ast | pafa | +0.7 ± 2.3 | +0.2 ± 3.2 | +0.3 ± 7.5 |
| ast | patchmix_cl | +0.0 ± 1.7 | -0.0 ± 1.7 | +2.6 ± 4.4 |
| htsat | patchmix_cl | +0.8 ± 1.5 | +1.0 ± 1.9 | +2.3 ± 4.4 |
| htsat | sg_scl | +0.5 ± 1.4 | +0.8 ± 1.8 | +1.5 ± 3.4 |
| beats | patchmix_cl | +1.0 ± 1.4 | +1.1 ± 1.8 | +1.2 ± 3.8 |
| beats | sg_scl | -0.1 ± 1.5 | -0.1 ± 2.0 | -1.4 ± 4.4 |
| beats | pafa | -0.3 ± 1.5 | +0.1 ± 1.3 | +3.3 ± 2.9 |
| beats | qlung | +0.6 ± 0.9 | +0.8 ± 1.5 | +0.8 ± 6.9 |

## Trained SC (static) on the official split vs no SC (same cell and seed, last epoch)

| Cell | Sp | Se | Score | HS | 2c Score | Worst-dev Score | Cov | Worst-dev Cov | Set size | dScore vs no SC | dWorst-dev Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ast+ft+patchmix_cl | 70.8 ± 2.0 | 45.8 ± 1.3 | 58.3 ± 1.1 | 55.6 ± 1.0 | 66.1 ± 0.8 | 49.8 ± 3.0 | 89.4 ± 0.1 | 87.5 ± 0.3 | 2.25 ± 0.06 | +0.1 ± 1.3 | +2.8 ± 5.0 |
| ast+ft+pafa | 71.7 ± 2.3 | 46.4 ± 1.6 | 59.0 ± 0.7 | 56.3 ± 0.8 | 67.0 ± 0.6 | 50.3 ± 2.0 | 89.5 ± 0.2 | 85.5 ± 0.6 | 2.34 ± 0.02 | +0.7 ± 0.5 | +2.9 ± 2.5 |
| htsat+ft+patchmix_cl | 68.8 ± 3.3 | 48.1 ± 1.0 | 58.5 ± 1.2 | 56.6 ± 0.6 | 65.9 ± 0.8 | 46.3 ± 3.5 | 90.1 ± 0.3 | 87.3 ± 0.5 | 2.24 ± 0.08 | -0.6 ± 1.4 | -0.2 ± 6.1 |
| htsat+ft+sg_scl | 73.0 ± 3.0 | 45.7 ± 1.6 | 59.3 ± 1.0 | 56.2 ± 0.7 | 66.5 ± 0.9 | 49.7 ± 4.3 | 90.2 ± 0.3 | 87.7 ± 1.1 | 2.34 ± 0.03 | -0.3 ± 1.4 | -0.1 ± 5.8 |
| beats+ft+patchmix_cl | 67.4 ± 1.6 | 52.2 ± 1.3 | 59.8 ± 0.8 | 58.8 ± 0.8 | 68.3 ± 0.7 | 52.0 ± 2.6 | 89.2 ± 0.2 | 86.2 ± 0.2 | 2.30 ± 0.06 | +0.1 ± 1.2 | -0.1 ± 4.0 |
| beats+ft+pafa | 75.6 ± 3.1 | 48.9 ± 2.9 | 62.3 ± 0.9 | 59.3 ± 1.3 | 69.5 ± 0.7 | 58.9 ± 1.4 | 89.6 ± 0.3 | 85.6 ± 0.7 | 2.33 ± 0.13 | +0.5 ± 1.5 | -0.1 ± 1.2 |
| beats+ft+sg_scl | 72.0 ± 2.9 | 49.4 ± 1.1 | 60.7 ± 1.0 | 58.6 ± 0.6 | 68.5 ± 0.8 | 51.8 ± 5.5 | 89.8 ± 0.2 | 87.5 ± 0.2 | 2.43 ± 0.16 | -0.4 ± 1.6 | -1.1 ± 6.8 |

Notes: Cov in %, set = mean labels per set (of 4). Macro = mean over the 4 folds per seed, then mean ± SD over seeds.
In the arms table d = arm minus source cell on the same seed (same calibration and test patients); `d|Delta|` = change of
|Delta| (negative = closer to nominal); "Set size Litt3200 (arm)" is the arm's absolute set size.
