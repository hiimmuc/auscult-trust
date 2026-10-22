# Ensembles by encoder group, equal and fitted weights (22/10/2026)

Regenerate: `PYTHONPATH=. .venv-train/bin/python scripts/ensemble.py` (CPU, about 10 min); check `tests/test_ensemble.py`.

## 1. Setup

- [Fact] Groups defined by the user on 22/10 (not the registered E1-E4 of Amendment 4, which use all 11 LODO cells, the
  5 BEATs, 3 AST and 3 HTS-AT cells, equal weights only):
  E1 = BEATs PAFA, SG-SCL, Patch-Mix CL; E2 = AST Patch-Mix CL, PAFA; E3 = HTS-AT Patch-Mix CL, SG-SCL; cross-group
  E1+E2, E1+E3, E2+E3, E1+E2+E3. All members fine-tuned, last epoch; members of one ensemble share seed and fold (cycle
  order checked).
- [Fact] Ensemble = weighted mean of the member softmaxes. Two weightings:
  - `equal`: 1/K per member.
  - `fit` (trainable): member weights on the simplex (softmax of K free parameters, L-BFGS) minimising the NLL of the
    calibration patients of each LODO unit (20% of the source-device patients, never used in training; no test cycle).
    One weight vector per fold and seed.
  - `fit*` (official split only): the official split has no labelled held-out set, so it uses the mean of the LODO-fitted
    weights of the same members (all folds and seeds). No weight is ever fitted on test cycles.
- [Fact] Arms: without SC (grid-v1 units on the official split, lodo-v1 no-SC units) and with trained SC static (the
  same 7 cells exist with `+sc+sc_static`). Reference = BEATs PAFA alone in the same arm (the chosen AuscultTrust),
  paired by seed. The reference rows reproduce the reported numbers (61.7, 58.8, 62.3, 60.3).
- [Fact] Coverage: official split = LAC over 20 device-stratified halves of the test patients; LODO = LAC calibrated on
  the source-device calibration patients, abs(Delta) = |0.90 - prior-matched coverage| on the held-out device (no fold
  passes the null gate, so no H2 reading).

## 2. Results

### Official split, no SC (last epoch, 5 seeds; d = paired vs BEATs PAFA no SC)

| Ensemble | Sp | Se | Score | HS | Worst-dev Score | Cov | Worst-dev Cov | Set size | dScore | dWorst-dev Score |
|---|---|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (reference) | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| E1 BEATs (equal) | 72.4 ± 2.7 | 52.7 ± 1.8 | 62.5 ± 0.5 | 60.9 ± 0.3 | 57.9 ± 2.8 | 89.3 ± 0.3 | 86.3 ± 0.5 | 2.18 ± 0.06 | +0.8 ± 0.5 | -1.0 ± 3.0 |
| E1 BEATs (fit*) | 72.0 ± 3.3 | 52.3 ± 1.6 | 62.1 ± 0.9 | 60.5 ± 0.3 | 56.1 ± 4.4 | 89.4 ± 0.2 | 86.3 ± 0.4 | 2.18 ± 0.06 | +0.4 ± 0.8 | -2.8 ± 4.6 |
| E2 AST (equal) | 72.6 ± 1.8 | 47.2 ± 1.3 | 59.9 ± 0.4 | 57.2 ± 0.5 | 49.7 ± 1.9 | 89.4 ± 0.2 | 86.3 ± 0.5 | 2.14 ± 0.04 | -1.8 ± 0.9 | -9.2 ± 1.8 |
| E2 AST (fit*) | 72.8 ± 3.1 | 46.4 ± 1.5 | 59.6 ± 1.2 | 56.6 ± 0.9 | 48.8 ± 3.4 | 89.4 ± 0.3 | 86.3 ± 0.4 | 2.14 ± 0.04 | -2.1 ± 1.7 | -10.2 ± 3.0 |
| E3 HTS-AT (equal) | 72.1 ± 2.8 | 48.0 ± 0.9 | 60.1 ± 1.0 | 57.6 ± 0.4 | 48.3 ± 4.1 | 89.8 ± 0.2 | 87.8 ± 0.2 | 2.18 ± 0.03 | -1.7 ± 1.0 | -10.6 ± 4.1 |
| E3 HTS-AT (fit*) | 71.8 ± 3.1 | 48.1 ± 0.9 | 60.0 ± 1.2 | 57.6 ± 0.5 | 48.0 ± 4.5 | 89.8 ± 0.1 | 87.8 ± 0.2 | 2.18 ± 0.03 | -1.7 ± 1.1 | -11.0 ± 4.4 |
| E1+E2 (equal) | 73.6 ± 1.0 | 50.9 ± 1.1 | 62.3 ± 0.6 | 60.2 ± 0.7 | 56.6 ± 0.9 | 89.4 ± 0.1 | 86.0 ± 0.1 | 1.92 ± 0.01 | +0.5 ± 1.2 | -2.4 ± 1.1 |
| E1+E2 (fit*) | 73.6 ± 1.4 | 50.9 ± 1.2 | 62.3 ± 0.8 | 60.2 ± 0.8 | 55.0 ± 2.4 | 89.4 ± 0.1 | 86.0 ± 0.1 | 1.92 ± 0.02 | +0.5 ± 1.3 | -4.0 ± 2.5 |
| E1+E3 (equal) | 74.0 ± 2.0 | 52.7 ± 1.7 | 63.4 ± 0.3 | 61.6 ± 0.5 | 57.2 ± 3.5 | 89.5 ± 0.1 | 86.6 ± 0.6 | 1.90 ± 0.01 | +1.7 ± 0.5 | -1.8 ± 3.7 |
| E1+E3 (fit*) | 73.7 ± 2.6 | 51.2 ± 1.5 | 62.5 ± 0.8 | 60.4 ± 0.6 | 53.5 ± 3.5 | 89.5 ± 0.1 | 86.7 ± 0.6 | 1.91 ± 0.02 | +0.8 ± 0.8 | -5.5 ± 3.6 |
| E2+E3 (equal) | 73.4 ± 1.9 | 48.3 ± 1.1 | 60.8 ± 0.9 | 58.2 ± 0.8 | 48.5 ± 3.3 | 89.5 ± 0.1 | 86.5 ± 0.4 | 1.98 ± 0.02 | -0.9 ± 1.0 | -10.5 ± 3.3 |
| E2+E3 (fit*) | 73.5 ± 2.0 | 48.6 ± 0.8 | 61.0 ± 0.9 | 58.5 ± 0.7 | 48.9 ± 4.2 | 89.6 ± 0.1 | 86.6 ± 0.4 | 1.99 ± 0.02 | -0.7 ± 1.0 | -10.0 ± 4.2 |
| E1+E2+E3 (equal) | 75.1 ± 0.9 | 51.4 ± 0.6 | 63.2 ± 0.3 | 61.0 ± 0.3 | 56.5 ± 1.4 | 89.5 ± 0.2 | 86.2 ± 0.2 | 1.87 ± 0.01 | +1.5 ± 0.4 | -2.4 ± 1.7 |
| E1+E2+E3 (fit*) | 74.5 ± 1.8 | 51.1 ± 0.8 | 62.8 ± 0.8 | 60.6 ± 0.7 | 54.0 ± 2.3 | 89.5 ± 0.1 | 86.3 ± 0.1 | 1.88 ± 0.01 | +1.1 ± 0.6 | -4.9 ± 2.5 |

### LODO v1, no SC (Score per held-out device, mean of 5 seeds; macro = mean over the 4 folds per seed)

| Ensemble | Meditron | LittC2SE | Litt3200 | AKGC417L | Macro Score | dMacro vs ref | dLitt3200 vs ref | Litt3200 Sp / Se | Macro abs(Delta) | Macro set size |
|---|---|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (reference) | 57.2 | 62.0 | 59.7 | 56.4 | 58.8 ± 1.4 | +0.0 ± 0.0 | +0.0 ± 0.0 | 88.5 / 30.9 | 0.057 | 2.23 |
| E1 BEATs (equal) | 59.2 | 66.0 | 58.2 | 58.7 | 60.5 ± 1.5 | +1.7 ± 0.9 | -1.5 ± 3.9 | 68.5 / 47.8 | 0.042 | 2.43 |
| E1 BEATs (fit) | 59.1 | 65.6 | 57.5 | 58.7 | 60.2 ± 1.5 | +1.4 ± 1.0 | -2.2 ± 3.7 | 64.2 / 50.7 | 0.038 | 2.40 |
| E2 AST (equal) | 58.2 | 65.3 | 52.5 | 56.4 | 58.1 ± 0.7 | -0.7 ± 1.3 | -7.2 ± 3.1 | 52.7 / 52.2 | 0.046 | 2.48 |
| E2 AST (fit) | 57.5 | 64.1 | 52.8 | 56.2 | 57.7 ± 1.1 | -1.1 ± 0.7 | -6.9 ± 3.5 | 53.1 / 52.5 | 0.047 | 2.49 |
| E3 HTS-AT (equal) | 58.5 | 64.1 | 50.0 | 56.2 | 57.2 ± 1.8 | -1.6 ± 2.4 | -9.7 ± 7.2 | 40.5 / 59.5 | 0.036 | 2.26 |
| E3 HTS-AT (fit) | 58.5 | 63.6 | 50.3 | 56.3 | 57.1 ± 2.0 | -1.7 ± 2.6 | -9.4 ± 7.4 | 41.0 / 59.5 | 0.037 | 2.26 |
| E1+E2 (equal) | 60.2 | 65.7 | 57.8 | 58.5 | 60.5 ± 1.5 | +1.7 ± 0.9 | -1.9 ± 3.4 | 64.5 / 51.0 | 0.045 | 2.24 |
| E1+E2 (fit) | 59.8 | 66.0 | 57.1 | 58.8 | 60.4 ± 1.5 | +1.6 ± 0.9 | -2.6 ± 3.4 | 61.0 / 53.2 | 0.047 | 2.22 |
| E1+E3 (equal) | 60.3 | 66.1 | 58.6 | 58.7 | 60.9 ± 1.5 | +2.1 ± 1.5 | -1.1 ± 3.6 | 62.2 / 55.0 | 0.037 | 2.16 |
| E1+E3 (fit) | 60.1 | 65.2 | 54.4 | 57.7 | 59.4 ± 1.2 | +0.6 ± 1.7 | -5.3 ± 5.2 | 50.2 / 58.5 | 0.036 | 2.14 |
| E2+E3 (equal) | 60.2 | 66.0 | 51.9 | 57.0 | 58.8 ± 1.6 | -0.0 ± 1.9 | -7.8 ± 5.7 | 46.7 / 57.1 | 0.037 | 2.14 |
| E2+E3 (fit) | 59.5 | 65.0 | 51.6 | 56.6 | 58.2 ± 1.9 | -0.6 ± 2.3 | -8.1 ± 7.1 | 44.7 / 58.4 | 0.036 | 2.12 |
| E1+E2+E3 (equal) | 60.2 | 66.4 | 57.8 | 58.3 | 60.7 ± 1.4 | +1.9 ± 1.0 | -1.9 ± 3.4 | 59.5 / 56.0 | 0.034 | 2.09 |
| E1+E2+E3 (fit) | 60.3 | 66.7 | 55.6 | 57.8 | 60.1 ± 1.3 | +1.3 ± 1.2 | -4.1 ± 4.1 | 53.0 / 58.1 | 0.037 | 2.05 |

### Fitted member weights, no SC (mean ± SD over 4 folds x 5 seeds)

| Ensemble | Weights |
|---|---|
| E1 BEATs | beats+ft+pafa 0.21 ± 0.09, beats+ft+sg_scl 0.46 ± 0.09, beats+ft+patchmix_cl 0.33 ± 0.08 |
| E2 AST | ast+ft+patchmix_cl 0.57 ± 0.10, ast+ft+pafa 0.43 ± 0.10 |
| E3 HTS-AT | htsat+ft+patchmix_cl 0.53 ± 0.09, htsat+ft+sg_scl 0.47 ± 0.09 |
| E1+E2 | beats+ft+pafa 0.11 ± 0.04, beats+ft+sg_scl 0.30 ± 0.10, beats+ft+patchmix_cl 0.18 ± 0.08, ast+ft+patchmix_cl 0.24 ± 0.09, ast+ft+pafa 0.17 ± 0.06 |
| E1+E3 | beats+ft+pafa 0.11 ± 0.05, beats+ft+sg_scl 0.19 ± 0.07, beats+ft+patchmix_cl 0.17 ± 0.06, htsat+ft+patchmix_cl 0.29 ± 0.07, htsat+ft+sg_scl 0.24 ± 0.09 |
| E2+E3 | ast+ft+patchmix_cl 0.22 ± 0.08, ast+ft+pafa 0.18 ± 0.06, htsat+ft+patchmix_cl 0.30 ± 0.08, htsat+ft+sg_scl 0.30 ± 0.09 |
| E1+E2+E3 | beats+ft+pafa 0.08 ± 0.03, beats+ft+sg_scl 0.15 ± 0.08, beats+ft+patchmix_cl 0.12 ± 0.06, ast+ft+patchmix_cl 0.14 ± 0.07, ast+ft+pafa 0.11 ± 0.05, htsat+ft+patchmix_cl 0.22 ± 0.06, htsat+ft+sg_scl 0.20 ± 0.08 |

### Official split, trained SC static (last epoch, 5 seeds; d = paired vs BEATs PAFA trained SC static)

| Ensemble | Sp | Se | Score | HS | Worst-dev Score | Cov | Worst-dev Cov | Set size | dScore | dWorst-dev Score |
|---|---|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (reference) | 75.6 ± 3.1 | 48.9 ± 2.9 | 62.3 ± 0.9 | 59.3 ± 1.3 | 58.9 ± 1.4 | 89.6 ± 0.3 | 85.6 ± 0.7 | 2.33 ± 0.13 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| E1 BEATs (equal) | 74.6 ± 1.1 | 51.7 ± 0.7 | 63.1 ± 0.6 | 61.1 ± 0.5 | 58.0 ± 2.5 | 89.2 ± 0.2 | 86.4 ± 0.3 | 2.06 ± 0.06 | +0.9 ± 1.1 | -0.9 ± 3.5 |
| E1 BEATs (fit*) | 73.3 ± 1.7 | 51.4 ± 1.0 | 62.3 ± 0.7 | 60.4 ± 0.6 | 55.4 ± 3.2 | 89.2 ± 0.2 | 86.4 ± 0.3 | 2.06 ± 0.06 | +0.1 ± 1.3 | -3.4 ± 4.0 |
| E2 AST (equal) | 73.6 ± 1.6 | 46.7 ± 0.5 | 60.2 ± 0.7 | 57.2 ± 0.4 | 52.2 ± 2.7 | 89.5 ± 0.1 | 86.9 ± 1.0 | 2.11 ± 0.03 | -2.1 ± 0.4 | -6.6 ± 2.8 |
| E2 AST (fit*) | 72.8 ± 1.6 | 46.2 ± 1.0 | 59.5 ± 0.6 | 56.5 ± 0.6 | 51.5 ± 3.3 | 89.5 ± 0.1 | 86.9 ± 0.9 | 2.11 ± 0.03 | -2.8 ± 0.8 | -7.4 ± 3.7 |
| E3 HTS-AT (equal) | 71.3 ± 2.2 | 47.8 ± 1.0 | 59.5 ± 0.8 | 57.2 ± 0.5 | 47.9 ± 2.0 | 90.0 ± 0.2 | 87.5 ± 0.4 | 2.20 ± 0.02 | -2.7 ± 1.4 | -11.0 ± 2.4 |
| E3 HTS-AT (fit*) | 71.1 ± 2.3 | 47.9 ± 1.0 | 59.5 ± 0.7 | 57.2 ± 0.4 | 48.0 ± 1.8 | 90.0 ± 0.2 | 87.5 ± 0.4 | 2.20 ± 0.02 | -2.8 ± 1.4 | -10.9 ± 2.1 |
| E1+E2 (equal) | 75.6 ± 1.3 | 49.9 ± 1.2 | 62.7 ± 0.3 | 60.1 ± 0.5 | 57.6 ± 2.1 | 89.4 ± 0.1 | 86.9 ± 0.4 | 1.88 ± 0.01 | +0.5 ± 1.0 | -1.3 ± 3.1 |
| E1+E2 (fit*) | 75.1 ± 1.2 | 49.4 ± 1.1 | 62.3 ± 0.4 | 59.6 ± 0.5 | 55.7 ± 2.2 | 89.4 ± 0.2 | 86.9 ± 0.4 | 1.88 ± 0.01 | -0.0 ± 1.0 | -3.1 ± 3.0 |
| E1+E3 (equal) | 75.9 ± 1.4 | 50.5 ± 0.8 | 63.2 ± 0.7 | 60.6 ± 0.7 | 57.7 ± 3.4 | 89.4 ± 0.2 | 86.5 ± 0.3 | 1.89 ± 0.01 | +0.9 ± 1.4 | -1.2 ± 4.2 |
| E1+E3 (fit*) | 75.0 ± 1.3 | 50.2 ± 0.9 | 62.6 ± 0.7 | 60.1 ± 0.6 | 54.5 ± 3.6 | 89.4 ± 0.1 | 86.4 ± 0.4 | 1.89 ± 0.02 | +0.3 ± 1.4 | -4.3 ± 4.0 |
| E2+E3 (equal) | 73.8 ± 1.2 | 47.9 ± 0.6 | 60.9 ± 0.4 | 58.1 ± 0.3 | 49.8 ± 2.3 | 89.6 ± 0.2 | 87.0 ± 0.5 | 1.96 ± 0.01 | -1.4 ± 0.7 | -9.1 ± 2.2 |
| E2+E3 (fit*) | 73.4 ± 1.6 | 48.0 ± 0.6 | 60.7 ± 0.6 | 58.0 ± 0.3 | 48.7 ± 2.5 | 89.7 ± 0.2 | 86.9 ± 0.5 | 1.96 ± 0.01 | -1.6 ± 1.0 | -10.2 ± 2.4 |
| E1+E2+E3 (equal) | 76.2 ± 1.3 | 49.8 ± 1.2 | 63.0 ± 0.7 | 60.2 ± 0.8 | 55.9 ± 2.7 | 89.6 ± 0.2 | 86.6 ± 0.2 | 1.85 ± 0.02 | +0.7 ± 1.1 | -3.0 ± 3.3 |
| E1+E2+E3 (fit*) | 75.3 ± 1.0 | 49.0 ± 0.6 | 62.2 ± 0.5 | 59.4 ± 0.5 | 53.6 ± 2.3 | 89.6 ± 0.1 | 86.6 ± 0.2 | 1.85 ± 0.02 | -0.1 ± 1.1 | -5.3 ± 2.8 |

### LODO v1, trained SC static (Score per held-out device, mean of 5 seeds; macro = mean over the 4 folds per seed)

| Ensemble | Meditron | LittC2SE | Litt3200 | AKGC417L | Macro Score | dMacro vs ref | dLitt3200 vs ref | Litt3200 Sp / Se | Macro abs(Delta) | Macro set size |
|---|---|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (reference) | 57.6 | 67.5 | 61.5 | 54.6 | 60.3 ± 0.8 | +0.0 ± 0.0 | +0.0 ± 0.0 | 88.2 / 34.8 | 0.069 | 2.13 |
| E1 BEATs (equal) | 59.2 | 70.2 | 61.0 | 57.3 | 61.9 ± 1.1 | +1.6 ± 1.1 | -0.6 ± 5.1 | 72.1 / 49.8 | 0.049 | 2.44 |
| E1 BEATs (fit) | 58.9 | 69.9 | 59.2 | 57.8 | 61.5 ± 1.3 | +1.2 ± 1.4 | -2.3 ± 5.2 | 70.2 / 48.2 | 0.048 | 2.43 |
| E2 AST (equal) | 55.3 | 68.4 | 55.8 | 55.2 | 58.7 ± 0.9 | -1.6 ± 1.2 | -5.7 ± 6.1 | 65.6 / 46.1 | 0.051 | 2.26 |
| E2 AST (fit) | 53.8 | 67.5 | 55.8 | 55.1 | 58.0 ± 0.7 | -2.3 ± 0.9 | -5.8 ± 6.6 | 67.2 / 44.4 | 0.052 | 2.25 |
| E3 HTS-AT (equal) | 60.2 | 67.9 | 50.1 | 56.5 | 58.7 ± 0.9 | -1.6 ± 1.6 | -11.5 ± 9.6 | 39.6 / 60.5 | 0.040 | 2.14 |
| E3 HTS-AT (fit) | 60.0 | 67.1 | 49.9 | 56.4 | 58.4 ± 1.0 | -1.9 ± 1.8 | -11.6 ± 9.7 | 39.7 / 60.1 | 0.042 | 2.14 |
| E1+E2 (equal) | 59.5 | 70.1 | 60.4 | 56.3 | 61.6 ± 1.1 | +1.3 ± 0.9 | -1.2 ± 3.2 | 72.4 / 48.3 | 0.045 | 2.20 |
| E1+E2 (fit) | 58.1 | 70.2 | 60.5 | 57.3 | 61.5 ± 1.3 | +1.2 ± 1.3 | -1.1 ± 5.3 | 72.8 / 48.1 | 0.045 | 2.18 |
| E1+E3 (equal) | 61.6 | 70.6 | 58.5 | 57.4 | 62.0 ± 0.7 | +1.7 ± 1.1 | -3.0 ± 5.0 | 62.5 / 54.6 | 0.046 | 2.13 |
| E1+E3 (fit) | 61.7 | 69.9 | 53.9 | 57.2 | 60.7 ± 0.7 | +0.4 ± 1.5 | -7.6 ± 7.6 | 50.1 / 57.7 | 0.047 | 2.09 |
| E2+E3 (equal) | 60.5 | 69.0 | 54.2 | 55.9 | 59.9 ± 0.8 | -0.4 ± 1.4 | -7.4 ± 8.6 | 53.2 / 55.1 | 0.042 | 2.02 |
| E2+E3 (fit) | 59.4 | 68.6 | 51.9 | 56.4 | 59.1 ± 1.1 | -1.2 ± 1.9 | -9.6 ± 9.9 | 46.3 / 57.5 | 0.042 | 2.01 |
| E1+E2+E3 (equal) | 61.8 | 70.7 | 59.4 | 56.7 | 62.1 ± 0.9 | +1.8 ± 0.9 | -2.1 ± 4.7 | 65.1 / 53.8 | 0.044 | 2.04 |
| E1+E2+E3 (fit) | 60.9 | 70.4 | 58.2 | 57.0 | 61.6 ± 0.9 | +1.3 ± 1.2 | -3.4 ± 6.5 | 59.4 / 57.0 | 0.047 | 2.00 |

### Fitted member weights, trained SC static (mean ± SD over 4 folds x 5 seeds)

| Ensemble | Weights |
|---|---|
| E1 BEATs | beats+ft+pafa 0.22 ± 0.09, beats+ft+sg_scl 0.44 ± 0.09, beats+ft+patchmix_cl 0.34 ± 0.08 |
| E2 AST | ast+ft+patchmix_cl 0.56 ± 0.07, ast+ft+pafa 0.44 ± 0.07 |
| E3 HTS-AT | htsat+ft+patchmix_cl 0.51 ± 0.06, htsat+ft+sg_scl 0.49 ± 0.06 |
| E1+E2 | beats+ft+pafa 0.11 ± 0.05, beats+ft+sg_scl 0.28 ± 0.10, beats+ft+patchmix_cl 0.19 ± 0.08, ast+ft+patchmix_cl 0.23 ± 0.08, ast+ft+pafa 0.18 ± 0.06 |
| E1+E3 | beats+ft+pafa 0.11 ± 0.05, beats+ft+sg_scl 0.21 ± 0.07, beats+ft+patchmix_cl 0.16 ± 0.05, htsat+ft+patchmix_cl 0.26 ± 0.07, htsat+ft+sg_scl 0.25 ± 0.06 |
| E2+E3 | ast+ft+patchmix_cl 0.22 ± 0.07, ast+ft+pafa 0.19 ± 0.07, htsat+ft+patchmix_cl 0.28 ± 0.07, htsat+ft+sg_scl 0.31 ± 0.08 |
| E1+E2+E3 | beats+ft+pafa 0.07 ± 0.04, beats+ft+sg_scl 0.16 ± 0.08, beats+ft+patchmix_cl 0.11 ± 0.06, ast+ft+patchmix_cl 0.13 ± 0.07, ast+ft+pafa 0.12 ± 0.06, htsat+ft+patchmix_cl 0.20 ± 0.07, htsat+ft+sg_scl 0.20 ± 0.06 |


## 3. Findings

- [Fact] Trainable weights do not help: `fit` (LODO) and `fit*` (official split) are below `equal` or level with it in
  every ensemble and arm (LODO macro -0.1 to -1.5; official split -0.9 to +0.2 Score). A class-balanced NLL objective (checked
  once on LODO without SC, not in the script) gives the same picture: E1 60.3, E1+E3 59.9, E1+E2+E3 60.3 macro vs
  60.5, 60.9, 60.7 with equal weights.
- [Interpretation] The NLL fit rewards members that are well calibrated on the source devices and gives BEATs PAFA the
  smallest weight (0.21 in E1, 0.07-0.11 in the cross-group ensembles), while PAFA is the member that holds Sp on the
  unseen Litt3200. Weights learned without target-device labels cannot see which member is robust to the shift.
- [Fact] Best equal-weight ensembles vs BEATs PAFA alone (paired by seed):
  - no SC: E1+E3 official 63.4 ± 0.3 (+1.7 ± 0.5), LODO macro 60.9 ± 1.5 (+2.1 ± 1.5); E1+E2+E3 63.2 ± 0.3 (+1.5 ±
    0.4), LODO 60.7 ± 1.4 (+1.9 ± 1.0); E1 BEATs 62.5 ± 0.5 (+0.8 ± 0.5), LODO 60.5 ± 1.5 (+1.7 ± 0.9).
  - trained SC static: E1+E2+E3 LODO 62.1 ± 0.9 (+1.8 ± 0.9), E1+E3 62.0 ± 0.7 (+1.7 ± 1.1), E1 61.9 ± 1.1 (+1.6 ±
    1.1); official split +0.5 to +0.9, all within one seed SD.
- [Fact] Every ensemble loses on Litt3200 against PAFA alone (no SC: -1.1 to -9.7; with SC: -0.6 to -11.6) and on the
  official-split worst-device Score (-0.9 to -11.0). The ensembles trade PAFA's high Sp for Se: Litt3200 Sp / Se 88.5 /
  30.9 (PAFA) vs 59.5 / 56.0 (E1+E2+E3) and 62.2 / 55.0 (E1+E3).
- [Fact] AST-only and HTS-AT-only groups (E2, E3, E2+E3) are below BEATs PAFA alone or level with it everywhere (official -0.7 to -2.8;
  LODO macro 0.0 to -2.3; Litt3200 -5.7 to -11.6). They help only as complements of E1; HTS-AT complements BEATs a little
  more than AST does (E1+E3 >= E1+E2 on both splits without SC).
- [Fact] Ensembles give smaller conformal sets at the same coverage (official split 1.87-1.98 labels for cross-group vs
  2.47 for PAFA alone) and a lower LODO macro abs(Delta) (0.034-0.047 vs 0.057 without SC); not a tested H2 result.

## 4. Reading for the final AuscultTrust

- [Interpretation] Without SC, equal-weight E1, E1+E3 and E1+E2+E3 beat BEATs PAFA on both the official split and the
  LODO macro by more than one paired seed SD (largest: E1+E3, +1.7 and +2.1), at 3-7 times the inference cost and with
  a worse Litt3200 and worst-device Score. With trained SC the LODO gains stay (+1.6 to +1.8) but the official-split
  gains (+0.7 to +0.9) fall within one SD.
- [Interpretation] Learned weights are not worth pursuing in this form. Options that might work, not run: weights fitted
  on a held-out source device (leave-one-source-device-out inside the calibration set, so the fit sees a device shift),
  or distillation of the equal-weight ensemble into one BEATs model (registered only for E1 of Amendment 4).
- [Assumption] Members were chosen after seeing official-split test results (the LODO cells), and the groups were defined
  after the LODO results: all numbers here are exploratory, not registered outcomes.
