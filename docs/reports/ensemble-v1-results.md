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

## 5. Raising Se while keeping Sp; best epoch on test (optimistic)

Regenerate: `PYTHONPATH=. .venv-train/bin/python scripts/ensemble_sesp.py` (CPU, about 15 min); checks in
`tests/test_ensemble.py`.

- [Fact] Goal set by the user (22/10): more Se than BEATs PAFA at its Sp. Equal-weight ensembles of section 1 with one
  decision parameter k: the normal probability is multiplied by k before the argmax (k = 1 = plain argmax). Rules:
  - `k cal` (LODO): Sp of the ensemble on the calibration patients (source devices) = Sp of PAFA there.
  - `k*` (official split): mean log k of the LODO folds (no test cycle).
  - `k rate` (label-free, transductive like SC): on the unlabelled test cycles, the ensemble calls the same fraction of
    cycles normal as PAFA alone does on the same cycles (PAFA anchors Sp; no label is read).
  - `k test` [optimistic, oracle]: Sp matched to PAFA's Sp on the labelled test cycles.
  - `best epoch on test` [optimistic]: every member at its own best epoch on test, k = 1 (the grid's literature column).
- [Fact] d = paired by seed vs PAFA alone with the same rule (official split) or vs PAFA alone with k = 1 (LODO).

### Official split, no SC (5 seeds; d = paired vs BEATs PAFA alone with the same rule)

| Ensemble (rule) | Sp | Se | Score | HS | Worst-dev Score | dSp | dSe | dScore |
|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (last, k = 1) | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 58.9 ± 0.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k*) | 73.0 ± 2.6 | 50.4 ± 2.8 | 61.7 ± 0.6 | 59.6 ± 1.2 | 58.9 ± 0.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k rate) | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 58.9 ± 0.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k test [optimistic]) | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 58.9 ± 0.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (best epoch on test [optimistic]) | 75.5 ± 2.9 | 51.0 ± 2.8 | 63.2 ± 0.6 | 60.8 ± 1.2 | 60.0 ± 0.9 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| E1 BEATs (last, k = 1) | 72.4 ± 2.7 | 52.7 ± 1.8 | 62.5 ± 0.5 | 60.9 ± 0.3 | 57.9 ± 2.8 | -0.9 ± 3.7 | 2.4 ± 3.5 | 0.8 ± 0.5 |
| E1 BEATs (last, k*) | 69.8 ± 2.8 | 54.2 ± 1.7 | 62.0 ± 0.6 | 60.9 ± 0.2 | 56.9 ± 4.0 | -3.3 ± 3.7 | 3.8 ± 3.3 | 0.3 ± 0.5 |
| E1 BEATs (last, k rate) | 73.9 ± 3.2 | 51.7 ± 2.2 | 62.8 ± 0.7 | 60.8 ± 0.5 | 58.6 ± 1.1 | 0.7 ± 0.9 | 1.5 ± 1.1 | 1.1 ± 1.0 |
| E1 BEATs (last, k test [optimistic]) | 73.2 ± 2.7 | 52.1 ± 1.9 | 62.7 ± 0.5 | 60.8 ± 0.4 | 58.2 ± 1.3 | 0.0 ± 0.0 | 1.8 ± 1.5 | 0.9 ± 0.8 |
| E1 BEATs (best epoch on test [optimistic]) | 80.1 ± 2.6 | 50.1 ± 2.1 | 65.1 ± 0.3 | 61.6 ± 0.9 | 60.0 ± 0.9 | 4.6 ± 2.4 | -0.9 ± 2.6 | 1.9 ± 0.4 |
| E2 AST (last, k = 1) | 72.6 ± 1.8 | 47.2 ± 1.3 | 59.9 ± 0.4 | 57.2 ± 0.5 | 49.7 ± 1.9 | -0.6 ± 2.9 | -3.0 ± 2.1 | -1.8 ± 0.9 |
| E2 AST (last, k*) | 62.5 ± 1.4 | 52.9 ± 0.7 | 57.7 ± 0.5 | 57.3 ± 0.4 | 47.3 ± 1.7 | -10.5 ± 3.1 | 2.5 ± 2.6 | -4.0 ± 0.9 |
| E2 AST (last, k rate) | 71.5 ± 3.1 | 47.7 ± 2.3 | 59.6 ± 0.4 | 57.2 ± 0.6 | 49.5 ± 2.1 | -1.7 ± 1.2 | -2.5 ± 0.8 | -2.1 ± 0.9 |
| E2 AST (last, k test [optimistic]) | 73.2 ± 2.7 | 46.5 ± 2.2 | 59.9 ± 0.3 | 56.8 ± 0.7 | 49.4 ± 1.9 | 0.0 ± 0.0 | -3.7 ± 1.7 | -1.8 ± 0.9 |
| E2 AST (best epoch on test [optimistic]) | 82.7 ± 1.0 | 42.6 ± 1.1 | 62.6 ± 0.7 | 56.2 ± 1.0 | 52.8 ± 2.4 | 7.2 ± 3.1 | -8.4 ± 3.0 | -0.6 ± 1.2 |
| E3 HTS-AT (last, k = 1) | 72.1 ± 2.8 | 48.0 ± 0.9 | 60.1 ± 1.0 | 57.6 ± 0.4 | 48.3 ± 4.1 | -1.1 ± 2.3 | -2.2 ± 2.3 | -1.7 ± 1.0 |
| E3 HTS-AT (last, k*) | 64.5 ± 2.7 | 53.3 ± 1.1 | 58.9 ± 0.9 | 58.3 ± 0.6 | 46.0 ± 3.5 | -8.5 ± 2.8 | 2.9 ± 2.2 | -2.8 ± 1.0 |
| E3 HTS-AT (last, k rate) | 71.7 ± 3.4 | 48.4 ± 2.4 | 60.0 ± 1.0 | 57.7 ± 1.1 | 48.1 ± 3.8 | -1.6 ± 1.0 | -1.9 ± 1.4 | -1.7 ± 1.1 |
| E3 HTS-AT (last, k test [optimistic]) | 73.2 ± 2.7 | 47.3 ± 2.0 | 60.3 ± 0.9 | 57.4 ± 1.1 | 48.7 ± 3.7 | 0.0 ± 0.0 | -2.9 ± 1.7 | -1.4 ± 0.9 |
| E3 HTS-AT (best epoch on test [optimistic]) | 77.8 ± 3.3 | 47.1 ± 2.2 | 62.5 ± 1.5 | 58.6 ± 1.5 | 51.9 ± 5.8 | 2.3 ± 4.1 | -3.9 ± 2.8 | -0.8 ± 1.5 |
| E1+E2 (last, k = 1) | 73.6 ± 1.0 | 50.9 ± 1.1 | 62.3 ± 0.6 | 60.2 ± 0.7 | 56.6 ± 0.9 | 0.4 ± 2.4 | 0.7 ± 2.8 | 0.5 ± 1.2 |
| E1+E2 (last, k*) | 64.1 ± 1.1 | 57.1 ± 1.0 | 60.6 ± 0.2 | 60.4 ± 0.2 | 54.0 ± 2.4 | -8.9 ± 2.5 | 6.7 ± 2.4 | -1.1 ± 0.6 |
| E1+E2 (last, k rate) | 73.7 ± 3.0 | 51.1 ± 2.8 | 62.4 ± 0.6 | 60.2 ± 1.0 | 56.7 ± 1.0 | 0.4 ± 1.3 | 0.8 ± 1.3 | 0.6 ± 1.3 |
| E1+E2 (last, k test [optimistic]) | 73.2 ± 2.7 | 51.2 ± 2.9 | 62.2 ± 0.4 | 60.2 ± 1.0 | 56.6 ± 1.2 | 0.0 ± 0.0 | 1.0 ± 2.2 | 0.5 ± 1.1 |
| E1+E2 (best epoch on test [optimistic]) | 83.2 ± 1.4 | 47.5 ± 1.1 | 65.3 ± 0.5 | 60.5 ± 0.7 | 59.6 ± 0.8 | 7.7 ± 1.7 | -3.5 ± 2.7 | 2.1 ± 0.7 |
| E1+E3 (last, k = 1) | 74.0 ± 2.0 | 52.7 ± 1.7 | 63.4 ± 0.3 | 61.6 ± 0.5 | 57.2 ± 3.5 | 0.8 ± 2.9 | 2.5 ± 3.0 | 1.7 ± 0.5 |
| E1+E3 (last, k*) | 64.1 ± 3.0 | 58.5 ± 1.5 | 61.3 ± 0.8 | 61.1 ± 0.6 | 53.8 ± 3.5 | -8.9 ± 3.1 | 8.1 ± 2.8 | -0.4 ± 0.6 |
| E1+E3 (last, k rate) | 73.8 ± 2.8 | 52.6 ± 2.7 | 63.2 ± 0.4 | 61.3 ± 0.9 | 57.3 ± 2.3 | 0.6 ± 0.5 | 2.3 ± 0.8 | 1.5 ± 0.6 |
| E1+E3 (last, k test [optimistic]) | 73.2 ± 2.7 | 53.1 ± 2.4 | 63.2 ± 0.3 | 61.5 ± 0.6 | 57.2 ± 2.4 | 0.0 ± 0.0 | 2.8 ± 1.2 | 1.4 ± 0.6 |
| E1+E3 (best epoch on test [optimistic]) | 81.9 ± 2.2 | 49.7 ± 1.9 | 65.8 ± 0.7 | 61.8 ± 1.1 | 61.5 ± 3.5 | 6.4 ± 3.4 | -1.3 ± 3.1 | 2.6 ± 0.8 |
| E2+E3 (last, k = 1) | 73.4 ± 1.9 | 48.3 ± 1.1 | 60.8 ± 0.9 | 58.2 ± 0.8 | 48.5 ± 3.3 | 0.1 ± 1.7 | -1.9 ± 2.1 | -0.9 ± 1.0 |
| E2+E3 (last, k*) | 62.1 ± 2.1 | 55.6 ± 1.3 | 58.8 ± 0.9 | 58.6 ± 0.9 | 43.3 ± 2.6 | -11.0 ± 1.5 | 5.2 ± 1.8 | -2.9 ± 0.9 |
| E2+E3 (last, k rate) | 72.0 ± 3.3 | 49.4 ± 2.3 | 60.7 ± 0.8 | 58.5 ± 0.9 | 48.1 ± 2.7 | -1.3 ± 1.0 | -0.8 ± 1.4 | -1.0 ± 1.0 |
| E2+E3 (last, k test [optimistic]) | 73.2 ± 2.7 | 48.6 ± 2.0 | 60.9 ± 0.7 | 58.3 ± 0.9 | 49.0 ± 2.9 | 0.0 ± 0.0 | -1.6 ± 1.6 | -0.8 ± 0.8 |
| E2+E3 (best epoch on test [optimistic]) | 83.5 ± 1.5 | 45.2 ± 1.2 | 64.4 ± 0.7 | 58.6 ± 0.9 | 56.3 ± 1.8 | 8.0 ± 3.8 | -5.8 ± 3.2 | 1.1 ± 1.0 |
| E1+E2+E3 (last, k = 1) | 75.1 ± 0.9 | 51.4 ± 0.6 | 63.2 ± 0.3 | 61.0 ± 0.3 | 56.5 ± 1.4 | 1.8 ± 2.3 | 1.2 ± 2.4 | 1.5 ± 0.4 |
| E1+E2+E3 (last, k*) | 63.1 ± 1.3 | 58.0 ± 1.2 | 60.6 ± 0.4 | 60.5 ± 0.4 | 51.3 ± 2.6 | -9.9 ± 2.4 | 7.6 ± 2.7 | -1.1 ± 0.9 |
| E1+E2+E3 (last, k rate) | 73.7 ± 3.0 | 52.0 ± 2.4 | 62.9 ± 0.3 | 60.9 ± 0.6 | 55.3 ± 0.7 | 0.5 ± 1.0 | 1.8 ± 0.9 | 1.1 ± 0.8 |
| E1+E2+E3 (last, k test [optimistic]) | 73.2 ± 2.7 | 52.3 ± 2.1 | 62.8 ± 0.4 | 61.0 ± 0.5 | 55.2 ± 0.9 | 0.0 ± 0.0 | 2.1 ± 1.4 | 1.1 ± 0.7 |
| E1+E2+E3 (best epoch on test [optimistic]) | 84.0 ± 1.5 | 47.6 ± 1.2 | 65.8 ± 0.5 | 60.8 ± 0.8 | 60.2 ± 1.8 | 8.5 ± 2.7 | -3.3 ± 2.6 | 2.6 ± 0.4 |

### LODO v1, no SC (macro over the 4 held-out devices per seed; d = paired vs BEATs PAFA alone, k = 1)

| Ensemble (rule) | Macro Sp | Macro Se | Macro Score | dSp | dSe | dScore | Litt3200 Sp / Se / Score | AKGC417L Sp / Se / Score |
|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (k = 1) | 87.4 ± 3.7 | 30.2 ± 1.8 | 58.8 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 88.5 / 30.9 / 59.7 | 84.7 / 28.0 / 56.4 |
| BEATs PAFA alone (k cal) | 87.2 ± 3.7 | 30.5 ± 1.8 | 58.8 ± 1.4 | -0.2 ± 0.1 | 0.3 ± 0.2 | 0.0 ± 0.1 | 88.4 / 31.1 / 59.7 | 84.0 / 28.8 / 56.4 |
| BEATs PAFA alone (k rate) | 87.4 ± 3.7 | 30.2 ± 1.8 | 58.8 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 88.5 / 30.9 / 59.7 | 84.7 / 28.0 / 56.4 |
| BEATs PAFA alone (k test [optimistic]) | 87.4 ± 3.7 | 30.6 ± 1.8 | 59.0 ± 1.5 | 0.0 ± 0.0 | 0.4 ± 0.4 | 0.2 ± 0.2 | 88.5 / 31.8 / 60.1 | 84.7 / 28.1 / 56.4 |
| E1 BEATs (k = 1) | 83.3 ± 5.4 | 37.8 ± 3.5 | 60.5 ± 1.5 | -4.1 ± 3.1 | 7.6 ± 2.2 | 1.7 ± 0.9 | 68.5 / 47.8 / 58.2 | 85.4 / 31.9 / 58.7 |
| E1 BEATs (k cal) | 80.7 ± 6.2 | 40.4 ± 3.5 | 60.6 ± 1.7 | -6.6 ± 3.1 | 10.2 ± 2.2 | 1.8 ± 1.0 | 64.1 / 52.8 / 58.5 | 84.2 / 33.3 / 58.7 |
| E1 BEATs (k rate) | 88.3 ± 3.6 | 32.8 ± 2.2 | 60.5 ± 0.9 | 0.9 ± 0.3 | 2.6 ± 1.2 | 1.7 ± 0.7 | 88.2 / 30.2 / 59.2 | 86.3 / 30.3 / 58.3 |
| E1 BEATs (k test [optimistic]) | 87.4 ± 3.7 | 34.7 ± 2.4 | 61.0 ± 0.9 | 0.0 ± 0.0 | 4.5 ± 1.3 | 2.2 ± 0.7 | 88.5 / 30.3 / 59.4 | 84.7 / 32.7 / 58.7 |
| E2 AST (k = 1) | 80.1 ± 1.9 | 36.1 ± 1.9 | 58.1 ± 0.7 | -7.3 ± 3.4 | 5.9 ± 1.8 | -0.7 ± 1.3 | 52.7 / 52.2 / 52.5 | 89.3 / 23.5 / 56.4 |
| E2 AST (k cal) | 72.8 ± 5.2 | 41.0 ± 4.3 | 56.9 ± 1.2 | -14.6 ± 5.1 | 10.8 ± 3.9 | -1.9 ± 1.5 | 41.7 / 60.0 / 50.8 | 84.7 / 26.7 / 55.7 |
| E2 AST (k rate) | 86.8 ± 3.0 | 29.5 ± 3.8 | 58.2 ± 1.5 | -0.5 ± 1.2 | -0.7 ± 3.3 | -0.6 ± 2.2 | 85.0 / 23.0 / 54.0 | 84.6 / 26.3 / 55.5 |
| E2 AST (k test [optimistic]) | 87.4 ± 3.7 | 28.5 ± 5.0 | 57.9 ± 1.3 | 0.0 ± 0.0 | -1.7 ± 4.2 | -0.9 ± 2.1 | 88.5 / 17.6 / 53.1 | 84.7 / 26.9 / 55.8 |
| E3 HTS-AT (k = 1) | 76.1 ± 4.8 | 38.3 ± 2.7 | 57.2 ± 1.8 | -11.3 ± 6.4 | 8.1 ± 2.7 | -1.6 ± 2.4 | 40.5 / 59.5 / 50.0 | 88.5 / 23.8 / 56.2 |
| E3 HTS-AT (k cal) | 69.8 ± 5.3 | 42.8 ± 3.6 | 56.3 ± 1.6 | -17.5 ± 7.1 | 12.6 ± 4.0 | -2.5 ± 2.4 | 31.5 / 63.2 / 47.3 | 89.0 / 23.7 / 56.3 |
| E3 HTS-AT (k rate) | 86.8 ± 3.6 | 30.2 ± 3.0 | 58.5 ± 1.4 | -0.6 ± 1.2 | -0.0 ± 2.5 | -0.3 ± 1.8 | 85.7 / 25.8 / 55.8 | 84.1 / 27.8 / 56.0 |
| E3 HTS-AT (k test [optimistic]) | 87.4 ± 3.7 | 29.2 ± 4.4 | 58.3 ± 1.4 | 0.0 ± 0.0 | -1.0 ± 3.8 | -0.5 ± 1.9 | 88.5 / 20.5 / 54.5 | 84.7 / 27.4 / 56.1 |
| E1+E2 (k = 1) | 84.1 ± 4.2 | 37.0 ± 2.7 | 60.5 ± 1.5 | -3.3 ± 2.2 | 6.8 ± 1.4 | 1.7 ± 0.9 | 64.5 / 51.0 / 57.8 | 88.5 / 28.5 / 58.5 |
| E1+E2 (k cal) | 77.0 ± 6.0 | 42.9 ± 4.8 | 59.9 ± 1.4 | -10.4 ± 3.8 | 12.7 ± 4.1 | 1.1 ± 0.5 | 51.3 / 59.6 / 55.4 | 83.7 / 34.1 / 58.9 |
| E1+E2 (k rate) | 88.6 ± 3.4 | 32.8 ± 2.6 | 60.7 ± 1.1 | 1.2 ± 0.9 | 2.6 ± 1.5 | 1.9 ± 1.1 | 88.1 / 29.0 / 58.5 | 86.6 / 30.2 / 58.4 |
| E1+E2 (k test [optimistic]) | 87.4 ± 3.7 | 34.5 ± 3.3 | 60.9 ± 1.2 | 0.0 ± 0.0 | 4.3 ± 2.2 | 2.2 ± 1.1 | 88.5 / 29.0 / 58.7 | 84.7 / 32.8 / 58.7 |
| E1+E3 (k = 1) | 83.4 ± 3.9 | 38.4 ± 2.7 | 60.9 ± 1.5 | -4.0 ± 3.6 | 8.2 ± 1.9 | 2.1 ± 1.5 | 62.2 / 55.0 / 58.6 | 88.6 / 28.8 / 58.7 |
| E1+E3 (k cal) | 75.4 ± 6.0 | 44.1 ± 4.2 | 59.8 ± 1.7 | -12.0 ± 5.2 | 13.9 ± 3.8 | 1.0 ± 1.7 | 44.7 / 62.4 / 53.6 | 85.2 / 32.7 / 59.0 |
| E1+E3 (k rate) | 88.6 ± 3.6 | 33.3 ± 2.1 | 60.9 ± 1.1 | 1.2 ± 0.6 | 3.1 ± 1.4 | 2.1 ± 0.8 | 88.1 / 29.5 / 58.8 | 86.7 / 30.7 / 58.7 |
| E1+E3 (k test [optimistic]) | 87.4 ± 3.7 | 35.4 ± 2.7 | 61.4 ± 1.3 | 0.0 ± 0.0 | 5.2 ± 2.1 | 2.6 ± 1.0 | 88.5 / 29.4 / 58.9 | 84.7 / 33.2 / 59.0 |
| E2+E3 (k = 1) | 80.6 ± 3.8 | 37.0 ± 2.6 | 58.8 ± 1.6 | -6.8 ± 5.0 | 6.8 ± 2.6 | -0.0 ± 1.9 | 46.7 / 57.1 / 51.9 | 91.1 / 22.8 / 57.0 |
| E2+E3 (k cal) | 70.0 ± 5.8 | 43.9 ± 4.0 | 57.0 ± 1.8 | -17.3 ± 6.4 | 13.7 ± 3.7 | -1.8 ± 2.2 | 27.5 / 64.9 / 46.2 | 87.1 / 27.0 / 57.1 |
| E2+E3 (k rate) | 88.0 ± 3.3 | 31.7 ± 3.7 | 59.8 ± 1.5 | 0.6 ± 1.2 | 1.5 ± 3.1 | 1.0 ± 2.1 | 86.5 / 26.4 / 56.5 | 85.6 / 28.4 / 57.0 |
| E2+E3 (k test [optimistic]) | 87.4 ± 3.7 | 31.7 ± 5.2 | 59.6 ± 1.7 | 0.0 ± 0.0 | 1.5 ± 4.6 | 0.8 ± 2.3 | 88.5 / 23.3 / 55.9 | 84.7 / 29.7 / 57.2 |
| E1+E2+E3 (k = 1) | 83.7 ± 3.6 | 37.6 ± 3.0 | 60.7 ± 1.4 | -3.6 ± 2.8 | 7.4 ± 2.2 | 1.9 ± 1.0 | 59.5 / 56.0 / 57.8 | 90.3 / 26.2 / 58.3 |
| E1+E2+E3 (k cal) | 74.9 ± 6.3 | 44.5 ± 5.0 | 59.7 ± 1.3 | -12.5 ± 5.2 | 14.3 ± 4.4 | 0.9 ± 1.0 | 41.6 / 62.9 / 52.3 | 85.1 / 32.6 / 58.9 |
| E1+E2+E3 (k rate) | 88.8 ± 3.4 | 33.3 ± 2.5 | 61.0 ± 1.1 | 1.4 ± 0.8 | 3.1 ± 1.6 | 2.2 ± 1.1 | 88.0 / 29.1 / 58.6 | 86.8 / 30.4 / 58.6 |
| E1+E2+E3 (k test [optimistic]) | 87.4 ± 3.7 | 35.5 ± 3.2 | 61.4 ± 1.1 | 0.0 ± 0.0 | 5.3 ± 2.5 | 2.6 ± 1.2 | 88.5 / 28.3 / 58.4 | 84.7 / 33.1 / 58.9 |

### Official split, trained SC static (5 seeds; d = paired vs BEATs PAFA alone with the same rule)

| Ensemble (rule) | Sp | Se | Score | HS | Worst-dev Score | dSp | dSe | dScore |
|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (last, k = 1) | 75.6 ± 3.1 | 48.9 ± 2.9 | 62.3 ± 0.9 | 59.3 ± 1.3 | 58.9 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k*) | 75.5 ± 3.2 | 49.0 ± 2.9 | 62.2 ± 0.8 | 59.3 ± 1.3 | 58.9 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k rate) | 75.6 ± 3.1 | 48.9 ± 2.9 | 62.3 ± 0.9 | 59.3 ± 1.3 | 58.9 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k test [optimistic]) | 75.6 ± 3.1 | 49.0 ± 2.9 | 62.3 ± 0.8 | 59.3 ± 1.3 | 58.9 ± 1.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (best epoch on test [optimistic]) | 85.1 ± 1.5 | 44.2 ± 1.9 | 64.6 ± 0.4 | 58.2 ± 1.3 | 62.1 ± 1.5 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| E1 BEATs (last, k = 1) | 74.6 ± 1.1 | 51.7 ± 0.7 | 63.1 ± 0.6 | 61.1 ± 0.5 | 58.0 ± 2.5 | -1.1 ± 3.5 | 2.8 ± 2.3 | 0.9 ± 1.1 |
| E1 BEATs (last, k*) | 71.8 ± 1.2 | 52.9 ± 0.7 | 62.4 ± 0.5 | 60.9 ± 0.4 | 56.3 ± 3.0 | -3.6 ± 3.5 | 3.9 ± 2.3 | 0.1 ± 1.2 |
| E1 BEATs (last, k rate) | 76.5 ± 3.3 | 50.3 ± 2.4 | 63.4 ± 0.5 | 60.6 ± 0.7 | 59.1 ± 1.2 | 0.9 ± 0.7 | 1.4 ± 1.1 | 1.2 ± 0.9 |
| E1 BEATs (last, k test [optimistic]) | 75.6 ± 3.1 | 50.8 ± 2.4 | 63.2 ± 0.5 | 60.7 ± 0.7 | 58.4 ± 0.9 | 0.0 ± 0.0 | 1.8 ± 1.5 | 0.9 ± 0.7 |
| E1 BEATs (best epoch on test [optimistic]) | 84.8 ± 1.9 | 46.8 ± 1.2 | 65.8 ± 0.5 | 60.3 ± 0.6 | 60.3 ± 1.6 | -0.3 ± 1.7 | 2.6 ± 2.1 | 1.1 ± 0.8 |
| E2 AST (last, k = 1) | 73.6 ± 1.6 | 46.7 ± 0.5 | 60.2 ± 0.7 | 57.2 ± 0.4 | 52.2 ± 2.7 | -2.1 ± 3.3 | -2.2 ± 2.8 | -2.1 ± 0.4 |
| E2 AST (last, k*) | 63.4 ± 1.7 | 52.5 ± 0.5 | 58.0 ± 0.7 | 57.4 ± 0.5 | 49.1 ± 1.8 | -12.1 ± 3.1 | 3.5 ± 3.1 | -4.3 ± 0.3 |
| E2 AST (last, k rate) | 73.6 ± 3.3 | 46.2 ± 2.6 | 59.9 ± 0.4 | 56.7 ± 0.9 | 52.1 ± 2.3 | -2.0 ± 0.7 | -2.7 ± 0.9 | -2.4 ± 0.7 |
| E2 AST (last, k test [optimistic]) | 75.6 ± 3.1 | 44.8 ± 2.9 | 60.2 ± 0.3 | 56.2 ± 1.4 | 52.1 ± 2.4 | 0.0 ± 0.0 | -4.2 ± 1.5 | -2.1 ± 0.8 |
| E2 AST (best epoch on test [optimistic]) | 81.0 ± 1.7 | 43.9 ± 1.4 | 62.5 ± 0.7 | 56.9 ± 0.9 | 56.6 ± 2.7 | -4.0 ± 2.7 | -0.3 ± 2.8 | -2.2 ± 1.0 |
| E3 HTS-AT (last, k = 1) | 71.3 ± 2.2 | 47.8 ± 1.0 | 59.5 ± 0.8 | 57.2 ± 0.5 | 47.9 ± 2.0 | -4.3 ± 3.6 | -1.1 ± 3.2 | -2.7 ± 1.4 |
| E3 HTS-AT (last, k*) | 64.1 ± 2.3 | 52.8 ± 0.7 | 58.5 ± 0.8 | 57.9 ± 0.5 | 45.0 ± 1.6 | -11.3 ± 3.1 | 3.8 ± 2.7 | -3.8 ± 1.3 |
| E3 HTS-AT (last, k rate) | 73.0 ± 3.5 | 46.4 ± 2.4 | 59.7 ± 0.8 | 56.6 ± 0.9 | 48.3 ± 1.6 | -2.7 ± 1.2 | -2.5 ± 1.6 | -2.6 ± 1.4 |
| E3 HTS-AT (last, k test [optimistic]) | 75.6 ± 3.1 | 44.2 ± 2.9 | 59.9 ± 0.6 | 55.7 ± 1.5 | 49.7 ± 1.5 | 0.0 ± 0.0 | -4.7 ± 2.4 | -2.4 ± 1.2 |
| E3 HTS-AT (best epoch on test [optimistic]) | 78.6 ± 2.4 | 47.8 ± 1.6 | 63.2 ± 0.7 | 59.4 ± 0.7 | 55.3 ± 5.0 | -6.5 ± 3.2 | 3.6 ± 2.9 | -1.5 ± 1.1 |
| E1+E2 (last, k = 1) | 75.6 ± 1.3 | 49.9 ± 1.2 | 62.7 ± 0.3 | 60.1 ± 0.5 | 57.6 ± 2.1 | -0.1 ± 2.6 | 1.0 ± 2.0 | 0.5 ± 1.0 |
| E1+E2 (last, k*) | 66.7 ± 1.6 | 55.2 ± 1.0 | 60.9 ± 0.7 | 60.4 ± 0.6 | 54.0 ± 1.3 | -8.8 ± 3.0 | 6.2 ± 2.1 | -1.3 ± 0.7 |
| E1+E2 (last, k rate) | 76.0 ± 3.1 | 49.6 ± 2.9 | 62.8 ± 0.3 | 59.9 ± 1.1 | 56.9 ± 1.7 | 0.3 ± 0.8 | 0.7 ± 1.3 | 0.5 ± 1.0 |
| E1+E2 (last, k test [optimistic]) | 75.6 ± 3.1 | 49.8 ± 2.9 | 62.7 ± 0.2 | 60.0 ± 1.1 | 56.6 ± 1.3 | 0.0 ± 0.0 | 0.9 ± 1.8 | 0.4 ± 0.9 |
| E1+E2 (best epoch on test [optimistic]) | 85.2 ± 1.4 | 45.5 ± 0.8 | 65.4 ± 0.4 | 59.3 ± 0.4 | 59.7 ± 1.8 | 0.1 ± 1.9 | 1.3 ± 1.6 | 0.7 ± 0.7 |
| E1+E3 (last, k = 1) | 75.9 ± 1.4 | 50.5 ± 0.8 | 63.2 ± 0.7 | 60.6 ± 0.7 | 57.7 ± 3.4 | 0.2 ± 3.4 | 1.6 ± 2.2 | 0.9 ± 1.4 |
| E1+E3 (last, k*) | 65.8 ± 1.6 | 56.9 ± 1.0 | 61.4 ± 0.6 | 61.0 ± 0.5 | 53.8 ± 3.2 | -9.6 ± 3.3 | 7.9 ± 2.3 | -0.9 ± 1.1 |
| E1+E3 (last, k rate) | 76.2 ± 3.1 | 50.5 ± 2.8 | 63.4 ± 0.4 | 60.7 ± 1.0 | 57.2 ± 1.9 | 0.6 ± 0.8 | 1.6 ± 1.3 | 1.1 ± 1.0 |
| E1+E3 (last, k test [optimistic]) | 75.6 ± 3.1 | 51.0 ± 2.9 | 63.3 ± 0.4 | 60.8 ± 1.1 | 57.3 ± 2.7 | 0.0 ± 0.0 | 2.0 ± 1.9 | 1.0 ± 1.0 |
| E1+E3 (best epoch on test [optimistic]) | 85.6 ± 1.6 | 46.5 ± 1.0 | 66.1 ± 0.5 | 60.3 ± 0.6 | 61.5 ± 3.5 | 0.5 ± 2.0 | 2.3 ± 2.3 | 1.4 ± 0.9 |
| E2+E3 (last, k = 1) | 73.8 ± 1.2 | 47.9 ± 0.6 | 60.9 ± 0.4 | 58.1 ± 0.3 | 49.8 ± 2.3 | -1.8 ± 3.5 | -1.0 ± 3.1 | -1.4 ± 0.7 |
| E2+E3 (last, k*) | 62.7 ± 1.1 | 55.0 ± 0.8 | 58.9 ± 0.3 | 58.6 ± 0.3 | 44.9 ± 1.1 | -12.8 ± 3.3 | 6.0 ± 3.0 | -3.4 ± 0.8 |
| E2+E3 (last, k rate) | 74.2 ± 3.5 | 47.5 ± 2.0 | 60.8 ± 0.9 | 57.8 ± 0.5 | 49.9 ± 2.8 | -1.5 ± 0.6 | -1.4 ± 1.2 | -1.4 ± 0.8 |
| E2+E3 (last, k test [optimistic]) | 75.6 ± 3.1 | 46.6 ± 2.1 | 61.1 ± 0.6 | 57.6 ± 0.7 | 50.6 ± 2.2 | 0.0 ± 0.0 | -2.4 ± 1.4 | -1.2 ± 0.7 |
| E2+E3 (best epoch on test [optimistic]) | 82.9 ± 1.8 | 45.7 ± 1.5 | 64.3 ± 0.5 | 58.9 ± 0.9 | 58.4 ± 1.5 | -2.2 ± 2.9 | 1.4 ± 2.7 | -0.4 ± 0.9 |
| E1+E2+E3 (last, k = 1) | 76.2 ± 1.3 | 49.8 ± 1.2 | 63.0 ± 0.7 | 60.2 ± 0.8 | 55.9 ± 2.7 | 0.5 ± 3.0 | 0.9 ± 1.9 | 0.7 ± 1.1 |
| E1+E2+E3 (last, k*) | 64.4 ± 1.3 | 57.1 ± 1.2 | 60.7 ± 0.7 | 60.5 ± 0.7 | 51.5 ± 1.9 | -11.1 ± 3.0 | 8.1 ± 2.0 | -1.5 ± 1.2 |
| E1+E2+E3 (last, k rate) | 76.1 ± 3.2 | 50.0 ± 3.1 | 63.1 ± 0.5 | 60.2 ± 1.2 | 55.8 ± 1.6 | 0.4 ± 0.9 | 1.1 ± 1.2 | 0.8 ± 1.0 |
| E1+E2+E3 (last, k test [optimistic]) | 75.6 ± 3.1 | 50.3 ± 3.1 | 63.0 ± 0.4 | 60.3 ± 1.3 | 55.7 ± 1.8 | 0.0 ± 0.0 | 1.3 ± 1.7 | 0.7 ± 0.8 |
| E1+E2+E3 (best epoch on test [optimistic]) | 86.1 ± 1.2 | 45.6 ± 1.0 | 65.8 ± 0.4 | 59.6 ± 0.6 | 60.2 ± 1.3 | 1.1 ± 1.9 | 1.3 ± 2.3 | 1.2 ± 0.8 |

### LODO v1, trained SC static (macro over the 4 held-out devices per seed; d = paired vs BEATs PAFA alone, k = 1)

| Ensemble (rule) | Macro Sp | Macro Se | Macro Score | dSp | dSe | dScore | Litt3200 Sp / Se / Score | AKGC417L Sp / Se / Score |
|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (k = 1) | 88.9 ± 2.1 | 31.7 ± 2.8 | 60.3 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 88.2 / 34.8 / 61.5 | 93.2 / 15.9 / 54.6 |
| BEATs PAFA alone (k cal) | 88.9 ± 2.2 | 31.9 ± 2.9 | 60.4 ± 0.8 | -0.0 ± 0.2 | 0.2 ± 0.3 | 0.1 ± 0.1 | 88.2 / 35.5 / 61.9 | 92.9 / 16.2 / 54.5 |
| BEATs PAFA alone (k rate) | 88.9 ± 2.1 | 31.7 ± 2.8 | 60.3 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 88.2 / 34.8 / 61.5 | 93.2 / 15.9 / 54.6 |
| BEATs PAFA alone (k test [optimistic]) | 88.9 ± 2.1 | 31.9 ± 2.6 | 60.4 ± 0.8 | 0.0 ± 0.0 | 0.1 ± 0.2 | 0.1 ± 0.1 | 88.2 / 35.2 / 61.7 | 93.2 / 15.9 / 54.6 |
| E1 BEATs (k = 1) | 82.7 ± 3.6 | 41.1 ± 3.3 | 61.9 ± 1.1 | -6.2 ± 1.8 | 9.4 ± 1.6 | 1.6 ± 1.1 | 72.1 / 49.8 / 61.0 | 92.0 / 22.7 / 57.3 |
| E1 BEATs (k cal) | 80.1 ± 4.5 | 43.2 ± 4.0 | 61.7 ± 1.0 | -8.8 ± 2.5 | 11.5 ± 1.9 | 1.4 ± 1.1 | 66.5 / 54.4 / 60.5 | 90.6 / 24.4 / 57.5 |
| E1 BEATs (k rate) | 90.0 ± 2.2 | 34.1 ± 2.3 | 62.0 ± 0.6 | 1.1 ± 0.2 | 2.4 ± 0.7 | 1.7 ± 0.5 | 88.4 / 36.0 / 62.2 | 94.9 / 18.1 / 56.5 |
| E1 BEATs (k test [optimistic]) | 88.9 ± 2.1 | 35.9 ± 2.2 | 62.4 ± 0.5 | 0.0 ± 0.0 | 4.2 ± 0.9 | 2.1 ± 0.5 | 88.2 / 36.0 / 62.1 | 93.2 / 21.3 / 57.3 |
| E2 AST (k = 1) | 81.9 ± 3.6 | 35.5 ± 3.2 | 58.7 ± 0.9 | -6.9 ± 3.8 | 3.8 ± 2.1 | -1.6 ± 1.2 | 65.6 / 46.1 / 55.8 | 94.8 / 15.7 / 55.2 |
| E2 AST (k cal) | 75.5 ± 3.7 | 40.3 ± 3.5 | 57.9 ± 0.5 | -13.3 ± 3.2 | 8.6 ± 1.0 | -2.4 ± 1.2 | 56.1 / 52.3 / 54.2 | 92.2 / 19.3 / 55.7 |
| E2 AST (k rate) | 88.0 ± 2.7 | 29.6 ± 2.1 | 58.8 ± 0.6 | -0.8 ± 1.0 | -2.1 ± 1.3 | -1.5 ± 1.0 | 84.6 / 25.4 / 55.0 | 93.7 / 17.3 / 55.5 |
| E2 AST (k test [optimistic]) | 88.9 ± 2.1 | 28.5 ± 2.0 | 58.7 ± 0.7 | 0.0 ± 0.0 | -3.3 ± 2.2 | -1.6 ± 1.1 | 88.2 / 18.7 / 53.5 | 93.2 / 17.9 / 55.6 |
| E3 HTS-AT (k = 1) | 76.3 ± 5.0 | 41.1 ± 3.6 | 58.7 ± 0.9 | -12.6 ± 5.6 | 9.4 ± 3.5 | -1.6 ± 1.6 | 39.6 / 60.5 / 50.1 | 91.5 / 21.5 / 56.5 |
| E3 HTS-AT (k cal) | 70.6 ± 5.4 | 45.1 ± 3.2 | 57.8 ± 1.2 | -18.3 ± 5.8 | 13.3 ± 3.1 | -2.5 ± 1.9 | 29.6 / 66.4 / 48.0 | 91.4 / 21.4 / 56.4 |
| E3 HTS-AT (k rate) | 88.1 ± 2.5 | 30.6 ± 2.7 | 59.3 ± 0.2 | -0.8 ± 1.0 | -1.1 ± 1.1 | -1.0 ± 1.0 | 83.6 / 25.3 / 54.5 | 93.5 / 18.4 / 56.0 |
| E3 HTS-AT (k test [optimistic]) | 88.9 ± 2.1 | 29.9 ± 3.0 | 59.4 ± 0.7 | 0.0 ± 0.0 | -1.8 ± 1.9 | -0.9 ± 0.9 | 88.2 / 19.9 / 54.1 | 93.2 / 18.9 / 56.0 |
| E1+E2 (k = 1) | 84.8 ± 3.1 | 38.3 ± 4.2 | 61.6 ± 1.1 | -4.1 ± 1.9 | 6.6 ± 1.7 | 1.3 ± 0.9 | 72.4 / 48.3 / 60.4 | 94.5 / 18.0 / 56.3 |
| E1+E2 (k cal) | 78.8 ± 4.5 | 44.4 ± 4.5 | 61.6 ± 1.0 | -10.1 ± 2.7 | 12.6 ± 2.1 | 1.3 ± 1.0 | 62.1 / 58.0 / 60.0 | 91.2 / 23.8 / 57.5 |
| E1+E2 (k rate) | 89.6 ± 2.4 | 33.0 ± 2.6 | 61.3 ± 0.7 | 0.7 ± 0.6 | 1.3 ± 0.8 | 1.0 ± 0.7 | 86.7 / 31.1 / 58.9 | 94.8 / 17.6 / 56.2 |
| E1+E2 (k test [optimistic]) | 88.9 ± 2.1 | 34.3 ± 2.2 | 61.6 ± 0.4 | 0.0 ± 0.0 | 2.6 ± 1.4 | 1.3 ± 0.7 | 88.2 / 29.0 / 58.6 | 93.2 / 20.5 / 56.9 |
| E1+E3 (k = 1) | 82.9 ± 3.7 | 41.2 ± 3.6 | 62.0 ± 0.7 | -6.0 ± 2.9 | 9.4 ± 1.8 | 1.7 ± 1.1 | 62.5 / 54.6 / 58.5 | 93.3 / 21.5 / 57.4 |
| E1+E3 (k cal) | 74.7 ± 6.6 | 47.2 ± 3.5 | 61.0 ± 1.7 | -14.1 ± 5.8 | 15.5 ± 2.1 | 0.7 ± 2.5 | 43.6 / 64.1 / 53.9 | 91.1 / 25.2 / 58.1 |
| E1+E3 (k rate) | 90.1 ± 2.3 | 34.6 ± 2.5 | 62.4 ± 0.8 | 1.2 ± 0.5 | 2.9 ± 0.7 | 2.1 ± 0.5 | 87.8 / 34.3 / 61.1 | 94.9 / 18.4 / 56.6 |
| E1+E3 (k test [optimistic]) | 88.9 ± 2.1 | 36.9 ± 2.1 | 62.9 ± 0.6 | 0.0 ± 0.0 | 5.1 ± 0.9 | 2.6 ± 0.4 | 88.2 / 33.9 / 61.1 | 93.2 / 21.5 / 57.4 |
| E2+E3 (k = 1) | 80.8 ± 5.0 | 38.9 ± 4.4 | 59.9 ± 0.8 | -8.1 ± 5.1 | 7.2 ± 2.8 | -0.4 ± 1.4 | 53.2 / 55.1 / 54.2 | 94.1 / 17.7 / 55.9 |
| E2+E3 (k cal) | 70.9 ± 6.0 | 46.1 ± 4.4 | 58.5 ± 1.0 | -18.0 ± 5.7 | 14.4 ± 2.4 | -1.8 ± 1.8 | 33.5 / 64.1 / 48.8 | 91.2 / 22.3 / 56.8 |
| E2+E3 (k rate) | 88.7 ± 2.4 | 31.8 ± 3.0 | 60.3 ± 0.6 | -0.1 ± 0.9 | 0.1 ± 1.3 | -0.0 ± 0.9 | 84.8 / 26.8 / 55.8 | 93.9 / 17.9 / 55.9 |
| E2+E3 (k test [optimistic]) | 88.9 ± 2.1 | 31.5 ± 2.9 | 60.2 ± 0.7 | 0.0 ± 0.0 | -0.2 ± 2.5 | -0.1 ± 1.2 | 88.2 / 21.6 / 54.9 | 93.2 / 19.5 / 56.4 |
| E1+E2+E3 (k = 1) | 84.1 ± 3.5 | 40.1 ± 4.1 | 62.1 ± 0.9 | -4.7 ± 2.7 | 8.4 ± 2.4 | 1.8 ± 0.9 | 65.1 / 53.8 / 59.4 | 94.5 / 18.8 / 56.7 |
| E1+E2+E3 (k cal) | 74.8 ± 5.7 | 47.1 ± 4.0 | 61.0 ± 1.0 | -14.0 ± 4.6 | 15.4 ± 1.8 | 0.7 ± 1.7 | 45.6 / 62.9 / 54.3 | 90.9 / 25.1 / 58.0 |
| E1+E2+E3 (k rate) | 89.9 ± 2.2 | 34.3 ± 2.8 | 62.1 ± 0.6 | 1.0 ± 0.6 | 2.5 ± 0.5 | 1.8 ± 0.4 | 86.9 / 31.8 / 59.3 | 94.8 / 18.1 / 56.4 |
| E1+E2+E3 (k test [optimistic]) | 88.9 ± 2.1 | 36.2 ± 2.6 | 62.5 ± 0.5 | 0.0 ± 0.0 | 4.5 ± 0.9 | 2.2 ± 0.4 | 88.2 / 30.5 / 59.4 | 93.2 / 21.3 / 57.3 |


### 5.1 Findings (Se at PAFA's Sp)

- [Fact] With the label-free `k rate` rule, the BEATs-containing ensembles keep Sp and raise Se in every setting:
  - E1+E3: official split Sp +0.6 ± 0.5, Se +2.3 ± 0.8, Score 63.2 ± 0.4 (+1.5 ± 0.6); LODO macro Sp +1.2 ± 0.6,
    Se +3.1 ± 1.4, Score 60.9 ± 1.1 (+2.1 ± 0.8). With trained SC static: official Sp +0.6, Se +1.6, 63.4 ± 0.4; LODO
    Sp +1.2 ± 0.5, Se +2.9 ± 0.7, Score 62.4 ± 0.8 (+2.1 ± 0.5).
  - E1 BEATs (3 models) with trained SC static: official 63.4 ± 0.5 (Sp +0.9, Se +1.4); LODO 62.0 ± 0.6 (Sp +1.1 ± 0.2,
    Se +2.4 ± 0.7), and the only ensemble that also gains on Litt3200 (Sp / Se 88.4 / 36.0 vs 88.2 / 34.8).
  - E1+E2+E3 is level with E1+E3 (LODO 61.0 ± 1.1 no SC, 62.1 ± 0.6 with SC).
- [Fact] The oracle `k test` bounds what a threshold can give at PAFA's Sp: LODO macro Se +2.6 to +5.3 for the
  BEATs-containing ensembles; `k rate` reaches about half to 60% of it without labels. On the official split `k rate` and
  `k test` give the same Score (within 0.3).
- [Fact] Without SC, no ensemble beats PAFA on Litt3200 at PAFA's Sp, even with the oracle k (Se 28.3-30.3 vs 31.8):
  PAFA alone ranks that device's cycles best. With trained SC only E1 does (oracle Se 36.0 vs 35.2). The LODO gains come from Meditron, LittC2SE and AKGC417L.
- [Fact] Rules fitted on source devices do not transfer: `k cal` keeps PAFA's Sp on the calibration patients but loses
  7-18 points of Sp on the held-out device, and `k*` loses 3-13 points of Sp on the official split. The ensemble's Sp
  falls under device shift; only a target-side anchor (PAFA's own normal rate on the same cycles) holds it.
- [Fact] AST and HTS-AT groups alone (E2, E3, E2+E3) do not reach the goal: at PAFA's Sp their Se is lower (official
  -1.6 to -4.7; Litt3200 Se 17.6-26.8).
- [Fact] Best epoch on test [optimistic]: E1+E3 65.8 ± 0.7 (no SC) and 66.1 ± 0.5 (trained SC static), E1+E2+E3
  65.8 ± 0.5 and 65.8 ± 0.4, vs PAFA alone 63.2 ± 0.6 and 64.6 ± 0.4. The test-picked epochs favour Sp (E1+E3 Sp 81.9,
  Se 49.7): this column raises Score, not Se. Published BEATs PC-MCL (best epoch on test): 65.37 ± 0.73.
- [Assumption] `k rate` needs a batch of unlabelled cycles from the target device (as SC does) and runs PAFA next to the
  ensemble; with few cycles the rate estimate is noisy (not measured here). Exploratory, as section 4.

## 6. E1 with trained SC dynamic (23/10)

Regenerate: `PYTHONPATH=. .venv-train/bin/python scripts/ensemble_e1_dynamic.py`. Only E1 (BEATs PAFA, SG-SCL, Patch-Mix
CL) has SC dynamic units (stage 8); reference = BEATs PAFA alone with SC dynamic, rules as in section 5.

### Official split, trained SC dynamic (last epoch, 5 seeds; d = paired vs BEATs PAFA trained SC dynamic)

| Ensemble | Sp | Se | Score | HS | Worst-dev Score | Cov | Worst-dev Cov | Set size | dScore | dWorst-dev Score |
|---|---|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (reference) | 77.7 ± 0.8 | 46.3 ± 0.5 | 62.0 ± 0.5 | 58.0 ± 0.4 | 59.3 ± 0.7 | 89.8 ± 0.3 | 86.8 ± 0.9 | 2.24 ± 0.10 | +0.0 ± 0.0 | +0.0 ± 0.0 |
| E1 BEATs (equal) | 77.4 ± 2.0 | 50.6 ± 0.6 | 64.0 ± 0.7 | 61.2 ± 0.3 | 61.5 ± 1.0 | 89.4 ± 0.2 | 86.9 ± 0.2 | 2.06 ± 0.06 | +2.0 ± 0.7 | +2.2 ± 0.5 |
| E1 BEATs (fit*) | 76.8 ± 2.2 | 50.6 ± 0.9 | 63.7 ± 0.7 | 61.0 ± 0.2 | 60.8 ± 1.2 | 89.4 ± 0.2 | 86.8 ± 0.2 | 2.06 ± 0.06 | +1.8 ± 0.8 | +1.5 ± 0.7 |

### LODO v1, trained SC dynamic (Score per held-out device, mean of 5 seeds; macro = mean over the 4 folds per seed)

| Ensemble | Meditron | LittC2SE | Litt3200 | AKGC417L | Macro Score | dMacro vs ref | dLitt3200 vs ref | Litt3200 Sp / Se | Macro abs(Delta) | Macro set size |
|---|---|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (reference) | 53.5 | 61.2 | 61.3 | 55.4 | 57.9 ± 1.6 | +0.0 ± 0.0 | +0.0 ± 0.0 | 88.5 / 34.2 | 0.038 | 2.33 |
| E1 BEATs (equal) | 57.1 | 64.6 | 62.2 | 57.3 | 60.3 ± 1.3 | +2.4 ± 0.7 | +0.9 ± 3.1 | 83.7 / 40.7 | 0.046 | 2.43 |
| E1 BEATs (fit) | 57.0 | 64.1 | 61.4 | 57.6 | 60.0 ± 1.1 | +2.2 ± 0.7 | +0.0 ± 3.6 | 82.5 / 40.3 | 0.044 | 2.40 |

### Fitted member weights, trained SC dynamic (mean ± SD over 4 folds x 5 seeds)

| Ensemble | Weights |
|---|---|
| E1 BEATs | beats+ft+pafa 0.22 ± 0.09, beats+ft+sg_scl 0.42 ± 0.09, beats+ft+patchmix_cl 0.36 ± 0.08 |

### Official split, trained SC dynamic (5 seeds; d = paired vs BEATs PAFA alone with the same rule)

| Ensemble (rule) | Sp | Se | Score | HS | Worst-dev Score | dSp | dSe | dScore |
|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (last, k = 1) | 77.7 ± 0.8 | 46.3 ± 0.5 | 62.0 ± 0.5 | 58.0 ± 0.4 | 59.3 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k*) | 77.5 ± 0.8 | 46.6 ± 0.5 | 62.0 ± 0.5 | 58.2 ± 0.4 | 59.3 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k rate) | 77.7 ± 0.8 | 46.3 ± 0.5 | 62.0 ± 0.5 | 58.0 ± 0.4 | 59.3 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (last, k test [optimistic]) | 77.7 ± 0.8 | 46.3 ± 0.5 | 62.0 ± 0.5 | 58.0 ± 0.4 | 59.3 ± 0.7 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| BEATs PAFA alone (best epoch on test [optimistic]) | 81.6 ± 1.1 | 46.2 ± 1.6 | 63.9 ± 0.3 | 59.0 ± 1.0 | 59.5 ± 2.8 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| E1 BEATs (last, k = 1) | 77.4 ± 2.0 | 50.6 ± 0.6 | 64.0 ± 0.7 | 61.2 ± 0.3 | 61.5 ± 1.0 | -0.3 ± 1.6 | 4.4 ± 0.9 | 2.0 ± 0.7 |
| E1 BEATs (last, k*) | 75.4 ± 2.1 | 52.0 ± 0.9 | 63.7 ± 0.7 | 61.5 ± 0.4 | 61.4 ± 1.4 | -2.1 ± 1.7 | 5.4 ± 1.0 | 1.7 ± 0.8 |
| E1 BEATs (last, k rate) | 79.5 ± 0.9 | 49.1 ± 0.5 | 64.3 ± 0.5 | 60.7 ± 0.4 | 61.7 ± 1.2 | 1.8 ± 0.6 | 2.9 ± 0.4 | 2.4 ± 0.5 |
| E1 BEATs (last, k test [optimistic]) | 77.7 ± 0.8 | 50.7 ± 0.8 | 64.2 ± 0.4 | 61.4 ± 0.5 | 62.0 ± 1.0 | 0.0 ± 0.0 | 4.4 ± 0.9 | 2.2 ± 0.5 |
| E1 BEATs (best epoch on test [optimistic]) | 82.0 ± 0.3 | 49.3 ± 2.2 | 65.6 ± 1.1 | 61.5 ± 1.7 | 61.4 ± 3.2 | 0.4 ± 1.3 | 3.1 ± 2.4 | 1.7 ± 1.1 |

### LODO v1, trained SC dynamic (macro over the 4 held-out devices per seed; d = paired vs BEATs PAFA alone, k = 1)

| Ensemble (rule) | Macro Sp | Macro Se | Macro Score | dSp | dSe | dScore | Litt3200 Sp / Se / Score | AKGC417L Sp / Se / Score |
|---|---|---|---|---|---|---|---|---|
| BEATs PAFA alone (k = 1) | 81.0 ± 2.0 | 34.7 ± 2.6 | 57.9 ± 1.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 88.5 / 34.2 / 61.3 | 90.3 / 20.5 / 55.4 |
| BEATs PAFA alone (k cal) | 80.9 ± 2.0 | 34.8 ± 2.8 | 57.9 ± 1.6 | -0.1 ± 0.1 | 0.1 ± 0.1 | -0.0 ± 0.0 | 88.4 / 34.3 / 61.4 | 89.9 / 20.9 / 55.4 |
| BEATs PAFA alone (k rate) | 81.0 ± 2.0 | 34.7 ± 2.6 | 57.9 ± 1.6 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 88.5 / 34.2 / 61.3 | 90.3 / 20.5 / 55.4 |
| BEATs PAFA alone (k test [optimistic]) | 81.0 ± 2.0 | 34.8 ± 2.6 | 57.9 ± 1.6 | 0.0 ± 0.0 | 0.0 ± 0.1 | 0.0 ± 0.1 | 88.5 / 34.3 / 61.4 | 90.3 / 20.6 / 55.5 |
| E1 BEATs (k = 1) | 81.3 ± 3.6 | 39.3 ± 3.1 | 60.3 ± 1.3 | 0.3 ± 2.1 | 4.6 ± 1.1 | 2.4 ± 0.7 | 83.7 / 40.7 / 62.2 | 91.4 / 23.3 / 57.3 |
| E1 BEATs (k cal) | 79.5 ± 4.5 | 41.3 ± 4.1 | 60.4 ± 1.2 | -1.5 ± 2.8 | 6.6 ± 2.6 | 2.5 ± 0.7 | 81.2 / 44.8 / 63.0 | 91.1 / 23.3 / 57.2 |
| E1 BEATs (k rate) | 82.1 ± 2.0 | 37.7 ± 2.2 | 59.9 ± 1.5 | 1.1 ± 0.6 | 3.0 ± 1.5 | 2.1 ± 1.0 | 88.5 / 33.6 / 61.1 | 92.0 / 22.1 / 57.0 |
| E1 BEATs (k test [optimistic]) | 81.0 ± 2.0 | 38.8 ± 2.3 | 59.9 ± 1.3 | 0.0 ± 0.0 | 4.1 ± 2.1 | 2.0 ± 1.1 | 88.5 / 33.2 / 60.9 | 90.3 / 24.6 / 57.5 |

### 6.1 Findings

- [Fact] E1 with SC dynamic has the best official-split last-epoch Score of any configuration: 64.3 ± 0.5 with `k rate`
  (Sp +1.8 ± 0.6, Se +2.9 ± 0.4 vs PAFA + SC dynamic; worst device 61.7), 64.0 ± 0.7 with plain argmax (Se +4.4).
  Against E1 with SC static and `k rate` (63.4 ± 0.5): +0.9.
- [Fact] In LODO it is weaker than E1 with SC static: macro 59.9 ± 1.5 with `k rate` (60.3 plain argmax) vs 62.0 ± 0.6;
  on Litt3200 `k rate` holds Sp (88.5) but Se stays at PAFA's level (33.6). The dynamic bound's LODO loss on Meditron
  (stage 8) carries into the ensemble.
- [Fact] Fitted weights again do not beat equal weights (official 63.7 vs 64.0, LODO 60.0 vs 60.3).
- [Fact] Best epoch on test [optimistic]: 65.6 ± 1.1 (E1 with SC static: 65.8 ± 0.5).
- [Interpretation] SC dynamic suits the official split, SC static suits unseen devices; for a device-shift claim E1 with
  SC static stays the stronger ensemble.
