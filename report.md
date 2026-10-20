# AuscultTrust: consolidated results report

| | |
| --- | --- |
| Report version | 4.0 (supersedes 3.0 of 2026-10-19, kept in git history; 2.0 is `docs/reports/report_archive_2026-10-07.md`) |
| Datetime | 2026-10-20 14:50 (+07) |
| Proposal / prereg | v9 (`docs/updates/research-proposal-v9-en.md`); `docs/updates/prereg-v8.md` with Amendments 1-5 (`docs/updates/prereg-v9-amendment*.md`) |
| Plans | `docs/updates/stage-plan-v1.md` (grid), `docs/updates/lodo-plan-v1.md` (LODO, ensembles; Amendment 4 wins where they differ) |
| Detailed sources | `docs/reports/report_2026-10-09.md` (registered core), `docs/reports/grid-v1-results.md` (grid, every cell), `docs/reports/pcmcl-v1-results.md` (PC-MCL), `docs/handoff/HANDOFF.md` (machines, commands, incidents) |

Tags: [Fact] measured in the repo, [Interpretation] reasoned, [Assumption], [Pending] not run, [Preliminary] computed on
the units finished so far (may change; no verdict). Score = (Sp + Se) / 2 on ICBHI 4-class, official split unless
stated. All numbers are mean ± SD over seeds 0-4 unless a seed count is given.

## 1. Status

[Fact] The registered core of proposal v9 is done on AST (report of 09/10): gate G passed, H2 partly met (Litt3200 fold
only, sets grew), H1 and H3 wait for the phantom. The method x encoder grid (70 cells, 350 units, 18/10) made BEATs the
reference encoder: BEATs PAFA 61.7 ± 0.7 at the last epoch, BEATs Patch-Mix CL 63.6 ± 0.8 at the test-selected epoch.
PC-MCL (20/10) did not pass its registered rule. The final AuscultTrust (encoder + method + SC) is chosen after LODO v1
and the trained-SC arms; the old AST AuscultTrust is treated as stale.

At 20/10 14:30:

| Run | What | Done | Where | Expected end |
| --- | --- | --- | --- | --- |
| `lodo-v1`, no SC (Amendment 4) | 11 cells x 4 held-out devices x 5 seeds = 220 units | 119 (54%) | A, B, E | about 20/10 18:00 |
| `lodo-v1`, trained SC static (Amendment 4) | 7 cells x (official split + 4 folds) x 5 seeds = 175 units | 20 (14 on C, 6 on D) | D now; D, A, B, E from about 18:00 | about 21/10 05:30 |
| `pcmcl-v1` (Amendment 5) | 2 cells x 5 seeds | 10 (100%) | D | done 20/10 |

Machine C is not used any more: our jobs there were killed three times by its user (section 4.2).

## 2. Overall plan

The pipeline: encoder x method grid on the official split, then LODO without SC, then SC (trained SC on 7 cells, and
test-time SC on the LODO models), then the final configuration; ensembles and distillation are optional. KAUH controls,
the screening view and the phantom tests close the project.

| # | Stage | What | Status |
| --- | --- | --- | --- |
| 0 | Phase 0 (v4-v8) | Frozen-encoder ladders, Part II shift study, Stage-1 screening of device variants | Done; `docs/reports/report_archive_2026-10-07.md` |
| 1 | Registered core (v9) | Gate G on AST Patch-Mix CL vs AuscultTrust (AST + SC); LODO on AST (baseline, Freq-MixStyle, trained and test-time SC, A2, TTA-EQ); KAUH control A | Done 09/10 (section 5) |
| 2 | Grid v1 | 5 encoders x 14 methods (8 fine-tuned, 6 frozen) x 5 seeds, official split | Done 18/10 (section 6) |
| 3 | LODO v1 without SC (L1) | 11 cells x 4 held-out devices x 5 seeds = 220 units | Running, 54% (section 8, preliminary) |
| 4 | Trained SC, static bound | 7 cells, official split + 4 LODO folds, 175 units | Running, 11% (section 9, preliminary) |
| 5 | PC-MCL (Amendment 5) | BEATs PC-MCL and PAFA + PC-MCL, official split | Done 20/10; rule not met, not carried on (section 7) |
| 6 | Inference arms (L2) | Test-time SC (static, dynamic), A2, TTA-EQ on every LODO unit (`src.evaluation.lodo_ttsc`) | [Pending], hours, after stage 3 |
| 7 | Analysis (L3) | Per-fold Sp/Se/Score/HS/2c; method vs CE; H2 (prior-matched coverage, null gate, two-level bootstrap) with set size beside abs(Delta); SC vs no SC paired by seed | [Pending], about 1 day; `docs/reports/lodo-v1-results.md` |
| 8 | Trained SC, dynamic bound | Same 7 cells | [Pending], after stage 4 |
| 9 | Ensembles E1-E4, distillation | E1 = equal-weight mean softmax of the 11 cells per fold and seed; distil only if E1 beats the best single cell | [Pending], optional |
| 10 | Final AuscultTrust | Choose encoder + method + SC with the user | After stages 7-8 |
| 11 | After LODO | KAUH positive controls on the final model, screening view (recording level), phantom tests (H1, H3) | [Pending]; phantom depends on hardware |
| - | Demo app (`app/`, another session) | FastAPI + web demo comparing SC vs no SC on BEATs Patch-Mix CL units | Built 20/10; public tunnel blocked (port 7844, domain not delegated) |

## 3. Decisions of the user

| Date | Decision |
| --- | --- |
| 15/10 | Stop earlier campaigns; run the method x encoder grid (Lung-SRAD as a reference row only, no noise arm). |
| 16/10 | SG-SCL with the DANN ramp (`grl_schedule: progress`) after the reference schedule collapsed BEATs. |
| 17/10 | After the power cut, rerun the 13 resumed units cleanly and swap them in. |
| 19/10 | LODO cells: AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs ce, patchmix_cl, sg_scl, pafa, qlung (BEATs CE replaces ce_pafa_cfg). 80/20 split stage dropped. Ensembles and distillation optional. The old AST baseline is stale; AuscultTrust = best configuration after LODO + SC. |
| 19/10 | Trained SC cells: AST patchmix_cl, pafa; HTS-AT patchmix_cl, sg_scl; BEATs patchmix_cl, pafa, sg_scl. Static bound first, dynamic later. |
| 19/10 | PC-MCL and PAFA + PC-MCL on BEATs on machine D (Amendment 5). |
| 19/10 | Machine E added; LODO without SC shared by A, B and E; E's other GPU process stopped to use the full GPU. |
| 20/10 | PC-MCL rule not met: SC cell list unchanged. |
| 20/10 | Machine C: 30%, then 60% of its GPU; after the third kill C is dropped. Remaining SC units: all to D, then shared by D, A, B and E once their LODO units end. Cut-off units always rerun from epoch 0 (no resume, no RNG mismatch). |
| 20/10 | Cleanup: unused checkpoints and caches deleted (about 175 GB); grid-v1, lodo-v1, lodo-ast, lodo-bench checkpoints and all outputs kept. |

## 4. Compute, timeline and incidents

### 4.1 Machines (20/10 14:30)

| Machine | GPU | Now | Next |
| --- | --- | --- | --- |
| A (local) | RTX 5000 Ada 32 GB | LODO no SC, `lodo-v1-A2` (25 of 36) | SC share `lodo-v1-A4` from about 18:00 |
| B | RTX 5090 32 GB | LODO no SC, `lodo-v1-B2` (45 of 89) | SC share `lodo-v1-B4` |
| E | RTX 5090 32 GB | LODO no SC, `lodo-v1-E2` (42 of 88) | SC share `lodo-v1-E4` |
| D | RTX 5090 32 GB | SC, `lodo-v1-D3` (6 of 161: D's 60 + C's 101) | SC share `lodo-v1-D4` |
| C | RTX PRO 6000 96 GB | not used | - |

- [Fact] `scripts/lodo_move.py` moves work between machines safely: it stops a scheduler by PID while its running jobs
  finish, sets cut-off units aside so they rerun from epoch 0, splits unstarted units longest first by measured
  throughput, restarts a scheduler that dies and copies the outputs to A. The D/A/B/E split starts when A2, B2 and E2
  have all ended (dry run: D 53, A 18, B 44, E 44 units).
- [Fact] Throughput in B-smoke hours per wall hour: A 0.80, B 1.94, E 1.96; D taken as E (same GPU).
- [Assumption] ±20% on every estimate.

### 4.2 Incidents

- [Fact] Power cut 17/10 10:43 (both machines, 9.6 h): 13 resumed grid units rerun cleanly and swapped in (resumed vs rerun
  Score: mean abs 0.61, largest 2.5).
- [Fact] Machine C (hostname `VMO-ANHDH35-D`): our jobs were killed by its user at 19/10 23:52, 20/10 01:30 and 20/10
  12:10. 14 SC units finished there and were copied to A (checksums verified); the 9 cut-off units were set aside and
  rerun from epoch 0 elsewhere. No technical protection exists against a user with root; C is dropped.
- [Fact] The 20/10 cleanup deleted the `screen-ast` checkpoints that the demo app served at first; the app was switched
  to BEATs Patch-Mix CL units (`lodo-v1/...+sc+sc_static/seed0`, `grid-v1/beats+ft+patchmix_cl/seed0`). The AST demo
  pipelines need two units retrained if wanted.
- [Fact] 0 failed units and 0 NaN or collapsed runs in every LODO, SC and PC-MCL list so far.

## 5. Registered core (v9, AST), from `docs/reports/report_2026-10-09.md`

| Item | Result |
| --- | --- |
| Patch-Mix CL reproduction (AST, best epoch on test) | 61.49 ± 0.91 vs 62.37 ± 0.61 published; last epoch 59.59 ± 0.47 |
| Refit at the CV epoch (seeds 0-4) | AuscultTrust (SC) 58.31 ± 1.08; baseline 56.80 ± 1.49; SC + random gain 59.72 (not selected by the registered rule) |
| Gate G (10 seeds) | Paired difference +1.91, 95% CI [+0.31, +3.50]: passes (lower bound > -1.5). Superiority not shown (patient bootstrap [-0.83, +4.84]) |
| LODO on AST (H2) | Only Litt3200 passes the null gate. Prior-matched coverage 0.670 (baseline) vs 0.904 (SC); D = +0.177 [+0.055, +0.326]; set size 1.92 vs 2.67 labels. Freq-MixStyle D = +0.126 [-0.006, +0.269] |
| KAUH control A | Test-time SC on baseline weights lowers the cross-filter flip rate 0.188 to 0.104 (+0.085 [+0.041, +0.132]); AuscultTrust weights do not (0.175). Control B2 not interpretable (no filter pair beyond the chance range) |
| H1, H3 | [Pending]: phantom recordings |

[Interpretation] H2 failed only on set size, so the LODO v1 analysis reads set size next to abs(Delta), and ensembles
(sharper softmax) may shrink sets.

## 6. Grid v1: method x encoder on the official split

### 6.1 Setup

- 70 cells x 5 seeds = 350 units, finished 18/10 23:49. Audit (`scripts/grid.py audit`): 0 problems; SHA-256 of the
  split file and every checkpoint in `outputs/train/grid-v1/provenance_grid-v1-final-{A,B}.json`.
- Encoders AST, HTS-AT, BEATs, DASS, OPERA-CT. Fine-tuned (ft) methods: CE, Patch-Mix, Patch-Mix CL, PAFA, SG-SCL,
  QLung, ADD-RSC, ce_pafa_cfg (CE with PAFA's settings: batch 32, no SpecAugment, 5 s cycles on AST/BEATs, BEATs 100
  epochs). Frozen: the same minus Patch-Mix CL and SG-SCL. Each method with its reference settings
  (`experiments/train/grid/<method>.yaml`). Lung-SRAD is a reference row only.
- Last epoch = primary. Best epoch on test = optimistic, for comparison with papers only.
- Columns (%): Sp = normal recall; Se = recall of crackle, wheeze, both; HS = harmonic mean of Sp and Se; 2c = normal
  vs abnormal view of the same predictions (2c Sp = Sp); worst-dev = lowest per-device value; Cov = LAC split-conformal
  coverage at 90% over 20 device-stratified calibration/evaluation halves of the test patients; set size = mean labels
  per set (of 4).

### 6.2 Best cell per encoder, last epoch

| Encoder | Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs | ft pafa | 73.2 ± 2.7 | 50.2 ± 2.9 | **61.7 ± 0.7** | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| HTS-AT | ft sg_scl | 72.4 ± 2.2 | 46.9 ± 1.2 | 59.6 ± 0.5 | 56.9 ± 0.2 | 62.3 ± 1.5 | 67.3 ± 0.6 | 66.9 ± 0.5 | 49.9 ± 2.8 | 90.1 ± 0.1 | 88.4 ± 0.3 | 2.38 ± 0.05 |
| DASS | ft patchmix | 70.6 ± 2.8 | 48.8 ± 1.2 | 59.7 ± 0.9 | 57.7 ± 0.4 | 61.8 ± 1.9 | 66.2 ± 0.8 | 65.9 ± 0.7 | 49.8 ± 3.4 | 89.7 ± 0.3 | 87.6 ± 0.3 | 2.15 ± 0.06 |
| AST | ft ce_pafa_cfg | 71.8 ± 2.4 | 46.2 ± 1.3 | 59.0 ± 1.0 | 56.2 ± 0.9 | 62.0 ± 1.7 | 66.9 ± 0.7 | 66.5 ± 0.7 | 48.5 ± 2.7 | 89.7 ± 0.2 | 86.6 ± 1.1 | 2.29 ± 0.04 |
| OPERA-CT | ft patchmix_cl | 65.4 ± 0.9 | 38.0 ± 1.4 | 51.7 ± 0.7 | 48.0 ± 1.1 | 53.8 ± 1.3 | 59.6 ± 0.6 | 59.0 ± 0.7 | 35.3 ± 0.9 | 90.1 ± 0.2 | 72.1 ± 2.2 | 2.83 ± 0.04 |
| AST | frozen add_rsc | 70.7 ± 2.4 | 43.7 ± 1.1 | 57.2 ± 1.5 | 54.0 ± 1.3 | 59.5 ± 1.0 | 65.1 ± 1.6 | 64.6 ± 1.4 | 47.4 ± 2.4 | 88.9 ± 0.3 | 84.0 ± 0.9 | 2.45 ± 0.04 |
| DASS | frozen add_rsc | 67.2 ± 1.5 | 45.3 ± 1.6 | 56.3 ± 0.5 | 54.1 ± 0.8 | 61.6 ± 2.3 | 64.4 ± 0.9 | 64.2 ± 1.0 | 48.5 ± 2.3 | 89.3 ± 0.3 | 81.9 ± 2.3 | 2.44 ± 0.04 |
| HTS-AT | frozen pafa | 63.1 ± 0.4 | 42.8 ± 0.2 | 52.9 ± 0.3 | 51.0 ± 0.3 | 60.0 ± 0.4 | 61.6 ± 0.3 | 61.5 ± 0.3 | 38.5 ± 1.5 | 90.0 ± 0.1 | 67.5 ± 4.5 | 2.66 ± 0.07 |
| BEATs | frozen patchmix | 67.3 ± 1.4 | 36.3 ± 1.2 | 51.8 ± 0.3 | 47.2 ± 0.7 | 50.7 ± 1.8 | 59.0 ± 0.4 | 57.8 ± 0.7 | 45.8 ± 1.0 | 89.6 ± 0.1 | 86.4 ± 0.5 | 2.22 ± 0.02 |
| OPERA-CT | frozen patchmix | 79.6 ± 2.3 | 18.6 ± 1.8 | 49.1 ± 0.5 | 30.1 ± 2.2 | 25.7 ± 2.0 | 52.6 ± 0.5 | 38.7 ± 2.1 | 43.1 ± 1.3 | 89.5 ± 0.1 | 87.5 ± 0.1 | 2.50 ± 0.02 |

### 6.3 Top 5, best epoch on test (optimistic)

| Rank | Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | beats+ft+patchmix_cl | 76.4 ± 3.8 | 50.9 ± 3.4 | 63.6 ± 0.8 | 60.9 ± 1.4 | 65.5 ± 3.2 | 70.9 ± 1.0 | 70.4 ± 0.9 | 57.8 ± 3.8 | 89.2 ± 0.3 | 86.2 ± 0.7 | 1.95 ± 0.12 |
| 2 | beats+ft+sg_scl | 78.6 ± 4.5 | 47.9 ± 3.3 | 63.2 ± 0.7 | 59.3 ± 1.3 | 62.4 ± 3.5 | 70.5 ± 0.7 | 69.4 ± 0.5 | 56.2 ± 2.9 | 89.5 ± 0.4 | 87.0 ± 0.4 | 1.99 ± 0.09 |
| 3 | beats+ft+pafa | 75.5 ± 2.9 | 51.0 ± 2.8 | 63.2 ± 0.6 | 60.8 ± 1.2 | 65.8 ± 3.2 | 70.6 ± 0.4 | 70.2 ± 0.6 | 60.0 ± 0.9 | 89.7 ± 0.2 | 86.0 ± 0.8 | 2.16 ± 0.28 |
| 4 | beats+ft+ce_pafa_cfg | 76.1 ± 3.2 | 50.0 ± 2.8 | 63.1 ± 1.1 | 60.3 ± 1.4 | 64.6 ± 3.6 | 70.3 ± 0.9 | 69.7 ± 1.1 | 58.5 ± 2.7 | 89.7 ± 0.1 | 86.3 ± 0.8 | 2.11 ± 0.26 |
| 5 | beats+ft+qlung | 77.3 ± 3.3 | 48.1 ± 2.5 | 62.7 ± 0.5 | 59.2 ± 0.9 | 62.3 ± 2.5 | 69.8 ± 0.5 | 68.9 ± 0.4 | 55.9 ± 3.2 | 89.4 ± 0.3 | 86.3 ± 1.3 | 1.98 ± 0.07 |
| (7) | beats+ft+ce | 74.9 ± 4.3 | 49.4 ± 2.3 | 62.2 ± 1.1 | 59.4 ± 0.8 | 65.2 ± 2.8 | 70.1 ± 0.8 | 69.6 ± 0.4 | 53.7 ± 3.2 | 89.7 ± 0.2 | 86.2 ± 1.0 | 2.10 ± 0.21 |
| best non-BEATs | htsat+ft+patchmix_cl | 77.0 ± 4.6 | 47.1 ± 3.5 | 62.1 ± 1.5 | 58.3 ± 1.9 | 60.9 ± 3.4 | 69.0 ± 1.4 | 67.8 ± 1.4 | 51.5 ± 5.4 | 89.6 ± 0.2 | 86.5 ± 0.6 | 2.01 ± 0.06 |

[Fact] The top 5 span 1.0 Score, below one seed SD. Best per encoder on the best epoch: BEATs Patch-Mix CL 63.6, HTS-AT
Patch-Mix CL 62.1, DASS Patch-Mix 61.7, AST CE 61.3, OPERA-CT CE 52.6. Best-epoch selection adds 0.6-4.4 Score over the
last epoch. [Interpretation] Both epoch rules put BEATs first; the method order inside BEATs is within noise.

### 6.4 Cells carried into LODO v1 and the trained-SC arms (last epoch)

LODO without SC: all 11 rows (section 8). Trained SC: rows marked SC (section 9).

| Cell | SC | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ast+ft+ce | | 68.4 ± 2.3 | 46.8 ± 2.5 | 57.6 ± 1.0 | 55.5 ± 1.4 | 64.5 ± 3.6 | 66.5 ± 1.2 | 66.3 ± 1.3 | 44.7 ± 0.9 | 89.5 ± 0.2 | 86.7 ± 0.4 | 2.35 ± 0.05 |
| ast+ft+patchmix_cl | SC | 71.7 ± 4.1 | 44.7 ± 1.7 | 58.2 ± 1.3 | 55.0 ± 0.5 | 60.7 ± 2.5 | 66.2 ± 0.9 | 65.7 ± 0.4 | 47.1 ± 3.2 | 89.3 ± 0.3 | 86.9 ± 0.4 | 2.25 ± 0.03 |
| ast+ft+pafa | SC | 69.3 ± 1.0 | 47.4 ± 1.2 | 58.3 ± 0.9 | 56.3 ± 1.0 | 63.2 ± 1.7 | 66.2 ± 1.1 | 66.1 ± 1.1 | 47.4 ± 1.9 | 89.7 ± 0.2 | 86.1 ± 0.7 | 2.34 ± 0.05 |
| htsat+ft+ce | | 70.0 ± 3.0 | 47.6 ± 0.8 | 58.8 ± 1.4 | 56.6 ± 0.9 | 62.4 ± 1.8 | 66.2 ± 1.2 | 65.9 ± 1.1 | 46.9 ± 3.7 | 90.0 ± 0.2 | 87.1 ± 1.2 | 2.23 ± 0.03 |
| htsat+ft+patchmix_cl | SC | 69.9 ± 2.7 | 48.3 ± 1.2 | 59.1 ± 1.0 | 57.1 ± 0.5 | 63.4 ± 1.9 | 66.7 ± 0.8 | 66.5 ± 0.7 | 46.5 ± 5.1 | 89.8 ± 0.1 | 87.6 ± 0.7 | 2.21 ± 0.02 |
| htsat+ft+sg_scl | SC | 72.4 ± 2.2 | 46.9 ± 1.2 | 59.6 ± 0.5 | 56.9 ± 0.2 | 62.3 ± 1.5 | 67.3 ± 0.6 | 66.9 ± 0.5 | 49.9 ± 2.8 | 90.1 ± 0.1 | 88.4 ± 0.3 | 2.38 ± 0.05 |
| beats+ft+ce | | 65.7 ± 3.7 | 52.7 ± 1.3 | 59.2 ± 1.5 | 58.4 ± 1.1 | 69.7 ± 1.4 | 67.7 ± 1.5 | 67.6 ± 1.5 | 50.4 ± 3.0 | 89.4 ± 0.4 | 86.6 ± 0.4 | 2.38 ± 0.06 |
| beats+ft+patchmix_cl | SC | 64.2 ± 2.6 | 55.2 ± 1.8 | 59.7 ± 1.3 | 59.3 ± 1.2 | 72.4 ± 1.5 | 68.3 ± 0.9 | 68.0 ± 1.0 | 52.1 ± 3.5 | 89.3 ± 0.2 | 86.3 ± 0.2 | 2.38 ± 0.08 |
| beats+ft+pafa | SC | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| beats+ft+sg_scl | SC | 71.9 ± 4.1 | 50.2 ± 1.5 | 61.1 ± 1.3 | 59.1 ± 0.5 | 65.8 ± 2.3 | 68.9 ± 1.0 | 68.6 ± 0.8 | 52.9 ± 6.4 | 89.7 ± 0.1 | 86.6 ± 0.6 | 2.59 ± 0.21 |
| beats+ft+qlung | | 67.4 ± 2.1 | 51.2 ± 1.4 | 59.3 ± 0.8 | 58.2 ± 0.7 | 67.5 ± 1.6 | 67.4 ± 0.7 | 67.4 ± 0.7 | 51.8 ± 6.9 | 89.6 ± 0.3 | 86.9 ± 0.4 | 2.40 ± 0.05 |

### 6.5 Method effects (paired difference to CE on the same encoder and seed, last epoch)

Only the rows with abs(mean) > SD on Score, plus the LODO cells; full table in `docs/reports/grid-v1-results.md`.

| Encoder | Method | ΔSp | ΔSe | ΔScore | ΔHS | Δ2c Score | ΔWorst-dev Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs | pafa | +7.5 ± 2.9 | -2.4 ± 2.4 | **+2.6 ± 1.4** | +1.1 ± 1.6 | +1.6 ± 1.8 | **+8.5 ± 2.9** |
| BEATs | sg_scl | +6.3 ± 7.2 | -2.4 ± 2.2 | +1.9 ± 2.7 | +0.7 ± 1.5 | +1.2 ± 2.3 | +2.5 ± 8.7 |
| BEATs | patchmix_cl | -1.5 ± 5.1 | +2.5 ± 2.9 | +0.5 ± 2.0 | +0.9 ± 1.7 | +0.6 ± 1.6 | +1.7 ± 6.0 |
| BEATs | qlung | +1.7 ± 3.8 | -1.4 ± 2.5 | +0.1 ± 1.3 | -0.2 ± 1.1 | -0.3 ± 1.2 | +1.4 ± 4.7 |
| AST | pafa | +0.9 ± 1.4 | +0.6 ± 3.2 | +0.7 ± 1.3 | +0.7 ± 2.0 | -0.2 ± 1.9 | +2.6 ± 2.2 |
| AST | patchmix_cl | +3.3 ± 6.2 | -2.1 ± 3.6 | +0.6 ± 2.0 | -0.5 ± 1.5 | -0.2 ± 1.7 | +2.3 ± 3.4 |
| AST | ce_pafa_cfg | +3.4 ± 2.4 | -0.7 ± 1.4 | **+1.4 ± 1.2** | +0.6 ± 1.0 | +0.5 ± 1.4 | +3.7 ± 3.1 |
| HTS-AT | patchmix_cl | -0.1 ± 1.4 | +0.7 ± 1.0 | +0.3 ± 0.5 | +0.4 ± 0.5 | +0.5 ± 0.8 | -0.4 ± 2.1 |
| HTS-AT | sg_scl | +2.4 ± 2.8 | -0.7 ± 0.6 | +0.8 ± 1.3 | +0.2 ± 0.9 | +1.1 ± 1.7 | +3.0 ± 4.0 |
| HTS-AT | pafa | -3.0 ± 2.4 | -1.4 ± 1.2 | **-2.2 ± 1.3** | -2.0 ± 1.2 | -1.4 ± 1.1 | -0.7 ± 4.9 |
| HTS-AT | ce_pafa_cfg | -3.0 ± 3.6 | -2.1 ± 3.2 | **-2.5 ± 1.1** | -2.6 ± 1.7 | -1.7 ± 1.0 | -5.2 ± 2.5 |
| DASS | patchmix | +2.5 ± 2.9 | +0.8 ± 1.6 | **+1.7 ± 1.1** | +1.4 ± 0.8 | +0.7 ± 0.4 | +3.8 ± 4.9 |
| DASS | pafa | +0.5 ± 3.3 | +1.8 ± 2.3 | **+1.2 ± 0.6** | +1.4 ± 0.6 | +1.3 ± 0.7 | +6.2 ± 2.7 |
| DASS | sg_scl | +1.9 ± 2.6 | +0.6 ± 2.0 | **+1.2 ± 1.1** | +1.0 ± 1.1 | +0.8 ± 1.0 | +2.5 ± 2.3 |
| OPERA-CT | pafa | -3.9 ± 3.2 | -0.4 ± 1.2 | **-2.2 ± 1.2** | -1.4 ± 0.7 | -1.2 ± 1.7 | +3.4 ± 1.9 |

- [Fact] The encoder matters more than the method: fine-tuned Score BEATs 58.9-61.7, HTS-AT 56.3-59.6, DASS 57.5-59.7,
  AST 56.9-59.0, OPERA-CT 49.3-51.7. Most methods gain less than one seed SD over CE.
- [Fact] Fine-tuning beats freezing (CE): AST +2.5, DASS +3.7, OPERA-CT +4.7, HTS-AT +7.0, BEATs +7.7.
- [Fact] PAFA on BEATs is the one large, consistent gain, mostly in Sp and in the worst device (+8.5). PAFA's losses vs
  its settings (pafa - ce_pafa_cfg): BEATs +1.6 ± 1.8, DASS +1.6 ± 0.4, AST -0.7 ± 0.8.
- [Fact] Frozen PAFA equals frozen ce_pafa_cfg within 0.2 Score on every encoder: its losses train a projector the
  classifier does not use. Frozen OPERA-CT nearly ignores abnormal sounds (Se 19-25%).

### 6.6 Per-class recall and ensembles (BEATs, last epoch)

| Cell | Normal (= Sp) | Crackle | Wheeze | Both | Se | Score |
| --- | --- | --- | --- | --- | --- | --- |
| beats+ft+ce | 65.7 | 62.3 | 42.6 | 35.8 | 52.7 | 59.2 |
| beats+ft+pafa | 73.2 | 60.4 | 38.7 | 35.1 | 50.2 | 61.7 |
| beats+ft+sg_scl | 71.9 | 60.6 | 39.0 | 33.6 | 50.2 | 61.1 |
| beats+ft+patchmix_cl | 64.2 | 64.5 | 45.8 | 38.2 | 55.2 | 59.7 |
| beats+ft+ce_pafa_cfg | 66.4 | 65.5 | 42.9 | 30.8 | 53.9 | 60.1 |
| beats+ft+qlung | 67.4 | 61.0 | 40.9 | 35.0 | 51.2 | 59.3 |
| beats+ft+add_rsc | 70.9 | 59.9 | 31.6 | 28.8 | 46.9 | 58.9 |

Ensembles = mean softmax at the last epoch, same seed. [test-picked] = members chosen after seeing test results.

| Ensemble (members) | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beats+ft+pafa alone (reference) | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| 8 fine-tuned methods on BEATs (8) | 70.6 ± 1.4 | 53.3 ± 1.2 | 62.0 ± 0.2 | 60.8 ± 0.3 | 68.6 ± 1.3 | 69.6 ± 0.2 | 69.6 ± 0.2 | 55.2 ± 1.2 | 89.3 ± 0.2 | 86.3 ± 0.4 | 1.95 ± 0.03 |
| CE on AST, HTS-AT, BEATs, DASS (4) | 70.4 ± 2.1 | 50.4 ± 1.0 | 60.4 ± 0.8 | 58.7 ± 0.5 | 64.6 ± 1.5 | 67.5 ± 0.6 | 67.4 ± 0.6 | 48.6 ± 2.6 | 89.6 ± 0.2 | 87.0 ± 0.6 | 1.94 ± 0.01 |
| 8 methods x 4 encoders (32) | 72.8 ± 1.3 | 50.6 ± 1.3 | 61.7 ± 0.4 | 59.7 ± 0.5 | 64.0 ± 1.1 | 68.4 ± 0.2 | 68.1 ± 0.2 | 50.9 ± 0.7 | 89.7 ± 0.1 | 85.9 ± 0.2 | 1.81 ± 0.01 |
| [test-picked] LODO set (11) | 74.2 ± 1.5 | 51.7 ± 0.5 | 63.0 ± 0.7 | 60.9 ± 0.4 | 65.9 ± 1.0 | 70.1 ± 0.3 | 69.8 ± 0.1 | 55.8 ± 2.0 | 89.5 ± 0.1 | 86.1 ± 0.2 | 1.86 ± 0.01 |
| [test-picked] LODO BEATs cells (5) | 71.3 ± 1.7 | 53.4 ± 1.3 | 62.4 ± 0.4 | 61.1 ± 0.4 | 68.8 ± 1.3 | 70.1 ± 0.3 | 70.0 ± 0.3 | 57.0 ± 2.2 | 89.5 ± 0.2 | 86.4 ± 0.3 | 2.08 ± 0.03 |

- [Fact] PAFA and SG-SCL raise Sp; Patch-Mix CL raises all three abnormal recalls. Two BEATs methods disagree on 18-28%
  of test cycles (seed 0). The 8-method BEATs ensemble picks no cell on test: +2.8 Score over BEATs CE, seed SD 1.5 to
  0.2, and sets of 1.95 labels instead of 2.47; it does not beat PAFA alone on the worst device. Averaging the last 10
  epochs of one model changes Score by 0.0-0.2: the ensemble gain comes from different models.
- [Note] The LODO-set ensemble above includes ce_pafa_cfg; LODO v1 runs BEATs CE in its place (user decision 19/10), so
  the registered E1 has the same 11-cell size with BEATs CE.

### 6.7 Coverage, published numbers, incidents

- [Fact] Pooled coverage is at nominal (88.2-90.5%) in every cell; the worst device (Littmann 3200 in 343 of 350 units)
  falls to 85-88% in most fine-tuned cells and 67-82% in many frozen cells.
- [Fact] AST best epoch on test vs published: Patch-Mix CL 61.1 ± 0.4 vs 62.37; QLung 61.0 ± 0.9 vs 62.01; ADD-RSC
  59.6 ± 0.5 vs 65.53 (the reference resizes AST input to 1024 x 256 with its own AST; here the denoising layer runs on
  each encoder's own input). Published BEATs PC-MCL: 65.37 ± 0.73 (best epoch on test; ours in section 7).
- Incidents: QLung's reference quality score is NaN on 71 AKGC417L cycles (leading silence), our port skips silent
  frames; SG-SCL's reference reversal schedule collapsed BEATs in 3 of 4 seeds, all SG-SCL cells use the DANN ramp;
  power cut 17/10 10:43 (9.6 h), 13 resumed units rerun cleanly and swapped in (resumed vs rerun Score: mean abs 0.61,
  largest 2.5).

## 7. PC-MCL on BEATs (Amendment 5), from `docs/reports/pcmcl-v1-results.md`

Official split, seeds 0-4, 100 epochs, run `pcmcl-v1` (machine D, 10 of 10 units, 0 failed).

| Cell | Epoch | Sp | Se | Score | HS | 2c Score | Worst-dev Score | Cov | Worst-dev Cov | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beats+ft+pcmcl | last | 69.3 ± 1.7 | 52.6 ± 1.5 | 60.9 ± 0.6 | 59.7 ± 0.7 | 69.3 ± 0.4 | 57.2 ± 2.6 | 89.2 ± 0.1 | 86.6 ± 0.4 | 1.95 ± 0.02 |
| beats+ft+pafa_pcmcl | last | 74.0 ± 1.7 | 48.6 ± 0.7 | 61.3 ± 1.0 | 58.6 ± 0.8 | 69.7 ± 0.9 | 58.8 ± 1.0 | 89.8 ± 0.2 | 86.8 ± 0.2 | 2.07 ± 0.05 |
| beats+ft+pafa (grid) | last | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 69.3 ± 1.1 | 58.9 ± 0.5 | 89.8 ± 0.3 | 85.4 ± 1.0 | 2.47 ± 0.16 |
| beats+ft+pcmcl | best on test (optimistic) | 77.0 ± 2.8 | 51.7 ± 3.0 | 64.3 ± 1.0 | 61.8 ± 1.6 | 71.9 ± 1.4 | 58.8 ± 3.6 | 89.2 ± 0.2 | 85.5 ± 0.5 | 1.85 ± 0.03 |
| beats+ft+pafa_pcmcl | best on test (optimistic) | 80.2 ± 4.0 | 46.9 ± 2.3 | 63.6 ± 1.1 | 59.1 ± 1.1 | 71.0 ± 0.8 | 59.3 ± 2.6 | 89.4 ± 0.3 | 84.5 ± 1.7 | 2.02 ± 0.06 |

- [Fact] Paired by seed against BEATs PAFA at the last epoch: PC-MCL -0.8 ± 0.9, PAFA + PC-MCL -0.4 ± 1.1. The
  registered rule (last-epoch Score above PAFA by more than one seed SD) is not met by either cell.
- [Fact] PC-MCL's best epochs on test are 4-6 of 100 (64.3, published 65.37 ± 0.73 with the same test-selected rule); it
  loses about 3.4 Score by the last epoch. It gives the smallest conformal sets of the BEATs cells (1.95).

## 8. LODO v1 without SC [Preliminary]

Units finished by 20/10 14:30 (119 of 220), fixed last epoch, `src.evaluation.lodo.fold_table`. Test = every cycle of the
held-out device; conformal LAC at 90% calibrated on the 20% calibration patients of the source devices. Delta = 0.90 minus
the prior-matched coverage (positive = under-coverage). Cells missing from a fold had no finished seed yet. No verdict;
the full analysis (null gate, H2 bootstrap, method effects) waits for all 220 units.

| Held-out device | Cell | Seeds | Sp | Se | Score | HS | Coverage | Delta | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meditron | ast+ft+ce | 1 | 81.1 | 28.9 | 55.0 | 42.6 | 0.942 | +0.011 | 2.80 |
| Meditron | ast+ft+patchmix_cl | 5 | 79.7 | 31.6 | 55.6 | 45.2 | 0.944 | -0.009 | 2.59 |
| Meditron | beats+ft+ce | 5 | 78.7 | 36.5 | 57.6 | 49.8 | 0.951 | -0.018 | 2.74 |
| Meditron | beats+ft+patchmix_cl | 5 | 78.9 | 36.4 | 57.7 | 49.7 | 0.952 | -0.019 | 2.67 |
| Meditron | beats+ft+sg_scl | 5 | 84.9 | 32.0 | 58.4 | 45.9 | 0.956 | -0.034 | 2.87 |
| Meditron | beats+ft+pafa | 3 | 83.6 | 27.9 | 55.7 | 41.8 | 0.917 | +0.044 | 1.99 |
| Meditron | beats+ft+qlung | 5 | 82.5 | 34.5 | 58.5 | 48.4 | 0.948 | -0.011 | 2.65 |
| LittC2SE | ast+ft+ce | 1 | 94.8 | 32.4 | 63.6 | 48.3 | 0.953 | -0.030 | 2.72 |
| LittC2SE | ast+ft+patchmix_cl | 5 | 94.7 | 31.3 | 63.0 | 46.7 | 0.926 | +0.028 | 2.49 |
| LittC2SE | htsat+ft+sg_scl | 1 | 89.6 | 38.5 | 64.0 | 53.8 | 0.912 | -0.003 | 2.24 |
| LittC2SE | beats+ft+ce | 5 | 91.3 | 37.6 | 64.4 | 53.1 | 0.926 | +0.034 | 2.34 |
| LittC2SE | beats+ft+patchmix_cl | 5 | 90.4 | 42.5 | 66.4 | 57.7 | 0.946 | +0.010 | 2.50 |
| LittC2SE | beats+ft+sg_scl | 5 | 92.2 | 37.4 | 64.8 | 53.2 | 0.953 | -0.056 | 2.55 |
| LittC2SE | beats+ft+pafa | 5 | 90.8 | 33.2 | 62.0 | 48.5 | 0.857 | +0.118 | 1.95 |
| LittC2SE | beats+ft+qlung | 5 | 92.0 | 38.3 | 65.1 | 53.9 | 0.924 | +0.027 | 2.32 |
| Litt3200 | ast+ft+ce | 1 | 19.7 | 64.2 | 42.0 | 30.1 | 0.896 | +0.069 | 2.29 |
| Litt3200 | ast+ft+patchmix_cl | 5 | 53.6 | 52.1 | 52.8 | 51.1 | 0.953 | -0.024 | 2.97 |
| Litt3200 | htsat+ft+patchmix_cl | 2 | 51.2 | 55.3 | 53.3 | 51.8 | 0.964 | -0.034 | 3.06 |
| Litt3200 | htsat+ft+sg_scl | 1 | 66.4 | 39.7 | 53.0 | 49.7 | 0.956 | +0.002 | 3.04 |
| Litt3200 | beats+ft+ce | 5 | 59.5 | 53.2 | 56.3 | 54.9 | 0.995 | -0.090 | 3.54 |
| Litt3200 | beats+ft+patchmix_cl | 5 | 56.8 | 58.3 | 57.5 | 53.4 | 0.994 | -0.090 | 3.54 |
| Litt3200 | beats+ft+sg_scl | 5 | 60.6 | 49.3 | 54.9 | 51.2 | 0.949 | -0.009 | 2.80 |
| Litt3200 | beats+ft+pafa | 5 | 88.5 | 30.9 | 59.7 | 44.5 | 0.966 | -0.002 | 2.76 |
| Litt3200 | beats+ft+qlung | 5 | 63.1 | 51.3 | 57.2 | 54.6 | 0.988 | -0.083 | 3.37 |
| AKGC417L (flagged) | ast+ft+patchmix_cl | 3 | 86.7 | 23.8 | 55.3 | 37.3 | 0.831 | -0.011 | 2.64 |
| AKGC417L (flagged) | beats+ft+ce | 5 | 83.4 | 32.5 | 57.9 | 46.7 | 0.811 | +0.004 | 2.35 |
| AKGC417L (flagged) | beats+ft+patchmix_cl | 5 | 79.9 | 37.8 | 58.8 | 51.0 | 0.840 | -0.009 | 2.36 |
| AKGC417L (flagged) | beats+ft+sg_scl | 5 | 85.3 | 30.5 | 57.9 | 44.8 | 0.793 | +0.016 | 2.05 |
| AKGC417L (flagged) | beats+ft+pafa | 1 | 88.0 | 21.3 | 54.7 | 34.3 | 0.625 | +0.087 | 1.34 |
| AKGC417L (flagged) | beats+ft+qlung | 5 | 78.2 | 37.4 | 57.8 | 49.8 | 0.852 | -0.004 | 2.46 |

Macro Score over the 4 folds, BEATs cells with all 20 units (official-split last epoch in brackets): CE 59.1 (59.2),
Patch-Mix CL 60.1 (59.7), SG-SCL 59.0 (61.1), QLung 59.7 (59.3); without the AKGC417L fold: 59.4, 60.5, 59.4, 60.3.

## 9. Trained SC (static bound) [Preliminary]

| Cell, official split, last epoch | Seeds | Sp | Se | Score | HS | Worst-dev Score | Cov | Worst-dev Cov | Set size | Paired Score vs grid no-SC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| beats+ft+patchmix_cl+sc+sc_static | 5 | 67.4 | 52.2 | 59.8 | 58.8 | 52.0 | 89.2 | 86.2 | 2.30 | +0.1 ± 1.2 |
| beats+ft+sg_scl+sc+sc_static | 2 | 73.7 | 49.0 | 61.3 | 58.8 | 53.5 | 89.8 | 87.6 | 2.46 | +0.6 ± 1.1 |

| Held-out device | Cell | Seeds | Sp | Se | Score | HS | Coverage | Delta | Set size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Meditron | beats+ft+patchmix_cl+sc+sc_static | 3 | 64.9 | 47.6 | 56.3 | 54.5 | 0.936 | -0.020 | 2.74 |
| LittC2SE | beats+ft+patchmix_cl+sc+sc_static | 5 | 92.2 | 44.8 | 68.5 | 60.0 | 0.949 | +0.003 | 2.38 |
| Litt3200 | beats+ft+patchmix_cl+sc+sc_static | 5 | 52.5 | 62.0 | 57.2 | 53.7 | 0.996 | -0.092 | 3.75 |

- [Preliminary] BEATs Patch-Mix CL with trained SC vs without (same seeds): official split +0.1; LittC2SE +2.1 Score;
  Litt3200 -0.3 Score with the same over-coverage (0.996, sets 3.75); Meditron 56.3 (3 seeds) vs 57.7.

## 10. Findings so far

1. [Fact] Encoder choice dominates method choice on the official split: BEATs leads under both epoch rules; most
   methods differ from CE on the same encoder by less than one seed SD (section 6).
2. [Fact] BEATs PAFA is the one large consistent method gain (+2.6 Score, +8.5 worst-device Score over BEATs CE), mostly
   through Sp; PAFA, SG-SCL and PC-MCL+PAFA push Sp up and Se down, Patch-Mix CL the other way.
3. [Fact] Averaging methods helps more than any single method: 8 BEATs methods 62.0 ± 0.2 (SD 0.2 vs 0.7-1.5 for single
   cells) with smaller sets (1.95 vs 2.47 labels).
4. [Fact] Test-selected epochs inflate Score by 0.6-4.4 points; PC-MCL's published-style number (64.3) comes from epochs
   4-6 and falls to 60.9 at its fixed last epoch, below PAFA.
5. [Fact] Littmann 3200 is the worst device in 343 of 350 grid units; pooled conformal coverage stays at nominal while
   the worst device drops to 85-88% (fine-tuned) and 67-82% (frozen).
6. [Preliminary] With the held-out device unseen, BEATs cells keep a macro Score of 59-60, level with their official-
   split Score; the Litt3200 fold costs 2-6 points against the official-split Score (Sp 51-63% for most cells) while BEATs PAFA keeps
   Sp 88.5% there (Score 59.7, the best on that fold).
7. [Preliminary] On Litt3200 most last-epoch BEATs cells now over-cover (coverage 0.99, sets of 3.4-3.75 of 4 labels):
   under the new shift the calibration threshold yields near-uninformative sets rather than missed labels. The registered
   AST LODO (epoch 2) under-covered on the same fold (0.670). PAFA keeps sets smaller (1.34-2.76) but under-covers on
   LittC2SE (0.857) and AKGC417L (1 seed, 0.625).
8. [Preliminary] Trained SC on BEATs Patch-Mix CL keeps the official-split Score (+0.1) and gains on LittC2SE (+2.1) but
   does not change the Litt3200 over-coverage. H2 for this run waits for the full units, the null gate and the
   test-time SC arms.

## 11. Caveats

- One split and 5 seeds: a 1-point difference is within seed noise for most cells. The last epoch is the reference
  schedule's, not a validated epoch; best-epoch numbers are test-selected.
- Grid cells differ in batch size, cycle length and SpecAugment as well as in the method; only PAFA has a settings
  control (ce_pafa_cfg).
- [Fact] The LODO v1 and SC cells were chosen after seeing official-split test results; the held-out folds contain those
  test cycles (mild leakage, stated in Amendment 4). The final AuscultTrust is selected on the LODO folds, so its LODO
  numbers are selected numbers. The registered AST results stay the registered outcomes.
- Sections 8 and 9 are partial: some cells have 1-3 seeds in a fold. Coverage claims need the null gate; expect power on
  Litt3200 mostly. The AKGC417L fold is class-mix dominated (training keeps at most 321 of 1,864 crackle cycles).
- ICBHI device is confounded with class mix and site. GPU numerics are not bit-reproducible (`cudnn.benchmark`, AMP).
  Units cut off by a crash or kill are rerun from epoch 0, never resumed, so every reported unit is an uninterrupted run.
- SC units ran on three code states (C: commit before the PC-MCL port; D, A, E: current; B: older), identical on every
  path the SC cells use (checked by file hashes and diff).

## 12. Next steps

1. Monitor (`bash scripts/grid_status.sh`; dispatcher log `logs/lodo-v1-share4-DABE.log`); at about 18:00 check that the
   D/A/B/E SC split started.
2. When LODO v1 ends (about 18:00): test-time SC (static, dynamic), A2 and TTA-EQ on every unit (stage 6), then the full
   analysis with the null gate and H2 statistics, method vs CE effects, per-fold tables (stage 7,
   `docs/reports/lodo-v1-results.md`).
3. When trained SC ends (about 21/10 05:30): SC vs no SC paired by seed on the official split and every fold; then the
   dynamic bound on the free machines.
4. Optional ensembles E1-E4 (fixed rules) and distillation if E1 beats the best single cell.
5. Choose the final AuscultTrust with the user; KAUH positive controls, screening view, phantom tests (H1, H3); update
   the demo app to the final model.

## 13. Phase 0 (before v9)

Frozen-encoder ladders, fine-tuning ladders, Part II shift study and the Stage-1 screening of device variants (P0-P8)
are in `docs/reports/report_archive_2026-10-07.md`. Main findings: the early weak base model was the encoder;
fine-tuning did not raise decodability or the coverage deficit (F4); no device variant separated from the baseline in
3-fold CV.
