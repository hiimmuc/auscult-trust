# AuscultTrust: consolidated results report

| | |
|---|---|
| Report version | 3.0 (supersedes version 2.0 of 2026-10-07, now `docs/reports/report_archive_2026-10-07.md`) |
| Datetime | 2026-10-19 18:50 (+07) |
| Proposal / prereg | v9 (`docs/updates/research-proposal-v9-en.md`); `docs/updates/prereg-v8.md` with Amendments 1-5 (`docs/updates/prereg-v9-amendment*.md`) |
| Plans | `docs/updates/stage-plan-v1.md` (grid), `docs/updates/lodo-plan-v1.md` (LODO, ensembles; superseded where Amendment 4 differs) |
| Detailed sources | `docs/reports/report_2026-10-09.md` (registered core), `docs/reports/grid-v1-results.md` (grid, every cell), `outputs/train/grid-v1/table_{last,best_test}.md`, `docs/handoff/HANDOFF.md` (machines, commands) |

Tags: [Fact] measured in the repo, [Interpretation] reasoned, [Assumption], [Pending] not run. Score = (Sp + Se) / 2 on
ICBHI 4-class, official split unless stated. All numbers are mean ± SD over seeds 0-4.

## 1. Status in one paragraph

[Fact] The registered core of proposal v9 is done on AST (report of 09/10): gate G passed (SC vs baseline +1.91 Score,
95% CI [+0.31, +3.50], 10 seeds); H2 is partly met (only the Litt3200 LODO fold passes the null gate; SC brings its
coverage from 0.670 to 0.904, but sets grow from 1.92 to 2.67 labels); H1 and H3 wait for phantom hardware. The method x
encoder grid (70 cells, 350 units, finished 18/10) showed that the encoder matters more than the method: BEATs is the
best encoder (BEATs PAFA 61.7 ± 0.7 at the last epoch; BEATs Patch-Mix CL 63.6 ± 0.8 at the test-selected epoch). The
user treats the old AST AuscultTrust as stale: the final AuscultTrust (encoder + method + SC) is chosen after LODO v1 and
the trained-SC arms (BEATs so far). Running now: LODO without SC (machines A, B; D joins when PC-MCL ends), trained SC
static (C), PC-MCL trials (D). On 19/10 machine E joined: LODO without SC is shared by A, B and E (end about 20/10
18:00); C trains SC until PC-MCL ends on D, then C and D share the remaining SC units (end about 21/10 01:30) (section 3).

## 2. Overall plan

The pipeline is: encoder sweep, then LODO without SC, then SC (test-time on the LODO models, and trained SC on 7 cells),
then the final configuration; ensembles and distillation are optional. Phantom tests close the project.

| # | Stage | What | Status |
|---|---|---|---|
| 0 | Phase 0 (v4-v8) | Frozen-encoder ladders, Part II shift study, Stage-1 screening of device variants | Done; archived in `docs/reports/report_archive_2026-10-07.md` |
| 1 | Registered core (v9) | Gate G on AST Patch-Mix CL vs AuscultTrust (AST + SC); LODO on AST (baseline, Freq-MixStyle, trained and test-time SC, A2, TTA-EQ); KAUH control A | Done 09/10 (section 4) |
| 2 | Grid v1 | 5 encoders x 14 methods (8 fine-tuned, 6 frozen) x 5 seeds, official split | Done 18/10 (section 5) |
| 3 | LODO v1 without SC (L1, Amendment 4) | 11 cells x 4 held-out devices x 5 seeds = 220 units | Running on A, B, E |
| 4 | Trained SC, static bound (Amendment 4) | 7 cells, official split + 4 LODO folds, 175 units | Running on C (D from about 20/10 11:30) |
| 5 | PC-MCL (Amendment 5) | BEATs PC-MCL and PAFA + PC-MCL, official split, 10 units | Running on D; enters LODO/SC only if its last-epoch Score beats BEATs PAFA by more than one seed SD |
| 6 | Inference arms (L2) | Test-time SC (static, dynamic), A2, TTA-EQ on every LODO unit | [Pending], hours, after stage 3 |
| 7 | Analysis (L3) | Per-fold Sp/Se/Score/HS/2c; method vs CE on the same encoder; H2 statistics (prior-matched coverage, null gate, two-level bootstrap) with set size beside abs(Delta); SC vs no SC paired by seed | [Pending], about 1 day; report `docs/reports/lodo-v1-results.md` |
| 8 | Trained SC, dynamic bound | Same 7 cells | [Pending], after stage 4 |
| 9 | Ensembles E1-E4, distillation (optional) | E1 = equal-weight mean softmax of the 11 cells per fold and seed; distil into one BEATs model only if E1 beats the best single cell | [Pending], optional |
| 10 | Final AuscultTrust | Choose encoder + method + SC with the user | After stages 7-8 |
| 11 | After LODO | KAUH positive controls on the final model, screening view (recording level), phantom tests (H1, H3) | [Pending]; phantom depends on hardware |

Decisions of the user (19/10): LODO cells as in stage 3 (BEATs CE replaces BEATs ce_pafa_cfg); the 80/20 split stage is
dropped; SC cells AST Patch-Mix CL, PAFA; HTS-AT Patch-Mix CL, SG-SCL; BEATs Patch-Mix CL, PAFA, SG-SCL; static bound
first; C limited to 30% of its GPU; PC-MCL on D now; ensembles and distillation optional.

## 3. Compute and timeline (re-estimated 19/10 18:45)

| Machine | GPU | Run | Units left | Expected end |
|---|---|---|---|---|
| A | RTX 5000 Ada 32 GB | `lodo-v1` no SC, list `lodo-v1-A2` | 36 | about 20/10 18:00 |
| B | RTX 5090 32 GB | `lodo-v1` no SC, list `lodo-v1-B2` | 89 | about 20/10 18:00 |
| E | RTX 5090 32 GB (`10.148.182.179`) | `lodo-v1` no SC, list `lodo-v1-E2` | 88 | about 20/10 18:00 |
| C | RTX PRO 6000 96 GB | `lodo-v1` trained SC static (3-4 jobs); its share `lodo-v1-C2` after PC-MCL ends on D | 161 at 20/10 01:00 | about 21/10 01:30 |
| D | RTX 5090 32 GB | `pcmcl-v1` (10 units), then its share of C's SC units (`lodo-v1-D2`) | 10 + about 60 | PC-MCL about 20/10 10:45; SC about 21/10 01:30 |

- [Fact] Machine E joined on 19/10 (data, weights, caches, code and venv copied from A; 1-epoch smoke of one AST, HTS-AT
  and BEATs LODO unit passed with all outputs). LODO without SC is shared by A, B and E; trained SC by C and D.
- [Fact] The dispatcher `scripts/lodo_move.py` does both shares. It stops the source schedulers by PID (their running
  jobs go on), keeps running and interrupted units on their machine, gives every unstarted unit, longest first, to the
  machine that would finish it first (measured throughput), and starts each machine's new sweep only after its own old
  jobs have ended, so nothing is killed and nothing runs twice. It restarts a sweep that dies (power cut) and copies the
  outputs to A at the end. A+B+E: applied 19/10 18:42, log `logs/lodo-v1-share-ABE.log`. C+D (user decision 20/10):
  waits for PC-MCL on D, then splits C's unstarted SC units between C and D, log `logs/lodo-v1-share-CD.log`.
- [Fact] Incident: our jobs on C were killed by another user at about 19/10 23:52; restarted 20/10 01:02; the two cut-off
  units rerun from epoch 0 (partial runs kept in `<outputs|checkpoints>/train/lodo-v1-interrupted/`).
- [Fact] Throughput in B-smoke hours (one job alone on B) per wall hour, measured since each sweep started: A 0.80, B 1.94,
  C 2.15; E and D assumed equal to B (same GPU). PC-MCL: about 1.9 h per unit, one at a time (a job takes about 19 GB).
- [Assumption] ±20% on every estimate; A and B idle partly while their last old jobs drain; E's speed is unmeasured.
- All training ends about 21/10 01:30, set by SC on C+D (moving all SC units to D alone would have ended about 21/10
  22:00; A, B and E are free from about 20/10 18:00 and could take SC units too).
- C runs 3-4 jobs since 20/10 01:20 (`--workers 4 --gpu-gb 34`, about 25 GB in use = 26% of the card, within the
  user's 30%); the 21/10 01:30 estimate assumes about 2.4 B-smoke hours per hour on C at 3 jobs (not yet measured).
- From about 20/10 18:00 A, B and E are idle; the dynamic-bound SC arms (stage 8, about 108 smoke hours) could run there.

## 4. Registered core (v9, AST), from `docs/reports/report_2026-10-09.md`

| Item | Result |
|---|---|
| Patch-Mix CL reproduction (AST, best epoch on test) | 61.49 ± 0.91 vs 62.37 ± 0.61 published; last epoch 59.59 ± 0.47 |
| Refit at the CV epoch (seeds 0-4) | AuscultTrust (SC) 58.31 ± 1.08; baseline 56.80 ± 1.49; SC + random gain 59.72 (not selected by the registered rule) |
| Gate G (10 seeds) | Paired difference +1.91, 95% CI [+0.31, +3.50]: passes (lower bound > -1.5). Superiority not shown (patient bootstrap [-0.83, +4.84]) |
| LODO on AST (H2) | Only Litt3200 passes the null gate. Prior-matched coverage 0.670 (baseline) vs 0.904 (SC); D = +0.177 [+0.055, +0.326]; set size 1.92 vs 2.67 labels. Freq-MixStyle D = +0.126 [-0.006, +0.269] |
| KAUH control A | Test-time SC on baseline weights lowers the cross-filter flip rate 0.188 to 0.104 (+0.085 [+0.041, +0.132]); AuscultTrust weights do not (0.175). Control B2 not interpretable (no filter pair beyond the chance range) |
| H1, H3 | [Pending]: phantom recordings |

[Interpretation] H2 failed only on set size, so the LODO v1 analysis reads set size next to abs(Delta), and ensembles
(sharper softmax) may shrink sets.

## 5. Grid v1: method x encoder on the official split

### 5.1 Setup

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

### 5.2 Best cell per encoder, last epoch

| Encoder | Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
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

### 5.3 Top 5, best epoch on test (optimistic)

| Rank | Cell | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
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

### 5.4 Cells carried into LODO v1 and the trained-SC arms (last epoch)

Run on A/B without SC: all 11 rows. Trained SC (machine C): rows marked SC.

| Cell | SC | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
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

### 5.5 Method effects (paired difference to CE on the same encoder and seed, last epoch)

Only the rows with abs(mean) > SD on Score, plus the LODO cells; full table in `docs/reports/grid-v1-results.md`.

| Encoder | Method | ΔSp | ΔSe | ΔScore | ΔHS | Δ2c Score | ΔWorst-dev Score |
|---|---|---|---|---|---|---|---|
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

### 5.6 Per-class recall and ensembles (BEATs, last epoch)

| Cell | Normal (= Sp) | Crackle | Wheeze | Both | Se | Score |
|---|---|---|---|---|---|---|
| beats+ft+ce | 65.7 | 62.3 | 42.6 | 35.8 | 52.7 | 59.2 |
| beats+ft+pafa | 73.2 | 60.4 | 38.7 | 35.1 | 50.2 | 61.7 |
| beats+ft+sg_scl | 71.9 | 60.6 | 39.0 | 33.6 | 50.2 | 61.1 |
| beats+ft+patchmix_cl | 64.2 | 64.5 | 45.8 | 38.2 | 55.2 | 59.7 |
| beats+ft+ce_pafa_cfg | 66.4 | 65.5 | 42.9 | 30.8 | 53.9 | 60.1 |
| beats+ft+qlung | 67.4 | 61.0 | 40.9 | 35.0 | 51.2 | 59.3 |
| beats+ft+add_rsc | 70.9 | 59.9 | 31.6 | 28.8 | 46.9 | 58.9 |

Ensembles = mean softmax at the last epoch, same seed. [test-picked] = members chosen after seeing test results.

| Ensemble (members) | Sp | Se | Score | HS | 2c Se | 2c Score | 2c HS | Worst-dev Score | Cov | Worst-dev Cov | Set size |
|---|---|---|---|---|---|---|---|---|---|---|---|
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

### 5.7 Coverage, published numbers, incidents

- [Fact] Pooled coverage is at nominal (88.2-90.5%) in every cell; the worst device (Littmann 3200 in 343 of 350 units)
  falls to 85-88% in most fine-tuned cells and 67-82% in many frozen cells.
- [Fact] AST best epoch on test vs published: Patch-Mix CL 61.1 ± 0.4 vs 62.37; QLung 61.0 ± 0.9 vs 62.01; ADD-RSC
  59.6 ± 0.5 vs 65.53 (the reference resizes AST input to 1024 x 256 with its own AST; here the denoising layer runs on
  each encoder's own input). Published BEATs PC-MCL: 65.37 ± 0.73 (best epoch on test; under test on D).
- Incidents: QLung's reference quality score is NaN on 71 AKGC417L cycles (leading silence), our port skips silent
  frames; SG-SCL's reference reversal schedule collapsed BEATs in 3 of 4 seeds, all SG-SCL cells use the DANN ramp;
  power cut 17/10 10:43 (9.6 h), 13 resumed units rerun cleanly and swapped in (resumed vs rerun Score: mean abs 0.61,
  largest 2.5).

## 6. Caveats

- One split and 5 seeds: a 1-point difference is within seed noise for most cells. The last epoch is the reference
  schedule's, not a validated epoch; the best epoch is test-selected.
- Grid cells differ in batch size, cycle length and SpecAugment as well as in the method; only PAFA has a settings
  control (ce_pafa_cfg).
- [Fact] The LODO v1 and SC cells were chosen after seeing official-split test results; the held-out-device folds contain
  those test cycles (mild leakage, stated in Amendment 4). The final AuscultTrust is selected on the LODO folds, so its
  LODO numbers are selected numbers. The registered AST results stay the registered outcomes.
- Expect statistical power for coverage on one fold only (Litt3200), as in the registered AST LODO; the other folds
  mainly show Score robustness. The AKGC417L fold is class-mix dominated (training keeps at most 321 of 1,864 crackle
  cycles): reported flagged, plus a macro mean without it.
- ICBHI device is confounded with class mix and site. Worst-device numbers rest on the Littmann 3200 test set (461
  cycles). GPU numerics are not bit-reproducible (`cudnn.benchmark`, AMP).
- Units resumed from `last.pth` after a crash do not restore the RNG state; rerun them cleanly at the end if exact runs
  matter (as done for the grid).

## 7. Next steps

1. Decide C's GPU share (30% cap vs 2 jobs at once).
2. Monitor A-E (`logs/<run>-<M>.log`, `logs/lodo-v1-share-{ABE,CD}.log`, `logs/gpu-<M>.csv`); when D's PC-MCL ends,
   build its table paired against BEATs PAFA and CE and apply the Amendment 5 rule.
3. When LODO v1 ends (about 21/10): stage 6 (inference arms), stage 7 (analysis, `docs/reports/lodo-v1-results.md`).
4. When C and D end: SC vs no SC paired by seed (official split vs grid units; LODO vs A/B/E units); then the dynamic bound,
   spread over the idle machines.
5. Optional ensembles E1-E4 and distillation; choose the final AuscultTrust with the user; KAUH controls, screening view,
   phantom.

## 8. Phase 0 (before v9)

Frozen-encoder ladders, fine-tuning ladders, Part II shift study and the Stage-1 screening of device variants (P0-P8)
are in `docs/reports/report_archive_2026-10-07.md`, with the v4-era detail it links. Their main findings: the early weak
base model was the encoder; fine-tuning did not raise decodability or the coverage deficit (F4); no device variant
separated from the baseline in 3-fold CV.
