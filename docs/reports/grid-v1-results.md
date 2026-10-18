# Grid v1 results: method x encoder on the ICBHI official split (19/10/2026)

Plan: `docs/updates/stage-plan-v1.md`. Tables: `outputs/train/grid-v1/table_last.md`, `table_best_test.md`.

## Summary

- [Fact] All 70 cells x 5 seeds (350 units) finished on 18/10/2026 23:49. The audit (`scripts/grid.py audit`) finds 0
  problems on both machines and on all 350 units after the sync. Checkpoint and split-file SHA-256 are in
  `outputs/train/grid-v1/provenance_grid-v1-final-{A,B}.json`.
- [Fact] Best cell: BEATs fine-tuned with PAFA, Score 61.7 +- 0.7 at the last epoch. It also has the highest worst-device
  Score (58.9 +- 0.5) and the smallest seed SD among the top cells.
- [Fact] The encoder matters more than the method. Fine-tuned Score ranges: BEATs 58.9-61.7, HTS-AT 56.3-59.6,
  DASS 57.5-59.7, AST 56.9-59.0, OPERA-CT 49.3-51.7.
- [Fact] Against CE on the same encoder (paired by seed), most methods gain less than one seed SD. The only gains above
  one SD are PAFA and SG-SCL on DASS (+1.2) and BEATs (+2.6 for PAFA), Patch-Mix on DASS (+1.7).
- [Fact] Split-conformal coverage is at nominal pooled (89-90% for 90%) but drops on the worst device, Littmann 3200 in
  343 of 350 units: 85-88% in most fine-tuned cells, 67-82% in frozen cells.
- [Interpretation] No reproduced method is a clear improvement over CE across encoders. The device-level coverage gap is
  the effect H2 targets.

## Setup

- ICBHI official 60/40 split (4,142 training, 2,756 test cycles), 4-class training; 2-class view of the same predictions.
- Encoders: AST, HTS-AT, OPERA-CT, BEATs, DASS. Fine-tuned (ft) cells: 8 methods; frozen cells: 6 methods.
- Each method keeps its reference training settings (`experiments/train/grid/<method>.yaml`). `ce_pafa_cfg` is CE with
  PAFA's settings (batch 32, no SpecAugment, 5 s cycles on AST/BEATs, BEATs 100 epochs): it separates PAFA's losses from
  its settings.
- Every unit trains to its last epoch. "Last epoch" is the primary number. "Best on test" picks the best epoch on the
  test set per seed: it is optimistic and only for comparison with papers, which report it.
- Score = (Sp + Se) / 2 in %, mean +- SD over seeds 0-4. Worst-device Score = lowest per-device Score of the unit.
  Coverage: LAC at 90%, 20 device-stratified calibration/evaluation halves of the test patients.

## Fine-tuned cells

Score at the last epoch:

| Method | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| ce | 57.6 ± 1.0 | 58.8 ± 1.4 | 51.5 ± 1.2 | 59.2 ± 1.5 | 58.1 ± 1.1 |
| patchmix | 57.5 ± 0.8 | 58.9 ± 1.2 | 51.3 ± 0.5 | 59.6 ± 1.1 | 59.7 ± 0.9 |
| patchmix_cl | 58.2 ± 1.3 | 59.1 ± 1.0 | 51.7 ± 0.7 | 59.7 ± 1.3 | 57.9 ± 0.5 |
| pafa | 58.3 ± 0.9 | 56.6 ± 1.3 | 49.3 ± 0.3 | **61.7 ± 0.7** | 59.2 ± 0.5 |
| sg_scl | 58.7 ± 0.6 | 59.6 ± 0.5 | 51.3 ± 0.8 | 61.1 ± 1.3 | 59.3 ± 1.0 |
| qlung | 58.2 ± 0.9 | 59.3 ± 1.4 | 51.0 ± 0.9 | 59.3 ± 0.8 | 58.1 ± 1.0 |
| add_rsc | 56.9 ± 0.6 | 57.8 ± 1.0 | 50.0 ± 0.5 | 58.9 ± 0.9 | 57.5 ± 1.2 |
| ce_pafa_cfg | 59.0 ± 1.0 | 56.3 ± 0.9 | 49.5 ± 0.9 | 60.1 ± 1.3 | 57.6 ± 0.8 |

Worst-device Score at the last epoch:

| Method | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| ce | 44.7 ± 0.9 | 46.9 ± 3.7 | 33.3 ± 2.6 | 50.4 ± 3.0 | 46.0 ± 3.1 |
| patchmix | 46.1 ± 1.4 | 43.6 ± 2.9 | 35.7 ± 0.9 | 48.9 ± 2.3 | 49.8 ± 3.4 |
| patchmix_cl | 47.1 ± 3.2 | 46.5 ± 5.1 | 35.3 ± 0.9 | 52.1 ± 3.5 | 47.4 ± 1.3 |
| pafa | 47.4 ± 1.9 | 46.2 ± 6.0 | 36.7 ± 1.7 | **58.9 ± 0.5** | 52.2 ± 1.8 |
| sg_scl | 43.7 ± 3.0 | 49.9 ± 2.8 | 34.0 ± 4.6 | 52.9 ± 6.4 | 48.5 ± 2.2 |
| qlung | 45.3 ± 2.6 | 47.8 ± 3.9 | 36.1 ± 1.6 | 51.8 ± 6.9 | 44.9 ± 2.6 |
| add_rsc | 45.3 ± 3.5 | 43.4 ± 3.0 | 37.1 ± 0.9 | 53.1 ± 2.4 | 45.6 ± 2.4 |
| ce_pafa_cfg | 48.5 ± 2.7 | 41.7 ± 3.6 | 38.1 ± 1.9 | 55.0 ± 2.5 | 47.1 ± 2.7 |

Paired difference to CE on the same encoder (method minus CE, same seed; * = |mean| > SD):

| Method | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| patchmix | -0.2 ± 1.6 | +0.1 ± 1.8 | -0.2 ± 1.0 | +0.4 ± 0.7 | +1.7 ± 1.1 * |
| patchmix_cl | +0.6 ± 2.0 | +0.3 ± 0.5 | +0.2 ± 1.5 | +0.5 ± 2.0 | -0.2 ± 1.1 |
| pafa | +0.7 ± 1.3 | -2.2 ± 1.3 * | -2.2 ± 1.2 * | +2.6 ± 1.4 * | +1.2 ± 0.6 * |
| sg_scl | +1.0 ± 1.3 | +0.8 ± 1.3 | -0.2 ± 1.8 | +1.9 ± 2.7 | +1.2 ± 1.1 * |
| qlung | +0.5 ± 1.0 | +0.5 ± 1.7 | -0.6 ± 0.9 | +0.1 ± 1.3 | +0.1 ± 1.1 |
| add_rsc | -0.7 ± 1.1 | -1.0 ± 1.5 | -1.6 ± 0.8 * | -0.3 ± 2.1 | -0.6 ± 1.6 |
| ce_pafa_cfg | +1.4 ± 1.2 * | -2.5 ± 1.1 * | -2.0 ± 1.4 * | +1.0 ± 1.6 | -0.5 ± 0.4 * |

PAFA losses vs PAFA settings (pafa minus ce_pafa_cfg, paired): AST -0.7 ± 0.8, HTS-AT +0.3 ± 0.8, OPERA-CT -0.1 ± 1.0,
BEATs +1.6 ± 1.8, DASS +1.6 ± 0.4.

- [Fact] PAFA's settings alone (ce_pafa_cfg) help AST (+1.4) and hurt HTS-AT and OPERA-CT (-2.5, -2.0). On HTS-AT,
  OPERA-CT and DASS the PAFA cells keep 8 s cycles (fixed input length), so only batch size and SpecAugment change.
- [Fact] PAFA's own losses add a gain beyond its settings only on DASS (+1.6 ± 0.4); on BEATs the gain (+1.6) is within
  one SD. On BEATs PAFA lifts the worst device most (58.9 vs 50.4 for CE): its worst device is AKGC417L in all 5 seeds,
  while Littmann 3200 is the worst device in 194 of 200 fine-tuned units overall.
- [Interpretation] Method rankings change with the encoder; no method is a safe default. The worst-device Score has large
  seed SD (up to 6.9), driven by the small Littmann 3200 test set (461 cycles).

## Best epoch on test (optimistic)

Per seed, the epoch with the highest test Score; mean ± SD over seeds. Test-selected: for comparison with papers only, not
for claims. Bold with rank = top 5 overall (the top 3 are all BEATs, so the list extends to 5).

Fine-tuned, Score:

| Method | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| ce | 61.3 ± 1.8 | 61.5 ± 1.2 | 52.6 ± 0.8 | 62.2 ± 1.1 | 61.0 ± 0.7 |
| patchmix | 59.5 ± 0.4 | 60.9 ± 0.8 | 52.4 ± 0.3 | 62.6 ± 1.1 | 61.7 ± 1.3 |
| patchmix_cl | 61.1 ± 0.4 | 62.1 ± 1.5 | 52.5 ± 0.4 | **63.6 ± 0.8** (1) | 60.3 ± 0.5 |
| pafa | 61.2 ± 0.7 | 59.0 ± 2.0 | 50.4 ± 0.3 | **63.2 ± 0.6** (3) | 60.9 ± 0.6 |
| sg_scl | 61.1 ± 0.4 | 61.8 ± 1.4 | 52.4 ± 1.0 | **63.2 ± 0.7** (2) | 61.6 ± 0.6 |
| qlung | 61.0 ± 0.9 | 61.3 ± 1.3 | 52.5 ± 0.4 | **62.7 ± 0.5** (5) | 61.4 ± 0.9 |
| add_rsc | 59.6 ± 0.5 | 61.3 ± 1.7 | 51.4 ± 0.5 | 61.4 ± 1.0 | 59.6 ± 1.6 |
| ce_pafa_cfg | 60.5 ± 0.4 | 58.3 ± 1.6 | 50.7 ± 0.8 | **63.1 ± 1.1** (4) | 59.2 ± 0.8 |

Frozen, Score:

| Method | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| ce | 56.8 ± 0.7 | 53.7 ± 0.9 | 49.9 ± 0.3 | 53.2 ± 0.4 | 55.5 ± 0.3 |
| patchmix | 57.4 ± 0.8 | 54.6 ± 1.2 | 51.3 ± 0.8 | 53.7 ± 0.7 | 56.4 ± 0.3 |
| pafa | 56.4 ± 0.4 | 56.2 ± 0.5 | 48.1 ± 0.3 | 53.3 ± 0.4 | 54.9 ± 0.1 |
| qlung | 56.5 ± 0.5 | 53.9 ± 0.9 | 50.0 ± 1.5 | 52.5 ± 0.6 | 55.4 ± 0.3 |
| add_rsc | 58.1 ± 1.1 | 54.0 ± 0.9 | 51.1 ± 1.2 | 53.8 ± 0.7 | 57.3 ± 0.7 |
| ce_pafa_cfg | 56.7 ± 0.4 | 55.8 ± 1.3 | 48.0 ± 0.4 | 52.9 ± 0.2 | 54.8 ± 0.1 |

Fine-tuned, worst-device Score:

| Method | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| ce | 51.6 ± 6.0 | 51.6 ± 5.3 | 36.8 ± 2.0 | 53.7 ± 3.2 | 53.3 ± 3.1 |
| patchmix | 47.2 ± 2.4 | 47.6 ± 2.5 | 38.4 ± 3.3 | 55.2 ± 4.1 | 51.2 ± 6.5 |
| patchmix_cl | 49.3 ± 1.3 | 51.5 ± 5.4 | 37.8 ± 1.5 | 57.8 ± 3.8 | 49.4 ± 1.7 |
| pafa | 52.9 ± 2.7 | 47.5 ± 4.4 | 38.2 ± 1.4 | 60.0 ± 0.9 | 55.3 ± 2.7 |
| sg_scl | 50.3 ± 1.5 | 52.7 ± 5.1 | 36.5 ± 2.4 | 56.2 ± 2.9 | 53.3 ± 5.2 |
| qlung | 49.3 ± 4.3 | 51.8 ± 4.9 | 37.8 ± 2.6 | 55.9 ± 3.2 | 51.5 ± 3.3 |
| add_rsc | 52.5 ± 1.4 | 53.3 ± 3.3 | 39.9 ± 4.4 | 57.6 ± 1.6 | 49.4 ± 4.4 |
| ce_pafa_cfg | 52.2 ± 1.8 | 45.6 ± 6.1 | 40.2 ± 0.6 | 58.5 ± 2.7 | 49.7 ± 3.1 |

Top results (best epoch on test):

| Rank | Cell | Score | Worst-device Score | Score at last epoch |
|---|---|---|---|---|
| 1 | beats+ft+patchmix_cl | 63.63 ± 0.8 | 57.8 ± 3.8 | 59.7 ± 1.3 |
| 2 | beats+ft+sg_scl | 63.24 ± 0.7 | 56.2 ± 2.9 | 61.1 ± 1.3 |
| 3 | beats+ft+pafa | 63.23 ± 0.6 | 60.0 ± 0.9 | 61.7 ± 0.7 |
| 4 | beats+ft+ce_pafa_cfg | 63.05 ± 1.1 | 58.5 ± 2.7 | 60.1 ± 1.3 |
| 5 | beats+ft+qlung | 62.67 ± 0.5 | 55.9 ± 3.2 | 59.3 ± 0.8 |
| (7) | beats+ft+ce | 62.16 ± 1.1 | 53.7 ± 3.2 | 59.2 ± 1.5 |
| best non-BEATs | htsat+ft+patchmix_cl | 62.05 ± 1.5 | 51.5 ± 5.4 | 59.1 ± 1.0 |

- [Fact] Ranks 2-4 are within 0.2 of each other, far below one seed SD; the top 5 span 1.0 point. BEATs CE is 62.16, so
  the best method gains 1.5 over CE on the best epoch, about 1.4 seed SD.
- [Fact] Best per encoder (best epoch): BEATs patchmix_cl 63.6, HTS-AT patchmix_cl 62.1, DASS patchmix 61.7,
  AST ce 61.3, OPERA-CT ce 52.6.
- [Fact] Best-epoch selection adds 0.6-4.4 points over the last epoch (smallest: dass+frozen+ce_pafa_cfg; largest:
  beats+frozen+add_rsc); among fine-tuned cells Patch-Mix CL on BEATs gains 3.9.
- [Interpretation] The best-epoch ranking and the last-epoch ranking agree on BEATs as the encoder but not on the method
  (Patch-Mix CL first vs PAFA first); the method order inside BEATs is within noise either way.

## Frozen cells

Score at the last epoch:

| Method | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| ce | 55.1 ± 0.2 | 51.8 ± 0.4 | 46.8 ± 0.5 | 51.5 ± 0.2 | 54.4 ± 0.3 |
| patchmix | 56.4 ± 0.4 | 50.5 ± 0.4 | 49.1 ± 0.5 | 51.8 ± 0.3 | 54.8 ± 0.2 |
| pafa | 55.3 ± 0.5 | 52.9 ± 0.3 | 46.6 ± 0.2 | 50.1 ± 0.2 | 54.2 ± 0.1 |
| qlung | 55.4 ± 0.6 | 51.9 ± 0.4 | 47.3 ± 0.5 | 51.6 ± 0.5 | 54.6 ± 0.2 |
| add_rsc | 57.2 ± 1.5 | 52.7 ± 0.5 | 48.0 ± 0.9 | 49.4 ± 1.8 | 56.3 ± 0.5 |
| ce_pafa_cfg | 55.2 ± 0.4 | 52.8 ± 0.3 | 46.6 ± 0.2 | 50.0 ± 0.1 | 54.2 ± 0.2 |

- [Fact] Fine-tuning beats freezing (CE, paired): AST +2.5, DASS +3.7, OPERA-CT +4.7, HTS-AT +7.0, BEATs +7.7.
- [Fact] Frozen PAFA equals frozen ce_pafa_cfg within 0.2 on every encoder; on seed 0 the predictions agree on
  99.6-99.9% of cycles (AST, BEATs, OPERA-CT; 88% on HTS-AT). PAFA's losses train only the projector, which the classifier
  does not use; with a frozen encoder they cannot change the classifier. [Interpretation] Like Patch-Mix CL and SG-SCL,
  PAFA reduces to CE when frozen; the plan's reason for keeping it frozen was wrong, and its 25 frozen units are duplicates
  of the control.
- [Fact] Frozen OPERA-CT nearly ignores abnormal sounds: Se 19-25% with CE, Patch-Mix, PAFA and QLung.

## Conformal coverage (LAC, 90%)

Worst-device coverage at the last epoch (pooled coverage: 89.1-90.2% fine-tuned, 88.2-90.5% frozen):

| Mode | AST | HTS-AT | OPERA-CT | BEATs | DASS |
|---|---|---|---|---|---|
| ft, range over 8 methods | 86.0-87.3 | 86.3-88.4 | 72.1-79.8 | 85.4-87.0 | 85.9-88.3 |
| frozen CE | 81.9 | 69.2 | 88.1 | 73.4 | 70.8 |
| frozen Patch-Mix | 88.2 | 77.2 | 87.5 | 86.4 | 81.3 |

- [Fact] Pooled coverage meets the 90% target; the worst device (Littmann 3200 in 343/350 units) falls short by
  2-23 points. Mean set size is 2.1-2.9 classes of 4.
- [Fact] Coverage by QLung quality tercile ranges 86-94%. The low-quality third is over-covered in the OPERA-CT, HTS-AT and
  frozen cells (up to 94.1%); no tercile falls below 85.9%.
- [Interpretation] Exchangeability holds for the pooled test set but not per device: the device-level gap is the target of
  H2. Patch-Mix on a frozen encoder narrows it most (69 to 77, 73 to 86 on HTS-AT, BEATs).

## Comparison with published numbers (best on test, optimistic)

| Method on AST | Here | Published |
|---|---|---|
| Patch-Mix CL | 61.1 ± 0.4 | 62.37 |
| QLung | 61.0 ± 0.9 | 62.01 |
| ADD-RSC | 59.6 ± 0.5 | 65.53 |

- [Fact] Patch-Mix CL and QLung land 1.0-1.3 points below the papers. The papers report one best-on-test number; seed
  counts are not stated.
- [Fact] ADD-RSC is 6 points below. Expected: the reference resizes AST input to 1024 x 256 with its own AST (patch
  stride 16); here the denoising layer runs on each encoder's own input.
- [Fact] BEATs reaches 61.4-63.6 best on test, above the AST paper numbers except ADD-RSC.

## Incidents and their handling

- QLung: the reference quality score is NaN for 71 AKGC417L cycles (leading digital silence); our port leaves silent
  frames out. No NaN in any unit.
- SG-SCL: the reference reversal schedule (weight 1 from step 0) collapsed BEATs in 3 of 4 seeds; all SG-SCL cells use
  the DANN ramp (`grl_schedule: progress`). Old results in `outputs/train/grid-v1-sgscl-ref/`.
- Out of memory on A at launch (15/10): fixed by budgeting summed reservations; no out-of-memory since.
- Power cut on both machines 17/10 10:43 (sweeps down 9.5 h). 13 units in total had been resumed from `last.pth`
  (resume does not restore the RNG state); all 13 were rerun without interruption and swapped in. Resumed vs rerun Score:
  mean difference +0.16, mean absolute 0.61, largest 2.5 (ast+ft+patchmix_cl seed 4); resumed results kept in
  `outputs|checkpoints/train/grid-v1-resumed/`. 5 of the 13 reruns ran on the other machine's GPU.
- Train loss and accuracy per epoch are in `report.json` for 246 of 350 units (added 16/10); earlier units have them in
  their job logs only.

## Limitations

- One split (official 60/40) and 5 seeds: a 1-point difference is within seed noise for most cells.
- "Last epoch" is the fixed reference-schedule epoch, not a validated epoch; "best on test" is test-selected.
- Each method keeps its reference settings, so cells differ in batch size, cycle length and SpecAugment as well as in
  the method (ce_pafa_cfg isolates this only for PAFA).
- Worst-device numbers rest on the Littmann 3200 test set (461 cycles).
- GPU numerics are not bit-reproducible (`cudnn.benchmark`); the rerun differences above bound this together with RNG.

## LODO scope: options

LODO = train with one device held out (Meditron, LittC2SE, Litt3200, AKGC417L), test on it; recipe and epoch count frozen
from this grid (`--holdout_device`, `--selection fixed`). 4 folds x 5 seeds = 20 units per cell. Hours: measured grid
throughput, both machines in parallel; an upper bound, since each fold trains on fewer cycles.

| Option | Cells | Units | Wall time |
|---|---|---|---|
| A. Top 3 + CE (BEATs) | beats+ft: patchmix_cl, sg_scl, pafa, ce | 80 | about 30 h |
| B. Top 5 + CE (BEATs) | A + ce_pafa_cfg, qlung | 120 | about 42 h |
| C. Encoder spread | best cell + CE per encoder: beats patchmix_cl/ce, htsat patchmix_cl/ce, dass patchmix/ce, ast ce, opera_ct ce | 160 | about 69 h |
| D. B + C | top 5 BEATs + BEATs CE + the C cells of the other encoders | 240 | about 94 h |
| add-on: frozen CE per encoder | 5 frozen cells | +100 | about +10 h |

- A answers "does the best recipe hold up on an unseen device"; with CE it gives the method effect under device shift.
- B adds the two other cells within 0.6 of the top; the top 5 differ by less than one seed SD on the official split.
- C answers "is the encoder ranking stable under device shift" (BEATs leads by 1.5-11 points here).
- The frozen add-on is cheap and has the largest device coverage gaps (67-82%), the clearest H2 test cases.
- Every option needs the SC arm on top for H2 (trained SC doubles the units of the cells it is added to; test-time SC is
  inference only). The AKGC417L fold is class-mix dominated and is reported flagged, with a macro mean without it.

## Decisions needed

1. Rule for "best" (top cells go to the 80/20 split, 5 folds x 5 seeds):
   - (a) last-epoch Score: beats+ft+pafa, beats+ft+sg_scl, beats+ft+ce_pafa_cfg;
   - (b) best epoch on test: beats+ft+patchmix_cl, beats+ft+sg_scl, beats+ft+pafa (top 5 adds ce_pafa_cfg, qlung);
   - (c) worst-device Score: beats+ft+pafa, beats+ft+ce_pafa_cfg, beats+ft+add_rsc;
   - (d) one cell per encoder family, to keep encoder diversity.
2. LODO scope: option A, B, C or D above, with or without the frozen add-on.
3. Drop frozen PAFA from later stages (identical to its control).
