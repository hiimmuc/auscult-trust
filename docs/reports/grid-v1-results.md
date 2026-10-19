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

## Combining methods

What each method changes (BEATs fine-tuned, last epoch, per-class recall in %: normal / crackle / wheeze / both):

| Cell | Score | Worst-device Score | Recall N / C / W / B | Effect |
|---|---|---|---|---|
| beats+ft+ce | 59.2 ± 1.5 | 50.4 | 66 / 62 / 43 / 36 | baseline |
| beats+ft+pafa | 61.7 ± 0.7 | 58.9 | 73 / 60 / 39 / 35 | Sp up, abnormal recall slightly down; best worst device |
| beats+ft+sg_scl | 61.1 ± 1.3 | 52.9 | 72 / 61 / 39 / 34 | same direction as PAFA |
| beats+ft+patchmix_cl | 59.7 ± 1.3 | 52.1 | 64 / 65 / 46 / 38 | Se up (all three abnormal classes), Sp down |
| beats+ft+ce_pafa_cfg | 60.1 ± 1.3 | 55.0 | 66 / 65 / 43 / 31 | crackle up, "both" down |
| beats+ft+add_rsc | 58.9 ± 0.9 | 53.1 | 71 / 60 / 32 / 29 | Sp up, wheeze and "both" down |

- [Fact] The methods move along the Sp/Se trade-off in different directions: PAFA and SG-SCL raise normal recall, Patch-Mix
  CL raises abnormal recall. Two BEATs methods disagree on 18-28% of test cycles (seed 0); cells of different encoders
  on 28-33%.
- [Interpretation] The errors are partly independent, so averaging the models' softmax should help.

Ensembles (mean of the softmax at the last epoch over the cells listed, same seed; mean ± SD over the 5 seeds). Sets
marked [test-picked] were chosen after seeing this table and are optimistic:

| Ensemble | Cells | Score | Worst-device Score | 2c Score |
|---|---|---|---|---|
| Single best cell, beats+ft+pafa | 1 | 61.7 ± 0.7 | 58.9 | 69.3 |
| All 8 fine-tuned methods on BEATs | 8 | 62.0 ± 0.2 | 55.2 | 69.6 |
| CE on AST, HTS-AT, BEATs, DASS | 4 | 60.4 ± 0.8 | 48.6 | 67.5 |
| All 8 methods x those 4 encoders | 32 | 61.7 ± 0.4 | 50.9 | 68.4 |
| SG-SCL on those 4 encoders | 4 | 62.3 ± 1.1 | 51.6 | 68.9 |
| [test-picked] BEATs PAFA + SG-SCL + Patch-Mix CL | 3 | 62.5 ± 0.5 | 57.9 | 70.1 |
| [test-picked] chosen LODO set: BEATs Patch-Mix CL, SG-SCL, PAFA; HTS-AT Patch-Mix CL; DASS Patch-Mix; AST CE | 6 | 63.0 ± 0.4 | 56.8 | 69.9 |
| 5 seeds of beats+ft+pafa | 1 | 62.7 (one number) | | |

- [Fact] Averaging all 8 BEATs methods, a rule that picks no cell on test, gives 62.0 ± 0.2: +2.8 over BEATs CE, with the
  seed SD falling from 1.5 to 0.2. It does not beat PAFA alone on the worst device (55.2 vs 58.9).
- [Fact] The 6-cell set chosen for LODO reaches 63.0 ± 0.4 at the last epoch, about the best single cell's best-epoch
  Score (63.6), but its cells were picked on test.
- [Fact] Adding the weaker encoders to the BEATs ensemble does not help (32 cells 61.7 vs 8 BEATs cells 62.0).
- [Fact] Averaging the last 10 epochs of one model changes Score by 0.0-0.2: the gain comes from different models, not
  from different epochs.

Ways to use the methods together, best first:

1. Ensemble of trained cells (no new training). Fix the rule before the next stage (for example: mean softmax of the
   BEATs Patch-Mix CL, SG-SCL and PAFA cells, plus the best cell of HTS-AT, DASS and AST) and test it on the 80/20 folds
   and in LODO, where these cells are trained anyway. Cost: inference only. Conformal sets are built on the ensemble
   softmax the same way.
2. Distill that ensemble into one BEATs model (ensemble knowledge distillation, already a planned step; reference code in
   `../repos/rsc-ensemble-kd`). Keeps one model at inference time. Cost: one training per seed and fold.
3. One model with the complementary losses together: PAFA's patient alignment (raises Sp and the worst device) plus
   Patch-Mix or Patch-Mix CL (raises Se). Needs code: the methods are exclusive branches of `engine.py` today. Loss
   weights must be chosen by grouped CV on training patients, not on test. Cost: CV screen, then the usual seeds.
4. Move along the Sp/Se trade-off: one bias on the normal-class logit chosen on calibration patients, to balance Sp and
   Se of a single model. Cheap, but it needs calibration patients the official split does not have (LODO and the 80/20
   folds have them).

Not worth pursuing from this grid: frozen encoders, OPERA-CT, ensembles over epochs, and ADD-RSC in this port.

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
| E. Chosen set: top 3 BEATs + best of AST, HTS-AT, DASS | beats patchmix_cl/sg_scl/pafa, ast ce, htsat patchmix_cl, dass patchmix | 120 | about 44 h (49 h with BEATs CE) |

- A answers "does the best recipe hold up on an unseen device"; with CE it gives the method effect under device shift.
- B adds the two other cells within 0.6 of the top; the top 5 differ by less than one seed SD on the official split.
- C answers "is the encoder ranking stable under device shift" (BEATs leads by 1.5-11 points here).
- The frozen add-on is cheap and has the largest device coverage gaps (67-82%), the clearest H2 test cases.
- Option E in detail. Per-unit wall time (measured, both machines' mean): BEATs Patch-Mix CL 1.08 h, SG-SCL 0.78 h,
  PAFA 0.55 h (100 epochs), AST CE 0.44 h, HTS-AT Patch-Mix CL 0.28 h, DASS Patch-Mix 2.06 h. LODO folds train on
  1,950-4,773 cycles (official split: 4,142) and test on 481-4,216: about 0.85 x the grid cost on average (AKGC417L 0.70,
  Meditron 0.83, LittC2SE and Litt3200 0.94). 120 units x 0.85 = about 88 machine-hours, 44 h on both machines; DASS
  is 40% of it. Adding BEATs CE as the method baseline: +5 h. The "best per encoder" uses the best-epoch rule; by the
  last-epoch rule it would be AST ce_pafa_cfg and HTS-AT sg_scl (DASS unchanged), within 0.4 of the cells above.
- Option E also gives the 6-cell ensemble above in every fold at no extra cost.
- Every option needs the SC arm on top for H2 (trained SC doubles the units of the cells it is added to; test-time SC is
  inference only). The AKGC417L fold is class-mix dominated and is reported flagged, with a macro mean without it.

## Decisions needed

1. Rule for "best" (top cells go to the 80/20 split, 5 folds x 5 seeds):
   - (a) last-epoch Score: beats+ft+pafa, beats+ft+sg_scl, beats+ft+ce_pafa_cfg;
   - (b) best epoch on test: beats+ft+patchmix_cl, beats+ft+sg_scl, beats+ft+pafa (top 5 adds ce_pafa_cfg, qlung);
   - (c) worst-device Score: beats+ft+pafa, beats+ft+ce_pafa_cfg, beats+ft+add_rsc;
   - (d) one cell per encoder family, to keep encoder diversity.
2. LODO scope: option E (chosen), with or without BEATs CE and the frozen add-on.
3. Whether to pre-register the ensemble rule (combining item 1) before LODO.
4. Drop frozen PAFA from later stages (identical to its control).

## Appendix: full metrics

All cells, mean ± SD over seeds 0-4, pooled over devices, in %. 4-class: Sp = normal recall, Se = recall of the three
abnormal classes, Score = (Sp + Se) / 2, HS = harmonic mean. 2-class (normal vs abnormal) view of the same predictions:
2-class Sp equals 4-class Sp by construction, so only 2-class Se, Score and HS are listed.

### Fine-tuned, last epoch

| Cell | 4c Sp | 4c Se | 4c Score | 4c HS | 2c Se | 2c Score | 2c HS |
|---|---|---|---|---|---|---|---|
| ast+ft+ce | 68.4 ± 2.3 | 46.8 ± 2.5 | 57.6 ± 1.0 | 55.5 ± 1.4 | 64.5 ± 3.6 | 66.5 ± 1.2 | 66.3 ± 1.3 |
| ast+ft+patchmix | 72.6 ± 2.4 | 42.3 ± 1.3 | 57.5 ± 0.8 | 53.4 ± 0.7 | 57.3 ± 1.6 | 65.0 ± 0.8 | 64.0 ± 0.7 |
| ast+ft+patchmix_cl | 71.7 ± 4.1 | 44.7 ± 1.7 | 58.2 ± 1.3 | 55.0 ± 0.5 | 60.7 ± 2.5 | 66.2 ± 0.9 | 65.7 ± 0.4 |
| ast+ft+pafa | 69.3 ± 1.0 | 47.4 ± 1.2 | 58.3 ± 0.9 | 56.3 ± 1.0 | 63.2 ± 1.7 | 66.2 ± 1.1 | 66.1 ± 1.1 |
| ast+ft+sg_scl | 71.6 ± 4.3 | 45.7 ± 3.3 | 58.7 ± 0.6 | 55.6 ± 1.1 | 62.1 ± 5.2 | 66.9 ± 0.8 | 66.3 ± 1.2 |
| ast+ft+qlung | 70.4 ± 2.0 | 45.9 ± 1.4 | 58.2 ± 0.9 | 55.5 ± 0.9 | 63.0 ± 2.5 | 66.7 ± 0.7 | 66.5 ± 0.8 |
| ast+ft+add_rsc | 68.4 ± 1.3 | 45.4 ± 1.1 | 56.9 ± 0.6 | 54.6 ± 0.7 | 63.1 ± 1.3 | 65.8 ± 0.6 | 65.7 ± 0.6 |
| ast+ft+ce_pafa_cfg | 71.8 ± 2.4 | 46.2 ± 1.3 | 59.0 ± 1.0 | 56.2 ± 0.9 | 62.0 ± 1.7 | 66.9 ± 0.7 | 66.5 ± 0.7 |
| htsat+ft+ce | 70.0 ± 3.0 | 47.6 ± 0.8 | 58.8 ± 1.4 | 56.6 ± 0.9 | 62.4 ± 1.8 | 66.2 ± 1.2 | 65.9 ± 1.1 |
| htsat+ft+patchmix | 70.0 ± 3.2 | 47.8 ± 1.9 | 58.9 ± 1.2 | 56.7 ± 1.0 | 63.0 ± 2.1 | 66.5 ± 1.2 | 66.2 ± 1.1 |
| htsat+ft+patchmix_cl | 69.9 ± 2.7 | 48.3 ± 1.2 | 59.1 ± 1.0 | 57.1 ± 0.5 | 63.4 ± 1.9 | 66.7 ± 0.8 | 66.5 ± 0.7 |
| htsat+ft+pafa | 67.0 ± 3.1 | 46.2 ± 0.9 | 56.6 ± 1.3 | 54.7 ± 0.8 | 62.5 ± 1.8 | 64.8 ± 0.7 | 64.6 ± 0.6 |
| htsat+ft+sg_scl | 72.4 ± 2.2 | 46.9 ± 1.2 | 59.6 ± 0.5 | 56.9 ± 0.2 | 62.3 ± 1.5 | 67.3 ± 0.6 | 66.9 ± 0.5 |
| htsat+ft+qlung | 69.8 ± 2.7 | 48.8 ± 1.5 | 59.3 ± 1.4 | 57.4 ± 1.2 | 63.9 ± 1.5 | 66.8 ± 1.0 | 66.7 ± 0.8 |
| htsat+ft+add_rsc | 66.9 ± 2.1 | 48.6 ± 2.2 | 57.8 ± 1.0 | 56.3 ± 1.2 | 64.0 ± 2.4 | 65.4 ± 0.7 | 65.4 ± 0.8 |
| htsat+ft+ce_pafa_cfg | 67.0 ± 4.3 | 45.5 ± 2.8 | 56.3 ± 0.9 | 54.1 ± 1.0 | 62.1 ± 2.9 | 64.5 ± 0.8 | 64.3 ± 0.5 |
| opera_ct+ft+ce | 65.4 ± 2.8 | 37.6 ± 0.8 | 51.5 ± 1.2 | 47.7 ± 0.5 | 52.6 ± 1.1 | 59.0 ± 1.0 | 58.3 ± 0.6 |
| opera_ct+ft+patchmix | 64.5 ± 2.3 | 38.2 ± 1.7 | 51.3 ± 0.5 | 47.9 ± 0.8 | 52.8 ± 2.9 | 58.6 ± 0.7 | 58.0 ± 1.1 |
| opera_ct+ft+patchmix_cl | 65.4 ± 0.9 | 38.0 ± 1.4 | 51.7 ± 0.7 | 48.0 ± 1.1 | 53.8 ± 1.3 | 59.6 ± 0.6 | 59.0 ± 0.7 |
| opera_ct+ft+pafa | 61.5 ± 1.1 | 37.2 ± 0.6 | 49.3 ± 0.3 | 46.4 ± 0.3 | 54.2 ± 1.6 | 57.8 ± 0.8 | 57.6 ± 0.8 |
| opera_ct+ft+sg_scl | 64.6 ± 1.2 | 38.0 ± 1.2 | 51.3 ± 0.8 | 47.8 ± 1.0 | 54.0 ± 1.3 | 59.3 ± 0.5 | 58.8 ± 0.6 |
| opera_ct+ft+qlung | 64.9 ± 1.3 | 37.1 ± 1.8 | 51.0 ± 0.9 | 47.1 ± 1.4 | 53.6 ± 1.9 | 59.2 ± 0.8 | 58.7 ± 1.0 |
| opera_ct+ft+add_rsc | 62.7 ± 1.4 | 37.2 ± 0.9 | 50.0 ± 0.5 | 46.7 ± 0.5 | 53.3 ± 1.6 | 58.0 ± 0.3 | 57.6 ± 0.5 |
| opera_ct+ft+ce_pafa_cfg | 62.1 ± 2.0 | 36.8 ± 1.8 | 49.5 ± 0.9 | 46.2 ± 1.2 | 55.0 ± 2.3 | 58.6 ± 1.0 | 58.3 ± 1.0 |
| beats+ft+ce | 65.7 ± 3.7 | 52.7 ± 1.3 | 59.2 ± 1.5 | 58.4 ± 1.1 | 69.7 ± 1.4 | 67.7 ± 1.5 | 67.6 ± 1.5 |
| beats+ft+patchmix | 64.6 ± 2.9 | 54.6 ± 1.7 | 59.6 ± 1.1 | 59.1 ± 0.9 | 71.0 ± 1.8 | 67.8 ± 1.4 | 67.6 ± 1.5 |
| beats+ft+patchmix_cl | 64.2 ± 2.6 | 55.2 ± 1.8 | 59.7 ± 1.3 | 59.3 ± 1.2 | 72.4 ± 1.5 | 68.3 ± 0.9 | 68.0 ± 1.0 |
| beats+ft+pafa | 73.2 ± 2.7 | 50.2 ± 2.9 | 61.7 ± 0.7 | 59.5 ± 1.3 | 65.4 ± 3.8 | 69.3 ± 1.1 | 69.0 ± 1.4 |
| beats+ft+sg_scl | 71.9 ± 4.1 | 50.2 ± 1.5 | 61.1 ± 1.3 | 59.1 ± 0.5 | 65.8 ± 2.3 | 68.9 ± 1.0 | 68.6 ± 0.8 |
| beats+ft+qlung | 67.4 ± 2.1 | 51.2 ± 1.4 | 59.3 ± 0.8 | 58.2 ± 0.7 | 67.5 ± 1.6 | 67.4 ± 0.7 | 67.4 ± 0.7 |
| beats+ft+add_rsc | 70.9 ± 3.2 | 46.9 ± 1.6 | 58.9 ± 0.9 | 56.4 ± 0.5 | 63.4 ± 2.1 | 67.1 ± 0.6 | 66.9 ± 0.4 |
| beats+ft+ce_pafa_cfg | 66.4 ± 3.3 | 53.9 ± 2.2 | 60.1 ± 1.3 | 59.4 ± 1.2 | 70.0 ± 2.4 | 68.2 ± 0.8 | 68.1 ± 0.9 |
| dass+ft+ce | 68.1 ± 3.2 | 48.1 ± 1.4 | 58.1 ± 1.1 | 56.3 ± 0.7 | 62.9 ± 1.9 | 65.5 ± 0.9 | 65.3 ± 0.8 |
| dass+ft+patchmix | 70.6 ± 2.8 | 48.8 ± 1.2 | 59.7 ± 0.9 | 57.7 ± 0.4 | 61.8 ± 1.9 | 66.2 ± 0.8 | 65.9 ± 0.7 |
| dass+ft+patchmix_cl | 65.5 ± 1.9 | 50.3 ± 1.2 | 57.9 ± 0.5 | 56.8 ± 0.3 | 66.0 ± 0.9 | 65.7 ± 0.8 | 65.7 ± 0.8 |
| dass+ft+pafa | 68.6 ± 1.0 | 49.9 ± 1.5 | 59.2 ± 0.5 | 57.7 ± 0.8 | 64.9 ± 1.3 | 66.8 ± 0.6 | 66.7 ± 0.7 |
| dass+ft+sg_scl | 70.0 ± 2.9 | 48.6 ± 2.1 | 59.3 ± 1.0 | 57.3 ± 0.9 | 62.7 ± 2.4 | 66.3 ± 0.6 | 66.1 ± 0.5 |
| dass+ft+qlung | 68.5 ± 1.9 | 47.7 ± 0.7 | 58.1 ± 1.0 | 56.3 ± 0.8 | 63.3 ± 1.7 | 65.9 ± 0.6 | 65.8 ± 0.6 |
| dass+ft+add_rsc | 66.1 ± 1.6 | 48.8 ± 1.3 | 57.5 ± 1.2 | 56.1 ± 1.2 | 64.5 ± 1.3 | 65.3 ± 1.1 | 65.3 ± 1.1 |
| dass+ft+ce_pafa_cfg | 65.2 ± 2.8 | 50.0 ± 1.4 | 57.6 ± 0.8 | 56.5 ± 0.5 | 65.7 ± 2.3 | 65.5 ± 0.4 | 65.4 ± 0.4 |

### Frozen, last epoch

| Cell | 4c Sp | 4c Se | 4c Score | 4c HS | 2c Se | 2c Score | 2c HS |
|---|---|---|---|---|---|---|---|
| ast+frozen+ce | 64.3 ± 0.5 | 45.9 ± 0.4 | 55.1 ± 0.2 | 53.6 ± 0.2 | 62.2 ± 0.4 | 63.2 ± 0.2 | 63.2 ± 0.2 |
| ast+frozen+patchmix | 70.5 ± 1.1 | 42.3 ± 0.7 | 56.4 ± 0.4 | 52.9 ± 0.3 | 54.9 ± 1.0 | 62.7 ± 0.3 | 61.7 ± 0.4 |
| ast+frozen+pafa | 63.8 ± 1.3 | 46.8 ± 0.5 | 55.3 ± 0.5 | 54.0 ± 0.4 | 61.6 ± 0.5 | 62.7 ± 0.5 | 62.7 ± 0.4 |
| ast+frozen+qlung | 65.4 ± 1.3 | 45.3 ± 1.0 | 55.4 ± 0.6 | 53.5 ± 0.6 | 61.3 ± 1.0 | 63.4 ± 0.5 | 63.3 ± 0.5 |
| ast+frozen+add_rsc | 70.7 ± 2.4 | 43.7 ± 1.1 | 57.2 ± 1.5 | 54.0 ± 1.3 | 59.5 ± 1.0 | 65.1 ± 1.6 | 64.6 ± 1.4 |
| ast+frozen+ce_pafa_cfg | 63.8 ± 1.0 | 46.7 ± 0.5 | 55.2 ± 0.4 | 53.9 ± 0.3 | 61.5 ± 0.5 | 62.7 ± 0.4 | 62.6 ± 0.4 |
| htsat+frozen+ce | 61.7 ± 0.7 | 41.9 ± 0.6 | 51.8 ± 0.4 | 49.9 ± 0.4 | 58.6 ± 0.6 | 60.2 ± 0.4 | 60.1 ± 0.4 |
| htsat+frozen+patchmix | 61.0 ± 1.3 | 40.0 ± 1.2 | 50.5 ± 0.4 | 48.3 ± 0.6 | 57.0 ± 1.1 | 59.0 ± 0.3 | 58.9 ± 0.3 |
| htsat+frozen+pafa | 63.1 ± 0.4 | 42.8 ± 0.2 | 52.9 ± 0.3 | 51.0 ± 0.3 | 60.0 ± 0.4 | 61.6 ± 0.3 | 61.5 ± 0.3 |
| htsat+frozen+qlung | 61.9 ± 0.7 | 41.8 ± 0.9 | 51.9 ± 0.4 | 49.9 ± 0.6 | 58.4 ± 1.5 | 60.2 ± 0.5 | 60.1 ± 0.5 |
| htsat+frozen+add_rsc | 61.2 ± 1.8 | 44.3 ± 2.0 | 52.7 ± 0.5 | 51.3 ± 0.8 | 61.8 ± 2.1 | 61.5 ± 0.8 | 61.5 ± 0.8 |
| htsat+frozen+ce_pafa_cfg | 62.9 ± 0.6 | 42.7 ± 0.3 | 52.8 ± 0.3 | 50.9 ± 0.3 | 60.1 ± 0.4 | 61.5 ± 0.1 | 61.5 ± 0.1 |
| opera_ct+frozen+ce | 68.6 ± 1.8 | 25.0 ± 0.9 | 46.8 ± 0.5 | 36.6 ± 0.8 | 36.1 ± 1.9 | 52.3 ± 0.4 | 47.2 ± 1.3 |
| opera_ct+frozen+patchmix | 79.6 ± 2.3 | 18.6 ± 1.8 | 49.1 ± 0.5 | 30.1 ± 2.2 | 25.7 ± 2.0 | 52.6 ± 0.5 | 38.7 ± 2.1 |
| opera_ct+frozen+pafa | 68.2 ± 0.4 | 24.9 ± 0.5 | 46.6 ± 0.2 | 36.5 ± 0.5 | 39.3 ± 0.5 | 53.8 ± 0.2 | 49.9 ± 0.4 |
| opera_ct+frozen+qlung | 71.4 ± 1.4 | 23.3 ± 0.7 | 47.3 ± 0.5 | 35.1 ± 0.7 | 33.4 ± 1.0 | 52.4 ± 0.8 | 45.5 ± 1.0 |
| opera_ct+frozen+add_rsc | 61.9 ± 1.1 | 34.1 ± 2.4 | 48.0 ± 0.9 | 44.0 ± 1.8 | 52.7 ± 1.7 | 57.3 ± 0.7 | 56.9 ± 0.8 |
| opera_ct+frozen+ce_pafa_cfg | 68.4 ± 0.6 | 24.9 ± 0.6 | 46.6 ± 0.2 | 36.5 ± 0.6 | 39.2 ± 0.7 | 53.8 ± 0.2 | 49.8 ± 0.4 |
| beats+frozen+ce | 61.7 ± 0.4 | 41.2 ± 0.6 | 51.5 ± 0.2 | 49.4 ± 0.3 | 58.7 ± 0.7 | 60.2 ± 0.2 | 60.2 ± 0.2 |
| beats+frozen+patchmix | 67.3 ± 1.4 | 36.3 ± 1.2 | 51.8 ± 0.3 | 47.2 ± 0.7 | 50.7 ± 1.8 | 59.0 ± 0.4 | 57.8 ± 0.7 |
| beats+frozen+pafa | 59.5 ± 0.3 | 40.7 ± 0.2 | 50.1 ± 0.2 | 48.3 ± 0.2 | 58.9 ± 0.2 | 59.2 ± 0.1 | 59.2 ± 0.1 |
| beats+frozen+qlung | 61.4 ± 1.5 | 41.8 ± 1.5 | 51.6 ± 0.5 | 49.7 ± 0.7 | 59.4 ± 1.9 | 60.4 ± 0.4 | 60.3 ± 0.5 |
| beats+frozen+add_rsc | 61.6 ± 2.8 | 37.3 ± 1.8 | 49.4 ± 1.8 | 46.4 ± 1.8 | 57.0 ± 1.2 | 59.3 ± 1.4 | 59.2 ± 1.3 |
| beats+frozen+ce_pafa_cfg | 59.4 ± 0.2 | 40.6 ± 0.2 | 50.0 ± 0.1 | 48.3 ± 0.1 | 59.0 ± 0.2 | 59.2 ± 0.1 | 59.2 ± 0.1 |
| dass+frozen+ce | 64.8 ± 0.6 | 44.0 ± 0.2 | 54.4 ± 0.3 | 52.4 ± 0.2 | 58.1 ± 0.2 | 61.4 ± 0.2 | 61.3 ± 0.2 |
| dass+frozen+patchmix | 66.4 ± 1.1 | 43.2 ± 0.9 | 54.8 ± 0.2 | 52.4 ± 0.4 | 56.6 ± 1.3 | 61.5 ± 0.3 | 61.1 ± 0.4 |
| dass+frozen+pafa | 64.0 ± 0.3 | 44.5 ± 0.1 | 54.2 ± 0.1 | 52.5 ± 0.1 | 58.5 ± 0.1 | 61.2 ± 0.2 | 61.1 ± 0.2 |
| dass+frozen+qlung | 66.5 ± 0.4 | 42.7 ± 0.3 | 54.6 ± 0.2 | 52.0 ± 0.2 | 55.9 ± 0.2 | 61.2 ± 0.2 | 60.7 ± 0.1 |
| dass+frozen+add_rsc | 67.2 ± 1.5 | 45.3 ± 1.6 | 56.3 ± 0.5 | 54.1 ± 0.8 | 61.6 ± 2.3 | 64.4 ± 0.9 | 64.2 ± 1.0 |
| dass+frozen+ce_pafa_cfg | 63.9 ± 0.3 | 44.4 ± 0.1 | 54.2 ± 0.2 | 52.4 ± 0.1 | 58.4 ± 0.2 | 61.2 ± 0.2 | 61.0 ± 0.2 |

### Fine-tuned, best epoch on test (optimistic)

| Cell | 4c Sp | 4c Se | 4c Score | 4c HS | 2c Se | 2c Score | 2c HS |
|---|---|---|---|---|---|---|---|
| ast+ft+ce | 77.5 ± 2.0 | 45.1 ± 3.6 | 61.3 ± 1.8 | 56.9 ± 2.7 | 59.9 ± 6.0 | 68.7 ± 2.4 | 67.4 ± 3.3 |
| ast+ft+patchmix | 78.3 ± 2.1 | 40.7 ± 2.3 | 59.5 ± 0.4 | 53.5 ± 1.5 | 53.9 ± 3.2 | 66.1 ± 0.7 | 63.8 ± 1.6 |
| ast+ft+patchmix_cl | 81.7 ± 2.6 | 40.5 ± 2.3 | 61.1 ± 0.4 | 54.0 ± 1.6 | 53.7 ± 2.7 | 67.7 ± 0.3 | 64.7 ± 1.2 |
| ast+ft+pafa | 78.4 ± 2.3 | 44.0 ± 2.6 | 61.2 ± 0.7 | 56.2 ± 1.6 | 58.1 ± 4.7 | 68.2 ± 1.4 | 66.6 ± 2.3 |
| ast+ft+sg_scl | 81.0 ± 3.2 | 41.1 ± 2.7 | 61.1 ± 0.4 | 54.5 ± 1.6 | 54.4 ± 3.7 | 67.7 ± 0.4 | 64.9 ± 1.6 |
| ast+ft+qlung | 81.6 ± 4.0 | 40.3 ± 4.5 | 61.0 ± 0.9 | 53.7 ± 3.3 | 54.4 ± 5.4 | 68.0 ± 1.0 | 65.0 ± 2.7 |
| ast+ft+add_rsc | 78.1 ± 2.1 | 41.0 ± 2.1 | 59.6 ± 0.5 | 53.7 ± 1.3 | 55.7 ± 3.3 | 66.9 ± 0.9 | 65.0 ± 1.6 |
| ast+ft+ce_pafa_cfg | 77.8 ± 5.4 | 43.1 ± 5.6 | 60.5 ± 0.4 | 55.1 ± 3.9 | 55.9 ± 7.8 | 66.9 ± 1.5 | 64.5 ± 4.4 |
| htsat+ft+ce | 78.7 ± 5.0 | 44.4 ± 3.9 | 61.5 ± 1.2 | 56.5 ± 2.1 | 57.9 ± 3.9 | 68.3 ± 1.6 | 66.5 ± 1.6 |
| htsat+ft+patchmix | 74.4 ± 2.8 | 47.3 ± 1.5 | 60.9 ± 0.8 | 57.8 ± 0.4 | 62.2 ± 2.7 | 68.3 ± 0.4 | 67.7 ± 0.5 |
| htsat+ft+patchmix_cl | 77.0 ± 4.6 | 47.1 ± 3.5 | 62.1 ± 1.5 | 58.3 ± 1.9 | 60.9 ± 3.4 | 69.0 ± 1.4 | 67.8 ± 1.4 |
| htsat+ft+pafa | 71.8 ± 3.6 | 46.1 ± 2.6 | 59.0 ± 2.0 | 56.1 ± 2.1 | 60.6 ± 2.7 | 66.2 ± 1.9 | 65.6 ± 1.9 |
| htsat+ft+sg_scl | 76.9 ± 4.8 | 46.7 ± 3.2 | 61.8 ± 1.4 | 58.0 ± 1.4 | 60.5 ± 4.3 | 68.7 ± 1.0 | 67.5 ± 1.1 |
| htsat+ft+qlung | 78.0 ± 6.4 | 44.5 ± 4.0 | 61.3 ± 1.3 | 56.4 ± 1.8 | 58.3 ± 5.7 | 68.2 ± 0.5 | 66.3 ± 1.7 |
| htsat+ft+add_rsc | 75.6 ± 6.1 | 46.9 ± 3.3 | 61.3 ± 1.7 | 57.6 ± 1.4 | 61.2 ± 4.1 | 68.4 ± 1.2 | 67.4 ± 0.7 |
| htsat+ft+ce_pafa_cfg | 72.1 ± 3.9 | 44.5 ± 1.7 | 58.3 ± 1.6 | 54.9 ± 1.1 | 60.1 ± 3.2 | 66.1 ± 1.8 | 65.4 ± 1.8 |
| opera_ct+ft+ce | 70.1 ± 3.8 | 35.0 ± 4.3 | 52.6 ± 0.8 | 46.4 ± 3.4 | 48.1 ± 5.9 | 59.1 ± 1.3 | 56.7 ± 3.4 |
| opera_ct+ft+patchmix | 69.1 ± 5.1 | 35.6 ± 4.6 | 52.4 ± 0.3 | 46.7 ± 3.3 | 49.2 ± 6.5 | 59.1 ± 0.8 | 57.0 ± 3.3 |
| opera_ct+ft+patchmix_cl | 67.9 ± 2.9 | 37.2 ± 2.7 | 52.5 ± 0.4 | 47.9 ± 1.6 | 51.6 ± 4.2 | 59.7 ± 0.9 | 58.5 ± 1.9 |
| opera_ct+ft+pafa | 66.0 ± 2.9 | 34.7 ± 2.5 | 50.4 ± 0.3 | 45.4 ± 1.5 | 49.8 ± 4.2 | 57.9 ± 0.7 | 56.6 ± 1.7 |
| opera_ct+ft+sg_scl | 65.4 ± 4.2 | 39.3 ± 2.6 | 52.4 ± 1.0 | 49.0 ± 0.8 | 54.3 ± 3.8 | 59.8 ± 0.7 | 59.1 ± 0.7 |
| opera_ct+ft+qlung | 71.3 ± 4.9 | 33.7 ± 5.1 | 52.5 ± 0.4 | 45.4 ± 4.1 | 47.3 ± 7.0 | 59.3 ± 1.1 | 56.4 ± 3.9 |
| opera_ct+ft+add_rsc | 70.6 ± 5.6 | 32.2 ± 5.1 | 51.4 ± 0.5 | 43.8 ± 4.1 | 45.7 ± 7.1 | 58.1 ± 0.8 | 54.9 ± 4.1 |
| opera_ct+ft+ce_pafa_cfg | 64.7 ± 2.5 | 36.8 ± 1.6 | 50.7 ± 0.8 | 46.8 ± 0.8 | 53.4 ± 2.3 | 59.0 ± 0.6 | 58.4 ± 0.7 |
| beats+ft+ce | 74.9 ± 4.3 | 49.4 ± 2.3 | 62.2 ± 1.1 | 59.4 ± 0.8 | 65.2 ± 2.8 | 70.1 ± 0.8 | 69.6 ± 0.4 |
| beats+ft+patchmix | 74.0 ± 4.9 | 51.3 ± 3.9 | 62.6 ± 1.1 | 60.4 ± 1.3 | 66.3 ± 4.1 | 70.1 ± 1.1 | 69.7 ± 1.1 |
| beats+ft+patchmix_cl | 76.4 ± 3.8 | 50.9 ± 3.4 | 63.6 ± 0.8 | 60.9 ± 1.4 | 65.5 ± 3.2 | 70.9 ± 1.0 | 70.4 ± 0.9 |
| beats+ft+pafa | 75.5 ± 2.9 | 51.0 ± 2.8 | 63.2 ± 0.6 | 60.8 ± 1.2 | 65.8 ± 3.2 | 70.6 ± 0.4 | 70.2 ± 0.6 |
| beats+ft+sg_scl | 78.6 ± 4.5 | 47.9 ± 3.3 | 63.2 ± 0.7 | 59.3 ± 1.3 | 62.4 ± 3.5 | 70.5 ± 0.7 | 69.4 ± 0.5 |
| beats+ft+qlung | 77.3 ± 3.3 | 48.1 ± 2.5 | 62.7 ± 0.5 | 59.2 ± 0.9 | 62.3 ± 2.5 | 69.8 ± 0.5 | 68.9 ± 0.4 |
| beats+ft+add_rsc | 79.6 ± 4.3 | 43.1 ± 3.1 | 61.4 ± 1.0 | 55.8 ± 1.8 | 57.5 ± 4.3 | 68.6 ± 0.9 | 66.6 ± 1.8 |
| beats+ft+ce_pafa_cfg | 76.1 ± 3.2 | 50.0 ± 2.8 | 63.1 ± 1.1 | 60.3 ± 1.4 | 64.6 ± 3.6 | 70.3 ± 0.9 | 69.7 ± 1.1 |
| dass+ft+ce | 76.0 ± 3.7 | 46.1 ± 3.7 | 61.0 ± 0.7 | 57.2 ± 2.1 | 59.2 ± 5.5 | 67.6 ± 1.1 | 66.3 ± 2.3 |
| dass+ft+patchmix | 76.7 ± 2.4 | 46.8 ± 1.3 | 61.7 ± 1.3 | 58.1 ± 1.1 | 57.9 ± 1.5 | 67.3 ± 1.4 | 66.0 ± 1.3 |
| dass+ft+patchmix_cl | 70.4 ± 2.4 | 50.2 ± 2.1 | 60.3 ± 0.5 | 58.5 ± 0.7 | 64.4 ± 1.5 | 67.4 ± 0.8 | 67.2 ± 0.7 |
| dass+ft+pafa | 73.7 ± 3.6 | 48.2 ± 3.8 | 60.9 ± 0.6 | 58.1 ± 1.6 | 62.4 ± 5.4 | 68.0 ± 1.2 | 67.3 ± 1.6 |
| dass+ft+sg_scl | 75.1 ± 3.5 | 48.1 ± 2.9 | 61.6 ± 0.6 | 58.5 ± 1.3 | 61.5 ± 3.5 | 68.3 ± 0.5 | 67.5 ± 1.0 |
| dass+ft+qlung | 76.8 ± 4.6 | 46.0 ± 4.0 | 61.4 ± 0.9 | 57.3 ± 2.0 | 59.7 ± 4.9 | 68.2 ± 0.8 | 66.9 ± 1.6 |
| dass+ft+add_rsc | 72.3 ± 3.9 | 47.0 ± 3.9 | 59.6 ± 1.6 | 56.8 ± 2.3 | 62.2 ± 4.2 | 67.3 ± 1.8 | 66.7 ± 2.0 |
| dass+ft+ce_pafa_cfg | 70.5 ± 4.1 | 48.0 ± 2.7 | 59.2 ± 0.8 | 56.9 ± 0.8 | 63.0 ± 3.2 | 66.7 ± 0.5 | 66.4 ± 0.1 |

### Frozen, best epoch on test (optimistic)

| Cell | 4c Sp | 4c Se | 4c Score | 4c HS | 2c Se | 2c Score | 2c HS |
|---|---|---|---|---|---|---|---|
| ast+frozen+ce | 71.0 ± 4.4 | 42.6 ± 3.5 | 56.8 ± 0.7 | 53.0 ± 1.3 | 56.0 ± 4.9 | 63.5 ± 0.7 | 62.3 ± 1.2 |
| ast+frozen+patchmix | 74.2 ± 4.8 | 40.6 ± 4.5 | 57.4 ± 0.8 | 52.2 ± 2.8 | 51.7 ± 5.6 | 62.9 ± 0.8 | 60.6 ± 2.5 |
| ast+frozen+pafa | 70.2 ± 1.8 | 42.7 ± 1.4 | 56.4 ± 0.4 | 53.1 ± 0.6 | 55.6 ± 1.3 | 62.9 ± 0.4 | 62.1 ± 0.3 |
| ast+frozen+qlung | 72.2 ± 4.0 | 40.7 ± 3.2 | 56.5 ± 0.5 | 51.9 ± 1.5 | 53.9 ± 3.5 | 63.1 ± 0.6 | 61.6 ± 0.9 |
| ast+frozen+add_rsc | 74.0 ± 2.2 | 42.1 ± 0.9 | 58.1 ± 1.1 | 53.7 ± 0.8 | 56.1 ± 1.9 | 65.1 ± 1.0 | 63.8 ± 1.1 |
| ast+frozen+ce_pafa_cfg | 69.9 ± 2.9 | 43.5 ± 2.5 | 56.7 ± 0.4 | 53.5 ± 1.1 | 56.4 ± 3.1 | 63.2 ± 0.4 | 62.3 ± 0.8 |
| htsat+frozen+ce | 72.2 ± 5.7 | 35.2 ± 5.3 | 53.7 ± 0.9 | 46.9 ± 4.0 | 48.6 ± 6.9 | 60.4 ± 1.4 | 57.6 ± 3.8 |
| htsat+frozen+patchmix | 78.7 ± 5.4 | 30.5 ± 4.1 | 54.6 ± 1.2 | 43.6 ± 3.6 | 42.9 ± 6.6 | 60.8 ± 1.4 | 55.1 ± 4.6 |
| htsat+frozen+pafa | 81.9 ± 2.8 | 30.6 ± 2.5 | 56.2 ± 0.5 | 44.4 ± 2.3 | 43.7 ± 3.4 | 62.8 ± 1.1 | 56.9 ± 2.3 |
| htsat+frozen+qlung | 74.1 ± 3.9 | 33.7 ± 2.7 | 53.9 ± 0.9 | 46.2 ± 1.9 | 47.0 ± 3.1 | 60.6 ± 1.2 | 57.4 ± 1.6 |
| htsat+frozen+add_rsc | 70.8 ± 2.2 | 37.3 ± 2.9 | 54.0 ± 0.9 | 48.7 ± 2.1 | 51.4 ± 3.1 | 61.1 ± 1.0 | 59.5 ± 1.6 |
| htsat+frozen+ce_pafa_cfg | 83.4 ± 5.6 | 28.2 ± 4.3 | 55.8 ± 1.3 | 41.8 ± 3.8 | 39.4 ± 6.7 | 61.4 ± 1.7 | 53.0 ± 4.5 |
| opera_ct+frozen+ce | 82.9 ± 6.1 | 17.0 ± 6.3 | 49.9 ± 0.3 | 27.5 ± 7.7 | 24.2 ± 9.6 | 53.6 ± 1.8 | 36.4 ± 10.1 |
| opera_ct+frozen+patchmix | 85.5 ± 4.3 | 17.0 ± 5.4 | 51.3 ± 0.8 | 28.0 ± 7.3 | 23.2 ± 7.8 | 54.4 ± 1.9 | 35.8 ± 9.4 |
| opera_ct+frozen+pafa | 84.1 ± 3.9 | 12.1 ± 3.9 | 48.1 ± 0.3 | 20.9 ± 5.8 | 20.0 ± 5.8 | 52.0 ± 1.2 | 31.8 ± 7.5 |
| opera_ct+frozen+qlung | 83.0 ± 6.0 | 17.0 ± 3.9 | 50.0 ± 1.5 | 27.9 ± 5.2 | 23.4 ± 5.8 | 53.2 ± 1.5 | 36.0 ± 6.8 |
| opera_ct+frozen+add_rsc | 81.2 ± 6.8 | 21.1 ± 7.5 | 51.1 ± 1.2 | 32.6 ± 9.8 | 30.5 ± 11.4 | 55.8 ± 3.1 | 42.8 ± 13.1 |
| opera_ct+frozen+ce_pafa_cfg | 81.7 ± 9.2 | 14.2 ± 8.9 | 48.0 ± 0.4 | 22.8 ± 13.0 | 22.2 ± 13.8 | 52.0 ± 2.4 | 32.3 ± 17.3 |
| beats+frozen+ce | 79.1 ± 5.8 | 27.2 ± 5.0 | 53.2 ± 0.4 | 40.1 ± 5.1 | 38.7 ± 7.5 | 58.9 ± 1.0 | 51.4 ± 6.0 |
| beats+frozen+patchmix | 85.1 ± 7.0 | 22.3 ± 6.6 | 53.7 ± 0.7 | 34.6 ± 7.5 | 29.9 ± 10.1 | 57.5 ± 1.8 | 43.0 ± 9.8 |
| beats+frozen+pafa | 81.6 ± 1.7 | 25.1 ± 1.3 | 53.3 ± 0.4 | 38.4 ± 1.4 | 36.0 ± 2.4 | 58.8 ± 0.7 | 49.9 ± 2.1 |
| beats+frozen+qlung | 67.1 ± 3.7 | 37.9 ± 4.1 | 52.5 ± 0.6 | 48.2 ± 2.4 | 54.7 ± 5.3 | 60.9 ± 1.5 | 60.0 ± 2.2 |
| beats+frozen+add_rsc | 80.4 ± 2.2 | 27.2 ± 2.5 | 53.8 ± 0.7 | 40.6 ± 2.6 | 41.3 ± 3.3 | 60.9 ± 0.9 | 54.5 ± 2.6 |
| beats+frozen+ce_pafa_cfg | 75.6 ± 3.3 | 30.3 ± 2.9 | 52.9 ± 0.2 | 43.1 ± 2.4 | 43.3 ± 4.4 | 59.4 ± 0.6 | 54.8 ± 2.7 |
| dass+frozen+ce | 70.4 ± 1.3 | 40.6 ± 0.9 | 55.5 ± 0.3 | 51.5 ± 0.4 | 52.4 ± 1.0 | 61.4 ± 0.2 | 60.1 ± 0.3 |
| dass+frozen+patchmix | 74.5 ± 3.5 | 38.2 ± 3.2 | 56.4 ± 0.3 | 50.4 ± 2.1 | 48.3 ± 3.9 | 61.4 ± 0.4 | 58.4 ± 2.0 |
| dass+frozen+pafa | 69.3 ± 1.2 | 40.4 ± 1.4 | 54.9 ± 0.1 | 51.0 ± 0.8 | 52.2 ± 2.2 | 60.8 ± 0.5 | 59.5 ± 1.0 |
| dass+frozen+qlung | 70.5 ± 3.0 | 40.2 ± 2.7 | 55.4 ± 0.3 | 51.1 ± 1.4 | 52.3 ± 3.4 | 61.4 ± 0.6 | 59.9 ± 1.2 |
| dass+frozen+add_rsc | 72.3 ± 3.3 | 42.4 ± 2.2 | 57.3 ± 0.7 | 53.3 ± 0.9 | 56.6 ± 3.6 | 64.4 ± 1.2 | 63.3 ± 1.6 |
| dass+frozen+ce_pafa_cfg | 68.0 ± 1.3 | 41.7 ± 1.4 | 54.8 ± 0.1 | 51.7 ± 0.7 | 53.3 ± 2.1 | 60.6 ± 0.5 | 59.7 ± 0.9 |
