# LODO plan v1 and ensemble plan (written 19/10/2026)

Input: grid v1 results (`docs/reports/grid-v1-results.md`). Cells chosen by the user on 19/10/2026.

## Cells (11)

| Encoder | Cells | Control in the set |
|---|---|---|
| AST | ce, pafa, patchmix_cl | ce (same settings as patchmix_cl; pafa differs in settings too) |
| HTS-AT | ce, patchmix_cl, sg_scl | ce (same settings as both) |
| BEATs | patchmix_cl, sg_scl, pafa, ce_pafa_cfg, qlung | ce_pafa_cfg controls pafa only |

- [Fact] BEATs CE is not in the set: on BEATs, the patchmix_cl, sg_scl and qlung effects under device shift are compared
  with each other and with ce_pafa_cfg, not with a same-settings CE. Adding BEATs CE later costs about +5 h.
- DASS and OPERA-CT are left out.

## Protocol

- Leave one device out: Meditron, LittC2SE, Litt3200, AKGC417L (`--holdout_device`, `experiments/train/lodo_<dev>.yaml`,
  split `src.processing.splits.lodo_split`). Test = every cycle of the held-out device. A patient with any recording on
  that device leaves training and calibration. 20% of each source device's patients are calibration patients (seeded by
  the run seed, so the same patients for every cell of one seed). The 2 patients on both sides of the official split are
  dropped.
- Fold sizes (cycles): Meditron train 3,791 / cal 1,461 / test 1,370; LittC2SE 4,737 / 1,104 / 594; Litt3200
  4,773 / 1,378 / 481; AKGC417L 1,950 / 484 / 4,216. The AKGC417L fold is class-mix dominated: reported flagged, plus a
  macro mean without it.
- Recipe and epoch count frozen from the grid: each cell's grid config + the fold file, `--selection fixed
  --report_epochs <last epoch>` (50; BEATs pafa and ce_pafa_cfg 100). Seeds 0-4. Every epoch's test softmax is kept.
- Run `lodo-v1`; cell `<grid cell>+lodo_<device>`; 11 cells x 4 folds x 5 seeds = 220 units.
- Time: about 97 machine-hours (grid throughput x fold size, 0.70-0.94 of a grid unit), about 49 h on A and B together.

## Stages

| Stage | What | Cost | Output |
|---|---|---|---|
| L0 | Job lists (`grid.py`-style, balanced on both machines, `RUN_ID=lodo-v1`); smoke 1 epoch of one unit per encoder x fold (PAFA patient ids, SG-SCL with 3 source devices, QLung quality on the LODO split, 5 s cycles); GPU logger | 2-3 h | lists in `logs/campaigns/lodo-v1-{A,B}.txt` |
| L1 | Train the 220 units, no SC | about 49 h | `outputs/train/lodo-v1/<cell>+lodo_<dev>/seed<s>/` with `cal_preds.npz`, `test_preds.npz`, `report.json` |
| L2 | Inference arms on every L1 unit (`src.evaluation.lodo_ttsc`): test-time SC (held-out device corrected, static and dynamic bound), A2, TTA-EQ (8 views) | a few hours | extra arm units |
| L3 | Analysis: per fold and macro Sp, Se, Score, HS, 2c, worst fold; method effects vs controls; H2 with `src.evaluation.lodo` (prior-matched coverage, null gate over 200 re-splits, two-level bootstrap) for each inference arm vs the same cell without it | 1 day | `docs/reports/lodo-v1-results.md` |
| L4 | Trained SC arms (`sc_mode` dynamic and static) on the cells chosen from L3 (about 3): official split 5 seeds (gate G) + LODO 20 units, per mode. SC turns the spectrogram cache off (slower loading) | about 50-70 h for 3 cells | arms `sc`, `sc+sc_static` |
| L5 | Ensembles (below) on L1, L2 and L4 outputs | inference only | ensemble arms |
| L6 | Distillation of the primary ensemble into one BEATs model | about 10 h | student arm |

After LODO: KAUH positive controls (frozen model), screening view, phantom tests (pending hardware), per the project plan.

GPU log for every training stage, on each machine:
`setsid nohup nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 >> logs/gpu-<M>.csv 2>&1 < /dev/null &`

## Ensemble plan (fixed now, before any LODO result)

- E1, primary: equal-weight mean of the softmax of all 11 LODO cells, same fold and seed, at the fixed epoch.
- E2: the 5 BEATs cells. E3, E4: the 3 AST cells, the 3 HTS-AT cells.
- No weights, no member selection after LODO results. 5 ensembles per fold (one per seed), reported as mean ± SD like a
  cell.
- Conformal: LAC on the ensemble softmax, calibrated on the ensemble softmax of the calibration patients (the same
  patients for all members of one seed).
- Inference arms: the ensemble of the members' test-time SC (or A2, TTA-EQ) outputs is that arm's ensemble.
- Code (to write, L5): `src/evaluation/ensemble.py` reads member unit directories and writes an ensemble unit directory
  in the same format (`cal_preds.npz`, `test_preds.npz`, `report.json`), so `src.evaluation.lodo` reads it as an arm.
- Official split, for reference (already computed, last epoch): 11-cell set 63.0 ± 0.4 Score (test-picked cells),
  8 BEATs methods 62.0 ± 0.2.

## Distillation plan (L6)

- Teachers: the E1 members of the same fold and seed. Teacher softmax on that fold's training cycles, from the final
  checkpoints (inference only). The held-out device never reaches the student.
- Soft target per training cycle: mean of the teacher softmax (reference: `../repos/rsc-ensemble-kd`, `mean` mode).
- Student: BEATs with the PAFA recipe (best single cell), loss = cross-entropy to the soft target (plus its PAFA losses).
  Optional mix with the hard label, weight chosen by grouped CV on training patients.
- Known weakness: teachers predict on cycles they trained on (over-confident targets). Clean variant: out-of-fold teacher
  predictions (cross-fitting), k times the teacher cost; only if the simple variant helps.
- Code (to write): teacher-softmax export on the training cycles; `--soft_targets <npz>` in the dataset and engine.
- Evaluated as one more LODO arm, against the best single cell and E1.

## Decisions still open

- BEATs CE as a control (+5 h).
- Which cells get the trained SC arms (L4), after L3.
- The 80/20 split step (top cells x 5 folds x 5 seeds) from the stage plan: keep for benchmark comparability or drop in
  favour of LODO.
