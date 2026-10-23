# AuscultTrust: consolidated results report

| | |
| --- | --- |
| Report version | 5.0 (4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0 and 3.0 in git history; 2.0 is `docs/reports/report_archive_2026-10-07.md`) |
| Datetime | 2026-10-23 15:00 (+07) |
| Proposal / prereg | v9 (`docs/updates/research-proposal-v9-en.md`); `docs/updates/prereg-v8.md` with Amendments 1-5 (`docs/updates/prereg-v9-amendment*.md`) |
| Plans | `docs/updates/stage-plan-v1.md` (grid), `docs/updates/lodo-plan-v1.md` (LODO, ensembles; Amendment 4 wins where they differ) |
| Results (all on A) | `outputs/train/<run>/<cell>/seed<s>/` in this repo (section 4.3); checkpoints of the BEATs PAFA family on A, the others stay on B, D, E |
| Detailed sources | `docs/reports/report_2026-10-09.md` (registered core), `docs/reports/grid-v1-results.md` (grid, every cell), `docs/reports/pcmcl-v1-results.md` (PC-MCL), `docs/reports/lodo-v1-results.md` (LODO v1, SC), `docs/reports/ensemble-v1-results.md` (ensembles by encoder group), `docs/handoff/HANDOFF.md` (machines, commands, incidents) |

Tags: [Fact] measured in the repo, [Interpretation] reasoned, [Assumption], [Pending] not run, [Preliminary] computed on
the units finished so far (may change; no verdict). Score = (Sp + Se) / 2 on ICBHI 4-class, official split unless
stated. All numbers are mean ± SD over seeds 0-4 unless a seed count is given.

## 1. Status and key findings

**Current step (23/10):** stage 10, choice of the final AuscultTrust. All training is done (stages 0-9: 0 runs left, 0
failed), every result is gathered on machine A in this repository (section 4.3), and the LODO and ensemble tables are
regenerated. The encoder and method are fixed (BEATs + PAFA, fine-tuned, user decision 22/10); the SC variant and the
use of an ensemble are open (section 15).

**Next steps:** (1) the user chooses the SC variant and single model vs ensemble; (2) results report for the submission
of 25/10 from this file; (3) with the frozen final model: KAUH positive controls, screening view, serving bundle of the deployed demo;
(4) phantom tests (H1, H3, co-primary H2) once the hardware exists. Details in section 18.

Results and findings are reported per stage of the plan (sections 5-16, numbering as in section 2). Key findings:

1. [Fact] Encoder choice dominates method choice: BEATs leads on the official split (stage 2) and under device shift
   (stage 3); most methods stay within one seed SD of CE on the same encoder.
2. [Fact] BEATs PAFA is the one consistent method gain (+2.6 Score, +8.5 worst-device Score over BEATs CE on the official
   split) and the best cell on the hardest unseen device (Litt3200).
3. [Fact] H2 cannot be tested in LODO v1: no held-out device shows a coverage error above chance once models are trained
   to their last epoch (stage 7). The registered AST study (models stopped at epoch 2) did show one.
4. [Fact] SC helps Score, not coverage: test-time SC raises Litt3200 Score for every cell (stage 6); trained SC raises
   the LODO macro Score for 6 of 7 cells at no cost on the official split (stage 4).
5. [Fact] Final AuscultTrust (user decision 22/10): BEATs + PAFA, fine-tuned. With trained SC static it scores 62.3 ±
   0.9 on the official split (best last-epoch Score so far) and 60.3 ± 0.8 macro in LODO; the SC variant (none, trained
   static, trained dynamic, test-time) is the user's choice now that stage 8 has ended (section 15).
6. [Fact] Ensembles raise Se at PAFA's Sp only with a target-side anchor: equal-weight E1+E3 (BEATs + HTS-AT) or E1
   (BEATs) with a label-free rule that calls as many cycles normal as PAFA does on the same unlabelled cycles keep Sp and
   gain 1.4-3.1 Se (+1.1 to +2.1 Score) on the official split and in LODO; learned member weights never beat equal
   weights (stage 9).
7. [Fact] The dynamic SC bound helps SG-SCL and Patch-Mix CL on the official split (+1.7, +1.8; BEATs SG-SCL + SC
   dynamic 62.8 ± 1.4 is the best single cell) but lowers the LODO macro for PAFA (-0.9); for BEATs PAFA the static bound
   is the better SC variant. PAFA + PC-MCL with SC static has the best LODO macro (60.7) and the smallest sets (stage 8).

Runs (all finished; at 23/10 15:00):

| Run | What | Done | Where | Expected end |
| --- | --- | --- | --- | --- |
| `lodo-v1`, no SC | 11 cells x 4 held-out devices x 5 seeds | 220 / 220 | A, B, E | done 21/10 00:17 |
| `lodo-v1`, inference arms | test-time SC static and dynamic, A2, TTA-EQ on every no-SC unit | 880 / 880 | A, B, E | done 21/10 15:05 |
| `lodo-v1`, trained SC static | 7 cells x (official split + 4 folds) x 5 seeds | 175 / 175 | C, D, A, B, E | done 21/10 13:25 |
| `lodo-v1`, trained SC dynamic + PC-MCL SC | BEATs PAFA, SG-SCL, Patch-Mix CL (dynamic, 75 units) + BEATs PAFA + PC-MCL (static and dynamic, 50 units) | 125 / 125, 0 failed | A, B, D, E | done 23/10 02:20 (outputs on A) |
| `pcmcl-v1` | 2 cells x 5 seeds | 10 / 10 | D | done 20/10 |

Machine C is not used any more: our jobs there were killed three times by its user (section 4.2).

## 2. Overall plan

The pipeline: encoder x method grid on the official split, then LODO without SC, then SC (trained SC on 7 cells, and
test-time SC on the LODO models), then the final configuration; ensembles and distillation are optional. KAUH controls,
the screening view and the phantom tests close the project.

| # | Stage | What | Status |
| --- | --- | --- | --- |
| 0 | Phase 0 (v4-v8) | Frozen-encoder ladders, Part II shift study, Stage-1 screening of device variants | Done (section 5); `docs/reports/report_archive_2026-10-07.md` |
| 1 | Registered core (v9) | Gate G on AST Patch-Mix CL vs AuscultTrust (AST + SC); LODO on AST (baseline, Freq-MixStyle, trained and test-time SC, A2, TTA-EQ); KAUH control A | Done 09/10 (section 6) |
| 2 | Grid v1 | 5 encoders x 14 methods (8 fine-tuned, 6 frozen) x 5 seeds, official split | Done 18/10 (section 7) |
| 3 | LODO v1 without SC (L1) | 11 cells x 4 held-out devices x 5 seeds = 220 units | Done 21/10 (section 8) |
| 4 | Trained SC, static bound | 7 cells, official split + 4 LODO folds, 175 units | Done 21/10 (section 9) |
| 5 | PC-MCL (Amendment 5) | BEATs PC-MCL and PAFA + PC-MCL, official split | Done 20/10; rule not met (section 10); its PAFA variant now also with SC (stage 8) |
| 6 | Inference arms (L2) | Test-time SC (static, dynamic), A2, TTA-EQ on every LODO unit (`src.evaluation.lodo_ttsc`) | Done 21/10 (section 11) |
| 7 | Analysis (L3) | Per-fold Sp/Se/Score/HS/2c; method vs CE; H2 (prior-matched coverage, null gate, two-level bootstrap) with set size beside abs(Delta); SC vs no SC paired by seed | Done 21/10: H2 not testable (section 12); `docs/reports/lodo-v1-results.md` |
| 8 | Trained SC, dynamic bound | BEATs PAFA, SG-SCL, Patch-Mix CL (75 units) + BEATs PAFA + PC-MCL with SC static and dynamic (50 units) | Done 23/10 (section 13) |
| 9 | Ensembles, distillation | Registered E1-E4 (equal weights, 11/5/3/3 cells); user groups by encoder with equal and fitted weights | User groups done 22/10 (section 14); registered E1-E4 and distillation [Pending], optional |
| 10 | Final AuscultTrust | Choose encoder + method + SC with the user | Decided 22/10: BEATs + PAFA (fine-tuned); SC variant open (section 15) |
| 11 | After LODO | KAUH positive controls on the final model, screening view (recording level), phantom tests (H1, H3) | [Pending] (section 16); phantom depends on hardware |
| - | Demo app (`app/`) | FastAPI + web demo; currently serves the BEATs + PAFA bundle | Deployed to Cloudflare Pages; GPU API available through Tailscale Funnel at `vmo-namdp15-d.tail09c58b.ts.net` (checked 23/10) |

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
| 21/10 | Start the inference arms on all LODO units; then the next steps (analysis, trained SC with the dynamic bound); fix and retry any failure at once. |
| 21/10 | Dynamic-bound SC only on BEATs PAFA, SG-SCL, Patch-Mix CL; add BEATs PAFA + PC-MCL with SC static and dynamic (SC on PC-MCL's concatenated cycles implemented: each part corrected with its device's coefficient). Report reorganised by plan stage. |
| 22/10 | After the review of stages 0-4: the final AuscultTrust is BEATs + PAFA, fine-tuned (replaces the AST Patch-Mix CL definition of v9). |
| 22/10 | Ensembles: report best epoch on test as an optimistic column; goal = raise Se while keeping Sp. |
| 22/10 | Re-evaluate ensembles by encoder group, with weights (trainable if useful): E1 BEATs PAFA, SG-SCL, Patch-Mix CL; E2 AST Patch-Mix CL, PAFA; E3 HTS-AT Patch-Mix CL, SG-SCL; cross-group E1+E2, E1+E3, E2+E3, E1+E2+E3. |

## 4. Compute, timeline and incidents

### 4.1 Machines (23/10 15:00: all idle, no run left)

| Machine | GPU | Role |
| --- | --- | --- |
| A (local) | RTX 5000 Ada 32 GB | grid, LODO, SC, analysis; holds all outputs (section 4.3) |
| B | RTX 5090 32 GB | grid, LODO, SC |
| D | RTX 5090 32 GB | PC-MCL, SC |
| E | RTX 5090 32 GB | LODO, SC |
| C | RTX PRO 6000 96 GB | dropped 20/10 (section 4.2) |

- [Fact] `scripts/lodo_move.py` moved the remaining static-SC units between machines (stop a scheduler by PID while its
  jobs finish, set cut-off units aside to rerun from epoch 0, split by measured throughput, keep schedulers alive, copy
  outputs to A). The first dynamic-SC lists (`lodo-v1-<M>5`, 7 cells) were stopped at 17:00 by PID (their running BEATs units
  continued) and replaced by `lodo-v1-<M>6` (BEATs cells + PC-MCL SC; BEATs units first in each list). 1-epoch smokes of
  the dynamic bound (BEATs, AST, HTS-AT, on D) and of PAFA + PC-MCL with SC (on A) passed.
- [Fact] Throughput in B-smoke hours per wall hour: A 0.80, B 1.94, E 1.96; B, D and E ran the last static-SC units at
  1.57-1.73.
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
- [Fact] Inference arms, 21/10: the first start failed at argument parsing (the scheduler appends `--num_workers`, which
  `lodo_ttsc` does not take; fixed by ending each job line with `&& :`); the 20 QLung units then failed at model build
  (`class_nums` not set at inference; one-line fix in `src/evaluation/lodo_ttsc.py`, same value as in training) and were
  rerun. All 880 arm units are complete.
- [Fact] A's GPU log (`logs/gpu-A.csv`) has a gap on 21/10 11:40-11:50 (logger stopped by mistake, restarted).
- [Fact] 0 failed training units and 0 NaN or collapsed runs in every LODO, SC and PC-MCL list.
- [Fact] Machine E rebooted on 22/10 about 14:40 (cause unknown): one PAFA + PC-MCL unit cut off at epoch 54 was set
  aside (`lodo-v1-interrupted/...seed2.reboot-2210`) and rerun from epoch 0. Stage 8 was rebalanced twice onto B (lists
  tag 7 and 8). Machine A was shut down on 23/10 morning after its last list had finished (02:18): no unit lost.

### 4.3 Where the results are (unified 23/10)

- [Fact] All outputs of machines B, D and E were copied to A (`rsync --ignore-existing`, nothing overwritten), so this
  repository holds every unit's `report.json`, `test_preds.npz`, `cal_preds.npz` (LODO) and `sc_state.npz` (SC):
  `outputs/train/grid-v1` (350 units), `lodo-v1` (1,400: 220 no SC, 880 inference arms, 175 SC static, 125 stage 8),
  `pcmcl-v1` (10), `lodo-ast` (260) and `lodo-bench` (95) for the registered AST study, plus the earlier runs
  (`screen-ast*`, `repro-wave1*`, `grid-v1-rerun|resumed|sgscl-ref`, `lung-srad`) and smoke runs. Phase 0 outputs stay
  in `outputs/` (ladders, shift study) and `../archive`.
- [Fact] Checkpoints (about 87 GB of BEATs alone on B, D, E) do not fit A's free disk; A received the BEATs PAFA family
  (`lodo-v1/beats+ft+pafa*`, 120 units with their `.pth`: LODO no SC, SC static and dynamic, PAFA + PC-MCL with SC;
  official split and LODO), and A already held `grid-v1/beats+ft+pafa` (official split, no SC). A has 106 GB free. Other
  checkpoints stay on the machine that trained them (B: grid-v1 B half, lodo-ast, lodo-bench, lodo-v1; D, E: lodo-v1).
- Regenerate the tables: `bash scripts/lodo_h2_all.sh`, `PYTHONPATH=. .venv-train/bin/python scripts/lodo_tables.py`,
  `scripts/ensemble.py`, `scripts/ensemble_sesp.py`, `PYTHONPATH=. .venv-train/bin/python scripts/grid.py table last`.

## 5. Stage 0: Phase 0 (before v9)

Frozen-encoder ladders, fine-tuning ladders, Part II shift study and the Stage-1 screening of device variants (P0-P8)
are in `docs/reports/report_archive_2026-10-07.md`. Main findings: the early weak base model was the encoder;
fine-tuning did not raise decodability or the coverage deficit (F4); no device variant separated from the baseline in
3-fold CV.

## 6. Stage 1: Registered core (v9, AST), from `docs/reports/report_2026-10-09.md`

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

## 7. Stage 2: Grid v1, method x encoder on the official split

### 7.1 Setup

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

### 7.2 Best cell per encoder, last epoch

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

### 7.3 Top 5, best epoch on test (optimistic)

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

### 7.4 Cells carried into LODO v1 and the trained-SC arms (last epoch)

LODO without SC: all 11 rows (stage 3). Trained SC: rows marked SC (stage 4).

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

### 7.5 Method effects (paired difference to CE on the same encoder and seed, last epoch)

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

### 7.6 Per-class recall and ensembles (BEATs, last epoch)

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

### 7.7 Coverage, published numbers, incidents

- [Fact] Pooled coverage is at nominal (88.2-90.5%) in every cell; the worst device (Littmann 3200 in 343 of 350 units)
  falls to 85-88% in most fine-tuned cells and 67-82% in many frozen cells.
- [Fact] AST best epoch on test vs published: Patch-Mix CL 61.1 ± 0.4 vs 62.37; QLung 61.0 ± 0.9 vs 62.01; ADD-RSC
  59.6 ± 0.5 vs 65.53 (the reference resizes AST input to 1024 x 256 with its own AST; here the denoising layer runs on
  each encoder's own input). Published BEATs PC-MCL: 65.37 ± 0.73 (best epoch on test; ours in stage 5, section 10).
- Incidents: QLung's reference quality score is NaN on 71 AKGC417L cycles (leading silence), our port skips silent
  frames; SG-SCL's reference reversal schedule collapsed BEATs in 3 of 4 seeds, all SG-SCL cells use the DANN ramp;
  power cut 17/10 10:43 (9.6 h), 13 resumed units rerun cleanly and swapped in (resumed vs rerun Score: mean abs 0.61,
  largest 2.5).

### 7.8 Findings (stage 2)

- [Fact] Encoder choice dominates method choice: fine-tuned Score BEATs 58.9-61.7, HTS-AT 56.3-59.6, DASS 57.5-59.7,
  AST 56.9-59.0, OPERA-CT 49.3-51.7; most methods differ from CE on the same encoder by less than one seed SD.
- [Fact] BEATs PAFA is the one large consistent method gain (+2.6 Score, +8.5 worst-device Score), mostly through Sp;
  PAFA and SG-SCL push Sp up and Se down, Patch-Mix CL the other way.
- [Fact] Averaging methods helps more than any single method: 8 BEATs methods 62.0 ± 0.2 (SD 0.2 vs 0.7-1.5 for single
  cells) with smaller sets (1.95 vs 2.47 labels).
- [Fact] Test-selected epochs inflate Score by 0.6-4.4 points.
- [Fact] Littmann 3200 is the worst device in 343 of 350 units; pooled coverage stays at nominal while the worst device
  drops to 85-88% (fine-tuned) and 67-82% (frozen).

## 8. Stage 3: LODO v1 without SC, from `docs/reports/lodo-v1-results.md`

220 units (11 cells x 4 held-out devices x 5 seeds), fixed last epoch; test = every cycle of the held-out device;
conformal LAC at 90% calibrated on the calibration patients of the source devices; Delta = 0.90 minus prior-matched
coverage.

### 8.1 Score and coverage per held-out device (5 seeds)

| Cell | Meditron | LittC2SE | Litt3200 | AKGC417L | Macro Score | Macro w/o AKGC417L | Litt3200 Cov / set | AKGC417L Cov / set |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ast+ft+ce | 56.2 | 65.1 | 50.2 | 53.7 | 56.3 ± 1.7 | 57.2 ± 2.5 | 96.3 / 3.04 | 78.1 / 2.40 |
| ast+ft+pafa | 57.0 | 64.5 | 50.5 | 56.1 | 57.0 ± 1.0 | 57.3 ± 1.2 | 97.6 / 3.22 | 74.5 / 1.92 |
| ast+ft+patchmix_cl | 55.6 | 63.0 | 52.8 | 54.0 | 56.3 ± 1.2 | 57.1 ± 1.9 | 95.3 / 2.97 | 85.2 / 2.81 |
| htsat+ft+ce | 58.1 | 63.0 | 47.3 | 55.9 | 56.1 ± 1.7 | 56.1 ± 1.9 | 88.9 / 2.67 | 77.9 / 2.13 |
| htsat+ft+patchmix_cl | 58.3 | 63.4 | 49.6 | 56.1 | 56.9 ± 2.0 | 57.1 ± 2.2 | 91.2 / 2.60 | 74.1 / 1.90 |
| htsat+ft+sg_scl | 58.2 | 63.7 | 48.8 | 55.7 | 56.6 ± 1.5 | 56.9 ± 1.7 | 89.7 / 2.87 | 74.6 / 1.89 |
| beats+ft+ce | 57.6 | 64.4 | 56.3 | 57.9 | 59.1 ± 1.3 | 59.5 ± 1.8 | 99.5 / 3.54 | 81.1 / 2.35 |
| beats+ft+patchmix_cl | 57.7 | 66.4 | 57.5 | 58.8 | 60.1 ± 1.7 | 60.5 ± 2.6 | 99.4 / 3.54 | 84.0 / 2.36 |
| beats+ft+sg_scl | 58.4 | 64.8 | 54.9 | 57.9 | 59.0 ± 1.7 | 59.4 ± 2.3 | 94.9 / 2.80 | 79.3 / 2.05 |
| beats+ft+pafa | 57.2 | 62.0 | 59.7 | 56.4 | 58.8 ± 1.4 | 59.6 ± 1.6 | 96.6 / 2.76 | 80.9 / 2.26 |
| beats+ft+qlung | 58.5 | 65.1 | 57.2 | 57.8 | 59.6 ± 0.8 | 60.3 ± 0.6 | 98.8 / 3.37 | 85.2 / 2.46 |

Sp, Se and HS per fold, and coverage on every fold, are in the detailed report. [Fact] Litt3200 is the hardest fold
for every cell; Sp falls to 33-63% there except for BEATs PAFA (88.5%). On AKGC417L raw coverage is 74-85% but the
prior-matched |Delta| is 0.03-0.06: the shortfall is class mix.

### 8.2 Method vs CE on the same encoder (paired by seed)

| Encoder | Method | dScore macro | dScore w/o AKGC417L | dScore Litt3200 |
| --- | --- | --- | --- | --- |
| AST | pafa | +0.7 ± 2.3 | +0.2 ± 3.2 | +0.3 ± 7.5 |
| AST | patchmix_cl | +0.0 ± 1.7 | -0.0 ± 1.7 | +2.6 ± 4.4 |
| HTS-AT | patchmix_cl | +0.8 ± 1.5 | +1.0 ± 1.9 | +2.3 ± 4.4 |
| HTS-AT | sg_scl | +0.5 ± 1.4 | +0.8 ± 1.8 | +1.5 ± 3.4 |
| BEATs | patchmix_cl | +1.0 ± 1.4 | +1.1 ± 1.8 | +1.2 ± 3.8 |
| BEATs | sg_scl | -0.1 ± 1.5 | -0.1 ± 2.0 | -1.4 ± 4.4 |
| BEATs | pafa | -0.3 ± 1.5 | +0.1 ± 1.3 | +3.3 ± 2.9 |
| BEATs | qlung | +0.6 ± 0.9 | +0.8 ± 1.5 | +0.8 ± 6.9 |

### 8.3 Findings (stage 3)

- [Fact] The encoder order holds under device shift: macro Score BEATs 58.8-60.1, AST 56.3-57.0, HTS-AT 56.1-56.9,
  about the official-split level for BEATs.
- [Fact] Litt3200 is the hardest held-out device for every cell (Score 47.3-59.7, Sp 33-63%); BEATs PAFA is best there
  (59.7, Sp 88.5%).
- [Fact] Methods vs CE stay within seed SD in LODO (macro -0.3 to +1.0); BEATs PAFA gains on Litt3200 (+3.3 ± 2.9).
- [Fact] Raw coverage on AKGC417L is 74-85%, but the prior-matched |Delta| is 0.03-0.06: the shortfall is class mix.

## 9. Stage 4: Trained SC, static bound

175 units: 7 cells x (official split + 4 held-out devices) x 5 seeds, paired by seed with the no-SC units.

### 9.1 Official split vs no SC (same cell and seed, last epoch)

| Cell | Sp | Se | Score | HS | 2c Score | Worst-dev Score | Cov | Worst-dev Cov | Set size | dScore vs no SC | dWorst-dev Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ast+ft+patchmix_cl | 70.8 ± 2.0 | 45.8 ± 1.3 | 58.3 ± 1.1 | 55.6 ± 1.0 | 66.1 ± 0.8 | 49.8 ± 3.0 | 89.4 ± 0.1 | 87.5 ± 0.3 | 2.25 ± 0.06 | +0.1 ± 1.3 | +2.8 ± 5.0 |
| ast+ft+pafa | 71.7 ± 2.3 | 46.4 ± 1.6 | 59.0 ± 0.7 | 56.3 ± 0.8 | 67.0 ± 0.6 | 50.3 ± 2.0 | 89.5 ± 0.2 | 85.5 ± 0.6 | 2.34 ± 0.02 | +0.7 ± 0.5 | +2.9 ± 2.5 |
| htsat+ft+patchmix_cl | 68.8 ± 3.3 | 48.1 ± 1.0 | 58.5 ± 1.2 | 56.6 ± 0.6 | 65.9 ± 0.8 | 46.3 ± 3.5 | 90.1 ± 0.3 | 87.3 ± 0.5 | 2.24 ± 0.08 | -0.6 ± 1.4 | -0.2 ± 6.1 |
| htsat+ft+sg_scl | 73.0 ± 3.0 | 45.7 ± 1.6 | 59.3 ± 1.0 | 56.2 ± 0.7 | 66.5 ± 0.9 | 49.7 ± 4.3 | 90.2 ± 0.3 | 87.7 ± 1.1 | 2.34 ± 0.03 | -0.3 ± 1.4 | -0.1 ± 5.8 |
| beats+ft+patchmix_cl | 67.4 ± 1.6 | 52.2 ± 1.3 | 59.8 ± 0.8 | 58.8 ± 0.8 | 68.3 ± 0.7 | 52.0 ± 2.6 | 89.2 ± 0.2 | 86.2 ± 0.2 | 2.30 ± 0.06 | +0.1 ± 1.2 | -0.1 ± 4.0 |
| beats+ft+pafa | 75.6 ± 3.1 | 48.9 ± 2.9 | **62.3 ± 0.9** | 59.3 ± 1.3 | 69.5 ± 0.7 | 58.9 ± 1.4 | 89.6 ± 0.3 | 85.6 ± 0.7 | 2.33 ± 0.13 | +0.5 ± 1.5 | -0.1 ± 1.2 |
| beats+ft+sg_scl | 72.0 ± 2.9 | 49.4 ± 1.1 | 60.7 ± 1.0 | 58.6 ± 0.6 | 68.5 ± 0.8 | 51.8 ± 5.5 | 89.8 ± 0.2 | 87.5 ± 0.2 | 2.43 ± 0.16 | -0.4 ± 1.6 | -1.1 ± 6.8 |

### 9.2 LODO vs no SC (paired by seed)

| Arm | dScore macro | dScore Litt3200 | Litt3200 coverage change | Set size change (macro) |
| --- | --- | --- | --- | --- |
| Trained SC, static bound (7 cells) | -0.0 to +1.5 (6 of 7 positive) | -0.4 to +2.9 | -1.4 to +1.7 | -0.29 to 0.00 |

Trained SC static per cell, macro dScore in LODO: AST PAFA +1.4 ± 0.7, AST Patch-Mix CL +0.3 ± 1.5, HTS-AT Patch-Mix CL
+1.2 ± 1.5, HTS-AT SG-SCL +1.5 ± 1.4, BEATs Patch-Mix CL -0.0 ± 1.4, BEATs PAFA +1.5 ± 1.2, BEATs SG-SCL +1.5 ± 1.1.

### 9.3 Findings (stage 4)

- [Fact] Trained SC costs nothing on the official split (-0.6 to +0.7 Score) and raises the LODO macro Score for 6 of 7
  cells (+1.2 to +1.5 for AST PAFA, HTS-AT both, BEATs PAFA and SG-SCL; +0.3 AST Patch-Mix CL; -0.0 BEATs Patch-Mix CL).
- [Fact] BEATs PAFA + trained SC reaches 62.3 ± 0.9 on the official split (best last-epoch Score so far) and about 60.3
  macro in LODO, 61.6 on Litt3200.
- [Fact] Coverage and set size barely move with trained SC (Litt3200 coverage -1.4 to +1.7 points, sets -0.29 to 0.00).

## 10. Stage 5: PC-MCL on BEATs (Amendment 5), from `docs/reports/pcmcl-v1-results.md`

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
- [Fact] Following the user (21/10), PAFA + PC-MCL is now also trained with SC (static and dynamic bound) in stage 8.

## 11. Stage 6: Inference arms on the LODO units

880 units: four arms on each of the 220 no-SC LODO units (`src.evaluation.lodo_ttsc`), paired by seed with the source
unit; ranges over the 11 cells. Test-time SC corrects only the held-out device towards the source devices' mean spectrum,
from its unlabelled cycles. A2 maps the per-bin mean and SD of the held-out spectrograms onto the training statistics.
TTA-EQ averages the softmax over 8 random smooth per-bin gain views.

| Arm | dScore macro | dScore Litt3200 | Litt3200 coverage change | Set size change (macro) |
| --- | --- | --- | --- | --- |
| Test-time SC, static bound | -0.8 to +1.1 | AST +3.4 to +4.6, HTS-AT +2.1 to +2.8, BEATs +0.6 to +2.4 | +0.1 to +2.5 points | -0.06 to +0.12 |
| Test-time SC, dynamic bound | -1.4 to +1.1 | AST +4.1 to +5.3, HTS-AT +1.9 to +3.7, BEATs +0.7 to +2.2 | 0.0 to +2.9 | -0.05 to +0.19 |
| A2 | -6.1 to +0.8 (BEATs -4.0 to -6.1) | BEATs -4.8 to -11.6, HTS-AT +5.0 to +5.2 | -2.7 to +5.2 | -0.07 to +0.35 |
| TTA-EQ (8 views) | 0.0 to +0.5 | -0.4 to +0.6 | -1.1 to -4.6 | -0.07 to -0.32 |

### 11.1 Findings (stage 6)

- [Fact] Test-time SC raises Litt3200 Score for every cell (AST +3.4 to +5.3, HTS-AT +1.9 to +3.7, BEATs +0.6 to
  +2.4); the macro change is small (-1.4 to +1.1); coverage and set size barely move. Static and dynamic bounds are close.
- [Fact] A2 hurts every BEATs cell (macro -4.0 to -6.1; Litt3200 -4.8 to -11.6) and AST PAFA (-2.6), and helps HTS-AT on
  Litt3200 (+5). [Interpretation] Matching whole-device statistics removes the held-out device's class mix along with
  its spectrum.
- [Fact] TTA-EQ changes Score by 0.0 to +0.5 and shrinks sets by 0.07-0.32 labels (Litt3200 over-coverage down 1-5
  points).

## 12. Stage 7: Coverage under device shift (H2)

[Fact] For all 11 cells and all 4 held-out devices the baseline |Delta| stays below the 95th percentile of |Delta| under
exchangeable re-splits of the calibration patients (closest: BEATs Patch-Mix CL on Litt3200, 0.090 vs 0.091). No fold
is tested, so the H2 statistic is not computed for any arm. The registered AST LODO (models stopped at epoch 2) did
have an under-covered Litt3200 fold (0.670); the converged models of LODO v1 do not.

### 12.1 Findings (stage 7)

- [Fact] H2 cannot be tested in LODO v1: no fold of any cell has a baseline coverage error above chance, so no H2
  statistic is computed for test-time SC, trained SC, A2 or TTA-EQ.
- [Fact] Coverage on an unseen device stays near nominal through large sets on Litt3200 (2.6-3.5 of 4 labels) and class
  mix on AKGC417L.
- [Interpretation] A coverage claim for SC now needs the phantom data, earlier-epoch models or another dataset.

## 13. Stage 8: Trained SC, dynamic bound, and PAFA + PC-MCL with SC

BEATs PAFA, SG-SCL and Patch-Mix CL (cells `<cell>+sc`, 75 units) and BEATs PAFA + PC-MCL with trained SC static and
dynamic (cells `beats+ft+pafa_pcmcl+sc+sc_static`, `beats+ft+pafa_pcmcl+sc`, 50 units). AST and HTS-AT dynamic units were dropped (user decision 21/10); 4 BEATs Patch-Mix CL dynamic
units finished before the switch.

Done 23/10 02:20, 125 / 125 units, 0 failed (one unit cut off by a reboot of E on 22/10 was rerun from epoch 0; lists
were rebalanced twice onto B). Official split = last epoch vs the no-SC unit of the same cell and seed (PAFA + PC-MCL vs
its `pcmcl-v1` unit); LODO = macro over the 4 held-out devices, d vs the no-SC LODO cell (PAFA + PC-MCL vs BEATs PAFA).

| Cell | Official Score | d | Worst-dev Score | Set size | LODO macro | d | Litt3200 Score (Sp / Se) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs PAFA, no SC | 61.7 ± 0.7 | | 58.9 | 2.47 | 58.8 ± 1.4 | | 59.7 (88.5 / 30.9) |
| BEATs PAFA, SC static | 62.3 ± 0.9 | +0.5 ± 1.5 | 58.9 | 2.33 | 60.3 ± 0.8 | +1.5 ± 1.2 | 61.5 (88.2 / 34.8) |
| BEATs PAFA, SC dynamic | 62.0 ± 0.5 | +0.3 ± 0.6 | 59.3 | 2.24 | 57.9 ± 1.6 | -0.9 ± 1.5 | 61.3 (88.5 / 34.2) |
| BEATs SG-SCL, SC static | 60.7 ± 1.0 | -0.4 ± 1.6 | 51.8 | 2.43 | 60.5 ± 1.5 | +1.5 ± 1.1 | 57.8 (68.3 / 47.3) |
| BEATs SG-SCL, SC dynamic | 62.8 ± 1.4 | +1.7 ± 1.3 | 58.4 | 2.38 | 59.5 ± 0.8 | +0.4 ± 1.2 | 60.2 (81.8 / 38.5) |
| BEATs Patch-Mix CL, SC static | 59.8 ± 0.8 | +0.1 ± 1.2 | 52.0 | 2.30 | about 60.1 | -0.0 ± 1.4 | about 57.2 |
| BEATs Patch-Mix CL, SC dynamic | 61.5 ± 1.0 | +1.8 ± 1.3 | 57.4 | 2.26 | 59.9 ± 1.6 | -0.2 ± 1.8 | 61.4 (73.6 / 49.3) |
| BEATs PAFA + PC-MCL, SC static | 61.8 ± 0.2 | +0.5 ± 1.1 | 59.0 | 2.01 | 60.7 ± 1.5 | +1.9 ± 1.2 | 59.5 (76.7 / 42.3) |
| BEATs PAFA + PC-MCL, SC dynamic | 60.9 ± 1.3 | -0.4 ± 1.7 | 56.6 | 2.03 | 57.4 ± 0.2 | -1.4 ± 1.3 | 63.0 (85.6 / 40.4) |

### 13.1 Findings (stage 8)

- [Fact] The dynamic bound helps on the official split for SG-SCL and Patch-Mix CL (+1.7, +1.8; worst device +5.5,
  +5.3): BEATs SG-SCL + SC dynamic, 62.8 ± 1.4, is the best last-epoch Score of any single cell, within one SD of BEATs
  PAFA + SC static (62.3 ± 0.9). For PAFA the two bounds are level on the official split (62.0 vs 62.3).
- [Fact] In LODO the dynamic bound raises Litt3200 for every cell (+1.7 to +5.2) but lowers the macro for PAFA (-0.9) and
  PAFA + PC-MCL (-1.4), mostly through Meditron (PAFA 57.2 to 53.5; PAFA + PC-MCL 47.9). The static bound gives the best
  LODO macro (PAFA + PC-MCL 60.7, SG-SCL 60.5, PAFA 60.3).
- [Fact] PAFA + PC-MCL with SC static: steadiest official Score (SD 0.2) and smallest sets (2.01 labels), LODO macro
  +1.9 over BEATs PAFA without SC; with SC dynamic it has the best Litt3200 Score of any arm (63.0, Sp 85.6, Se 40.4).
- [Fact] No arm changes coverage beyond noise; no LODO fold passes the null gate, so H2 stays untestable (stage 7).
- [Interpretation] For BEATs PAFA the static bound is the better SC variant: equal on the official split, +2.4 macro
  over the dynamic bound in LODO, same Litt3200.

## 14. Stage 9: Ensembles and distillation, from `docs/reports/ensemble-v1-results.md`

Groups of the user (22/10; not the registered E1-E4): E1 = BEATs PAFA, SG-SCL, Patch-Mix CL; E2 = AST Patch-Mix CL, PAFA;
E3 = HTS-AT Patch-Mix CL, SG-SCL; and every cross-group union. Weighted mean softmax at the last epoch: `equal`, or
`fit` = simplex weights minimising the NLL of the LODO calibration patients (source devices; the official split reuses
the mean LODO weights). d = paired by seed vs BEATs PAFA alone in the same arm.

| Ensemble (equal weights) | Official Score | d | Worst-dev Score | Set size | LODO macro | d | Litt3200 | Litt3200 Sp / Se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BEATs PAFA alone, no SC | 61.7 ± 0.7 | | 58.9 | 2.47 | 58.8 ± 1.4 | | 59.7 | 88.5 / 30.9 |
| E1 BEATs | 62.5 ± 0.5 | +0.8 ± 0.5 | 57.9 | 2.18 | 60.5 ± 1.5 | +1.7 ± 0.9 | 58.2 | 68.5 / 47.8 |
| E1+E2 | 62.3 ± 0.6 | +0.5 ± 1.2 | 56.6 | 1.92 | 60.5 ± 1.5 | +1.7 ± 0.9 | 57.8 | 64.5 / 51.0 |
| E1+E3 | 63.4 ± 0.3 | +1.7 ± 0.5 | 57.2 | 1.90 | 60.9 ± 1.5 | +2.1 ± 1.5 | 58.6 | 62.2 / 55.0 |
| E1+E2+E3 | 63.2 ± 0.3 | +1.5 ± 0.4 | 56.5 | 1.87 | 60.7 ± 1.4 | +1.9 ± 1.0 | 57.8 | 59.5 / 56.0 |
| E2 AST | 59.9 ± 0.4 | -1.8 ± 0.9 | 49.7 | 2.14 | 58.1 ± 0.7 | -0.7 ± 1.3 | 52.5 | 52.7 / 52.2 |
| E3 HTS-AT | 60.1 ± 1.0 | -1.7 ± 1.0 | 48.3 | 2.18 | 57.2 ± 1.8 | -1.6 ± 2.4 | 50.0 | 40.5 / 59.5 |
| E2+E3 | 60.8 ± 0.9 | -0.9 ± 1.0 | 48.5 | 1.98 | 58.8 ± 1.6 | -0.0 ± 1.9 | 51.9 | 46.7 / 57.1 |
| BEATs PAFA alone, trained SC static | 62.3 ± 0.9 | | 58.9 | 2.33 | 60.3 ± 0.8 | | 61.5 | 88.2 / 34.8 |
| E1 BEATs, SC static | 63.1 ± 0.6 | +0.9 ± 1.1 | 58.0 | 2.06 | 61.9 ± 1.1 | +1.6 ± 1.1 | 61.0 | 72.1 / 49.8 |
| E1+E2, SC static | 62.7 ± 0.3 | +0.5 ± 1.0 | 57.6 | 1.88 | 61.6 ± 1.1 | +1.3 ± 0.9 | 60.4 | 72.4 / 48.3 |
| E1+E3, SC static | 63.2 ± 0.7 | +0.9 ± 1.4 | 57.7 | 1.89 | 62.0 ± 0.7 | +1.7 ± 1.1 | 58.5 | 62.5 / 54.6 |
| E1+E2+E3, SC static | 63.0 ± 0.7 | +0.7 ± 1.1 | 55.9 | 1.85 | 62.1 ± 0.9 | +1.8 ± 0.9 | 59.4 | 65.1 / 53.8 |
| E2 AST, SC static | 60.2 ± 0.7 | -2.1 ± 0.4 | 52.2 | 2.11 | 58.7 ± 0.9 | -1.6 ± 1.2 | 55.8 | 65.6 / 46.1 |
| E3 HTS-AT, SC static | 59.5 ± 0.8 | -2.7 ± 1.4 | 47.9 | 2.20 | 58.7 ± 0.9 | -1.6 ± 1.6 | 50.1 | 39.6 / 60.5 |
| E2+E3, SC static | 60.9 ± 0.4 | -1.4 ± 0.7 | 49.8 | 1.96 | 59.9 ± 0.8 | -0.4 ± 1.4 | 54.2 | 53.2 / 55.1 |

E2, E3 and E2+E3 are below BEATs PAFA or level with it everywhere (official -0.7 to -2.8, LODO macro 0.0 to -2.3,
Litt3200 -5.7 to -11.6); fitted-weight rows, fitted weights and per-device Scores in the detailed report.

### 14.1 Findings (stage 9)

- [Fact] Trainable weights do not help: fitted weights are below equal weights or level with them in every ensemble
  (LODO macro -0.1 to -1.5, official -0.9 to +0.2); a class-balanced NLL objective gives the same picture. The fit gives
  BEATs PAFA the smallest weight (0.07-0.21), although PAFA is the member that holds Sp on the unseen Litt3200.
- [Fact] Equal-weight E1, E1+E3 and E1+E2+E3 beat BEATs PAFA alone by more than one paired seed SD on both the official
  split and the LODO macro without SC (best E1+E3: +1.7 and +2.1); with trained SC the LODO gains stay (+1.6 to +1.8)
  and the official-split gains fall within one SD (+0.7 to +0.9).
- [Fact] Every ensemble loses on Litt3200 (-0.6 to -11.6) and on the official-split worst-device Score (-0.9 to -11.0):
  ensembles trade PAFA's Sp for Se. HTS-AT complements BEATs a little more than AST (E1+E3 >= E1+E2).
- [Fact] Ensembles give smaller conformal sets at the same coverage (1.85-2.18 vs 2.33-2.47 labels on the official split).
- [Assumption] Exploratory: members and groups were chosen after seeing test results. The registered E1-E4 and
  distillation are not run.

### 14.2 Raising Se while keeping Sp; best epoch on test (optimistic)

Rule `k rate` (label-free): the normal probability is scaled so that the ensemble calls the same fraction of the
unlabelled test cycles normal as BEATs PAFA alone does on the same cycles (transductive, like SC). d = paired by seed vs
PAFA alone (Sp and Se in points). Best epoch on test = every member at its own best epoch on test, plain argmax
[optimistic]. Other rules (Sp matched on source-device calibration, oracle on test labels): detailed report, section 5.

| Ensemble (equal, `k rate`) | Official Score | dSp / dSe | LODO macro | dSp / dSe | Litt3200 Sp / Se | Best epoch on test [optimistic] (Sp / Se) |
| --- | --- | --- | --- | --- | --- | --- |
| BEATs PAFA alone, no SC | 61.7 ± 0.7 | | 58.8 ± 1.4 | | 88.5 / 30.9 | 63.2 ± 0.6 (75.5 / 51.0) |
| E1 BEATs | 62.8 ± 0.7 | +0.7 / +1.5 | 60.5 ± 0.9 | +0.9 / +2.6 | 88.2 / 30.2 | 65.1 ± 0.3 (80.1 / 50.1) |
| E1+E2 | 62.4 ± 0.6 | +0.4 / +0.8 | 60.7 ± 1.1 | +1.2 / +2.6 | 88.1 / 29.0 | 65.3 ± 0.5 (83.2 / 47.5) |
| E1+E3 | 63.2 ± 0.4 | +0.6 / +2.3 | 60.9 ± 1.1 | +1.2 / +3.1 | 88.1 / 29.5 | 65.8 ± 0.7 (81.9 / 49.7) |
| E1+E2+E3 | 62.9 ± 0.3 | +0.5 / +1.8 | 61.0 ± 1.1 | +1.4 / +3.1 | 88.0 / 29.1 | 65.8 ± 0.5 (84.0 / 47.6) |
| E2+E3 | 60.7 ± 0.8 | -1.3 / -0.8 | 59.8 ± 1.5 | +0.6 / +1.5 | 86.5 / 26.4 | 64.4 ± 0.7 (83.5 / 45.2) |
| BEATs PAFA alone, trained SC static | 62.3 ± 0.9 | | 60.3 ± 0.8 | | 88.2 / 34.8 | 64.6 ± 0.4 (85.1 / 44.2) |
| E1 BEATs, SC static | 63.4 ± 0.5 | +0.9 / +1.4 | 62.0 ± 0.6 | +1.1 / +2.4 | 88.4 / 36.0 | 65.8 ± 0.5 (84.8 / 46.8) |
| E1+E3, SC static | 63.4 ± 0.4 | +0.6 / +1.6 | 62.4 ± 0.8 | +1.2 / +2.9 | 87.8 / 34.3 | 66.1 ± 0.5 (85.6 / 46.5) |
| E1+E2+E3, SC static | 63.1 ± 0.5 | +0.4 / +1.1 | 62.1 ± 0.6 | +1.0 / +2.5 | 86.9 / 31.8 | 65.8 ± 0.4 (86.1 / 45.6) |
| E2+E3, SC static | 60.8 ± 0.9 | -1.5 / -1.4 | 60.3 ± 0.6 | -0.1 / +0.1 | 84.8 / 26.8 | 64.3 ± 0.5 (82.9 / 45.7) |

- [Fact] With `k rate`, E1, E1+E3 and E1+E2+E3 keep PAFA's Sp (dSp +0.4 to +1.4) and raise Se on both splits (official
  +1.1 to +2.3, LODO macro +2.4 to +3.1); best: E1+E3, LODO Score +2.1 (no SC: 60.9; trained SC: 62.4).
- [Fact] Plain argmax (section 14 table) trades Sp for Se under shift (LODO macro Sp -3 to -6, Litt3200 Sp 59-72); Sp
  matched on source-device calibration does not transfer (held-out Sp -7 to -18). Only the target-side anchor holds Sp.
- [Fact] Litt3200 is the exception: no ensemble raises Se there at PAFA's Sp, except E1 with trained SC (36.0 vs 34.8).
- [Fact] Best epoch on test [optimistic]: E1+E3 65.8 ± 0.7 (no SC) and 66.1 ± 0.5 (trained SC); published BEATs PC-MCL
  65.37 ± 0.73 on the same rule. The test-picked epochs raise Sp, not Se.
- [Assumption] `k rate` needs a batch of unlabelled target cycles and PAFA running beside the ensemble; its behaviour with
  few cycles is not measured.

## 15. Stage 10: Candidates for the final AuscultTrust

Official split = last epoch, 5 seeds; LODO = macro Score over the 4 held-out devices (arm = source + paired change).

| Candidate | Official split Score | LODO macro Score | LODO Litt3200 Score | Note |
| --- | --- | --- | --- | --- |
| BEATs PAFA + trained SC static | 62.3 ± 0.9 | 60.3 ± 0.8 | 61.5 ± 6.0 | best official-split Score; best on the hardest device |
| BEATs SG-SCL + trained SC static | 60.7 ± 1.0 | about 60.5 | about 57.8 | best LODO macro, by a margin within seed SD |
| BEATs Patch-Mix CL (no SC) | 59.7 ± 1.3 | 60.1 ± 1.7 | 57.5 | trained SC adds nothing here |
| BEATs PAFA (no SC) + test-time SC static | 61.7 ± 0.7 | about 58.5 | about 62.1 | no retraining; SC only at test time |
| BEATs PAFA + trained SC dynamic | 62.0 ± 0.5 | 57.9 ± 1.6 | 61.3 ± 3.9 | stage 8; weaker LODO macro (Meditron) |
| BEATs SG-SCL + trained SC dynamic | 62.8 ± 1.4 | 59.5 ± 0.8 | 60.2 ± 5.3 | stage 8; best official-split single cell |
| BEATs PAFA + PC-MCL + trained SC static | 61.8 ± 0.2 | 60.7 ± 1.5 | 59.5 ± 4.2 | stage 8; best LODO macro, smallest sets |
| E1 BEATs ensemble + trained SC static, `k rate` | 63.4 ± 0.5 | 62.0 ± 0.6 | 62.2 | stage 9; 3 models, keeps PAFA's Sp |

[Interpretation] The differences between the BEATs candidates are within one seed SD on every axis; BEATs PAFA + trained
SC is the only one that is first or close to first on all three.

**Decision (user, 22/10): the final AuscultTrust is BEATs + PAFA, fine-tuned.** Stage 8 has ended; the SC variant is the
user's choice: no SC, trained SC static (recommended: best of the PAFA variants on LODO macro, level on the official
split), trained SC dynamic, or test-time SC on the no-SC model. The ensemble and PAFA + PC-MCL rows are alternatives.

### 15.1 Review of stages 0-4 for the chosen configuration (22/10)

- [Fact] The BEATs PAFA numbers of sections 7-9 were recomputed from the `report.json` files (last epoch, 5 seeds):
  official split 61.73 ± 0.67 (no SC) and 62.27 ± 0.86 (trained SC static); LODO macro 58.8 ± 1.4 and 60.3 ± 0.8;
  per fold without / with SC: Meditron 57.2 / 57.6, LittC2SE 62.0 / 67.5, Litt3200 59.7 / 61.5, AKGC417L 56.4 / 54.6.
  They match the tables (the earlier "about 61.6" on Litt3200 is 61.5 from the units directly).
- [Fact] Why BEATs PAFA: the encoder effect dominates (stages 2-3: BEATs first on both epoch rules and in LODO), and PAFA
  is the one consistent method gain on BEATs (+2.6 Score, +8.5 worst-device Score over BEATs CE, paired by seed); in LODO
  it is the best cell on Litt3200, the hardest device (59.7 vs 47.3-57.5).
- [Fact] Its weak side is sensitivity: PAFA raises Sp and lowers Se (official split Sp 73.2, Se 50.2; Litt3200 fold Sp
  88.5, Se 30.9 without SC, 34.8 with SC). In LODO its macro Score without SC (58.8) is the lowest of the BEATs cells
  (58.8-60.1, all within seed SD); trained SC lifts it to 60.3.
- [Fact] Trained SC static on BEATs PAFA is not uniform across devices: +5.5 on LittC2SE, +1.9 on Litt3200 (seed SD
  rises from 3.3 to 6.0), -1.8 on AKGC417L; the official-split worst-device Score does not change (-0.1 ± 1.2).
- [Interpretation] The choice rests on Score and worst-device Score, not on coverage: no LODO v1 fold passes the null
  gate (stage 7), so coverage under device shift is not yet shown for this model. The LODO numbers are selected numbers
  (section 17).

## 16. Stage 11: After LODO [Pending]

KAUH positive controls A and B2 on the final model (frozen checkpoint + SHA-256, saved s_ref), screening view
(recording level, 8 s windows, triage counts), phantom tests (H1, H3, co-primary H2) once the hardware is ready. The web
demo is deployed; its current bundle serves BEATs + PAFA. Needs the final SC variant first (section 15), which the
serving bundle must then match.

## 17. Caveats

- One split and 5 seeds: a 1-point difference is within seed noise for most cells. The last epoch is the reference
  schedule's, not a validated epoch; best-epoch numbers are test-selected.
- Grid cells differ in batch size, cycle length and SpecAugment as well as in the method; only PAFA has a settings
  control (ce_pafa_cfg).
- [Fact] The LODO v1 and SC cells were chosen after seeing official-split test results; the held-out folds contain those
  test cycles (mild leakage, stated in Amendment 4). The final AuscultTrust is selected on the LODO folds, so its LODO
  numbers are selected numbers. The registered AST results stay the registered outcomes.
- No LODO v1 fold passes the null gate, so no coverage claim (H2) is made from LODO v1. The AKGC417L fold is class-mix
  dominated (training keeps at most 321 of 1,864 crackle cycles).
- ICBHI device is confounded with class mix and site. GPU numerics are not bit-reproducible (`cudnn.benchmark`, AMP).
  Units cut off by a crash or kill are rerun from epoch 0, never resumed, so every reported unit is an uninterrupted run.
- SC units ran on three code states (C: commit before the PC-MCL port; D, A, E: current; B: older), identical on every
  path the SC cells use (checked by file hashes and diff).

## 18. Next steps

Current step: stage 10 (final choice). Every training run is done and gathered on A.

1. [User] Choose the SC variant of BEATs PAFA (recommended: trained SC static) and single model vs ensemble
   (section 15). Then freeze it (checkpoint SHA-256, s_ref, softmax export with `export_probs.py`) and make the serving
   bundle of the demo match it.
2. Results report for the submission of 25/10, built from this file (sections 1, 7-15, 17).
3. Ensembles by encoder group done (section 14): equal weights beat learned weights; with the label-free `k rate` rule,
   E1 or E1+E3 raise Se at PAFA's Sp. Open: whether the final AuscultTrust becomes such an ensemble (decision of the
   user), `k rate` with few target cycles, the registered E1-E4, distillation into one BEATs model.
4. With the frozen final model: KAUH positive controls and screening view (recording level); phantom tests (H1, H3)
   once the hardware is ready. A coverage test of SC (H2) now needs the phantom data, earlier-epoch models or
   another dataset, since LODO v1 has no tested fold.
