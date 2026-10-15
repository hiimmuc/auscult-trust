# Stage plan v1: method x encoder grid (written 15/10/2026)

Scope: the stages before LODO. LODO scope and the definition of "best" are decided after the grid table is in.

## Grid (run `grid-v1`, launched 15/10/2026 on both machines)

- Data: ICBHI official 60/40 split, 4-class training; 2-class view of the same predictions in the table. Seeds 0-4.
- Encoders: AST, HTS-AT, OPERA-CT, BEATs, DASS.
  OPERA-CT, not OPERA-CE: the OPERA paper ranks CT best on health tasks (mean reciprocal rank 0.694 vs 0.469 for CE). Full
  fine-tuning helps CT (T7 AUROC 0.855 to 0.957) and hurts CE (0.872 to 0.808; the authors fine-tune only CE's first 5
  blocks). Our frozen ladder also had CE below CT on ICBHI 4-class (39.4 vs 42.9).
- Fine-tuned cells (8 x 5): CE, Patch-Mix, Patch-Mix CL, PAFA, SG-SCL, QLung, ADD-RSC, CE at the PAFA setting (control).
- Frozen cells (6 x 5): CE, Patch-Mix, PAFA, QLung, ADD-RSC, CE at the PAFA setting.
  Not frozen: Patch-Mix CL and SG-SCL. Their extra loss acts on a projector that the classifier does not use, so with a
  frozen encoder they reduce to CE. Patch-Mix mixes input patches and still changes training when frozen. ADD-RSC trains
  its denoising layer in front of the frozen encoder. QLung trains a margin-based cosine head.
- 70 cells, 350 runs. Cell name `<encoder>+<ft|frozen>+<method>`; settings in `experiments/train/grid/<method>.yaml`.
- Every run trains to its last epoch (`--selection fixed --report_epochs <last>`) and keeps the test predictions of every
  epoch (`test_preds.npz`). Any epoch rule chosen later can be applied without retraining. Checkpoint kept: final weights only.

### Settings per method (from each method's reference training script)

| Method | Batch | Epochs | Cycle | SpecAugment | Other |
|---|---|---|---|---|---|
| CE, Patch-Mix, Patch-Mix CL | 8 | 50 | 8 s | on | CL weight 1.0, temperature 0.06 |
| PAFA (AST setting) | 32 | 50 | 5 s | off | lambda_pcsl 5.0, lambda_gpal 0.005 |
| PAFA (BEATs setting) | 32 | 100 | 5 s | off | lambda_pcsl 50.0, lambda_gpal 0.0005 |
| CE control at PAFA setting | 32 | 50 (BEATs 100) | 5 s | off | |
| SG-SCL | 8 | 50 | 8 s | two views | device SupCon weight 1.0, temperature 0.06, gradient reversal |
| QLung | 8 | 50 | 8 s | on | margin weight 0.4, gamma 0.5, m_target 0.2, moving average 0.6 |
| ADD-RSC | 8 | 50 | 8 s | on | bias denoising loss weight 0.5, 8 heads |

Shared: Adam, lr 5e-5 (frozen cells 1e-3), weight decay 1e-6, cosine schedule, moving average 0.5 unless stated.

Noise injection: none of these methods uses it. ADD-RSC (paper section 3.1.3; `raw_augment` default 0) and QLung (SpecAugment
only) do not, so no noise arm is run.

### Deviations from the reference code

- HTS-AT, OPERA-CT and DASS have a fixed input length, so their PAFA cells keep 8 s cycles; AST and BEATs use PAFA's 5 s.
- ADD-RSC: the reference resizes the AST image to 1024 x 256 and uses its own AST (patch stride 16). Here the denoising
  layer runs on each encoder's own image and encoder, so the diagonal cell may land below the published 65.53.
- QLung: the reference quality score is NaN for 71 cycles (14 training, 57 test). They are the first cycles of AKGC417L
  multi-channel recordings that start with 4,096 digital-zero samples, so their first STFT frame is all zero and its
  entropy is 0/0. This was reproduced with the reference code itself (`util/icbhi_util.py` cycle extraction and
  `util/get_acoustic_features.py`). In the reference loop one NaN quality makes the whole batch loss NaN, and the AMP
  GradScaler then skips that step and halves its scale (about 14 batches per epoch). A simulation of the scaler with
  these counts reaches scale 0 near epoch 10, after which the weights become NaN. So the reference run as released can
  only have produced its best epoch early. Here the silent frames are left out of the frame average: no NaN, and every
  step trains.
- SG-SCL: SpecAugment masks are drawn per sample, not once per batch.
- Lung-SRAD is not in the grid. Its published number is the reference row; it is reproduced only if it beats the grid.

## Machines and estimate

`scripts/grid.py smoke` ran a few synthetic training steps of every cell on both machines: all 70 cells passed on A and B.
Peak memory and step time per cell are in `logs/grid_smoke_{A,B}.json`. Job lists are balanced by measured hours: A has
149 units, B 201 units, each about 113 GPU-hours if run one at a time. Order on both machines: the reference cells
(AST methods and PAFA on BEATs) first, so the port checks come first; then encoder by encoder (BEATs, HTS-AT, OPERA-CT),
DASS last (`scripts/grid.py jobs --pin`, which keeps already started units on their machine).

DASS fine-tuning is the bulk: about 2.4-3.5 h per run against 0.3-1.5 h for the other encoders.

With 4 parallel jobs per GPU (measured gain about 1.3-1.5x), the grid should end around 19-20/10/2026.

## Next steps

1. When the grid is in: `python scripts/grid.py table --epoch last` (and `--epoch best_test`, labelled optimistic) after
   syncing B's outputs to A. The user then chooses the definition of "best".
2. Top 3 cells x 80/20 split: `--test_fold 0-4`, 5 seeds per fold. No reference paper states the seeds per fold, so 5 are
   used. Fold sizes from `patient_list_foldwise.txt` are 80/20 in patients, not in cycles (fold 0: 3,443 train vs 3,455
   test cycles).
3. LODO scope from the table.

## Commands

Monitoring is handed to a separate session: `docs/handoff/HANDOFF.md`, status with `bash scripts/grid_status.sh`.

- Status: `tail logs/grid-v1-A.log` on A, `tail logs/grid-v1-B.log` on B; one log per job in `logs/grid-v1-{A,B}/`.
- Restart after a stop (finished units are skipped, interrupted ones resume from `last.pth`):
  `RUN_ID=grid-v1 SWEEP_TAG=-A setsid nohup .venv-train/bin/python scripts/sweep.py logs/campaigns/grid-v1-A.txt --workers 4 --ram-gb 6 > logs/grid-v1-A.log 2>&1 < /dev/null &`
  (B: same with `-B`).
- Sync results B to A: `rsync -a namdp15@10.148.183.8:Workspace/Personal/KHKT-CVA-2026/auscult-trust/outputs/train/grid-v1/ outputs/train/grid-v1/`.
