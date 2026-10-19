# Handoff (19/10/2026 16:50): LODO v1, SC and PC-MCL running on four machines

For the next session. Read this file, then `CLAUDE.md` (repo rules), then as needed:
`docs/updates/lodo-plan-v1.md` (stages), `docs/updates/prereg-v9-amendment-4.md` (LODO v1 + SC),
`docs/updates/prereg-v9-amendment-5.md` (PC-MCL), `docs/reports/grid-v1-results.md` (grid results).
Archived handoffs: `docs/handoff/HANDOFF_2026-10-15-grid.md` (grid monitoring), `HANDOFF_2026-10-14.md`.
Session style asked by the user: terse (caveman) replies, minimal code (ponytail), one runnable check per change.
Commit lines end with `Claude-Session: https://claude.ai/code/session_013aXEm6WMY7UmpWbh6Y6xvL`; push to `origin main`.

## 1. Project state in one paragraph

Grid v1 (70 method x encoder cells, 350 units, ICBHI official split) is done and audited; BEATs is the best encoder
(BEATs PAFA 61.7 ± 0.7 Score at the last epoch, BEATs Patch-Mix CL 63.6 ± 0.8 at the optimistic best epoch). The
registered core of proposal v9 (Gate G passed; H2 partly met on AST, Litt3200 fold only; H1/H3 wait for phantom
hardware) is in `docs/reports/report_2026-10-09.md`. The user treats the old AST AuscultTrust as stale: the final
AuscultTrust configuration (encoder + method + SC) is chosen after LODO v1 and SC (BEATs so far). Now running:
LODO without SC (A, B), trained SC static (C), PC-MCL trials (D).

## 2. Machines and runs

Repo path on all: `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust`; data, model weights, `../repos` and `.venv-train`
were rsynced from A to C and D (ICBHI audio lives in `data/raw/icbhi`; `data/processed` holds symlinks to it).
SSH from A by key to all.

| Machine | Host | GPU | Run / list | Units | Started | Est. end |
|---|---|---|---|---|---|---|
| A | local | RTX 5000 Ada 32 GB | `lodo-v1`, `logs/campaigns/lodo-v1-A.txt` | 75 | 19/10 13:40 | about 21/10 morning |
| B | `namdp15@10.148.183.8` | RTX 5090 32 GB | `lodo-v1`, `lodo-v1-B.txt` | 145 | 19/10 13:40 | about 21/10 morning |
| C | `namdp15@10.148.183.119` | RTX PRO 6000 96 GB | `lodo-v1`, `lodo-v1-C.txt` (SC static) | 175 | 19/10 13:05 | about 22/10 (to re-estimate) |
| D | `namdp15@10.148.183.171` | RTX 5090 32 GB | `pcmcl-v1`, `pcmcl-v1-D.txt` | 10 | 19/10 16:40 | about 20/10 morning |

Status 16:49: A 0/75, B 3/145, C 4/175, D 0/10 done; 0 FAILED, 0 out of memory everywhere.
Logs: sweep `logs/<run>-<M>.log`, jobs `logs/<run>-<M>/<n>.log`, GPU every minute `logs/gpu-<M>.csv`.
Restart command (same on every machine, after a power cut or crash; back up the old sweep log first):
`RUN_ID=<run> SWEEP_TAG=-<M> setsid nohup .venv-train/bin/python scripts/sweep.py logs/campaigns/<run>-<M>.txt --workers <W> [--gpu-gb 29 on C] --ram-gb 6 > logs/<run>-<M>.log 2>&1 < /dev/null &`
(W: A/B 4, C 3, D 2). Over ssh start it with `ssh host 'bash -s' < script.sh`, else the ssh client stays attached.
Finished units are skipped; interrupted ones resume from `last.pth` (RNG state not restored: rerun such units
cleanly at the end if exact runs matter, as `scripts/grid_rerun.py` did for the grid).

**Open issue on C (tell the user first):** the user allowed up to 30% of C's GPU. The sweep caps the summed `MEM_GB`
reservations at 29 GB, but the jobs use more than reserved on this card: at 16:49 three jobs used 52 GB (one 35.6 GB).
Options: restart C's sweep with `--workers 1` (or a smaller `--gpu-gb`), or set a per-process memory fraction. Ask
the user; killing jobs must be by PID only.

**LODO units move to D (19/10 17:05, user request).** `scripts/lodo_move.py` runs on A (log `logs/lodo-v1-move.log`;
started from the worktree copy `.claude/worktrees/lodo-move/scripts/lodo_move.py` with `AT_REPO=<main checkout>` until
the commit reaches `main`). When D's `pcmcl-v1` sweep ends it stops the A and B schedulers by PID (their jobs go on),
moves unstarted units from the end of each list to D until the expected ends are level (measured throughput; D = B's),
writes `logs/campaigns/lodo-v1-{A2,B2,D}.txt` (originals kept), starts schedulers with tags -A2, -B2, -D (new job log
directories), restarts any that dies without its final line, and rsyncs B's and D's `outputs/train/lodo-v1/` to A at the
end. Dry run: `.venv-train/bin/python scripts/lodo_move.py --dry --d-free <hours until D is free>`. At 17:10 it moved
48 (A) + 35 (B) units, all ending about 21/10 01:00 instead of A 22/10 03:00 / B 21/10 05:00. Restart command after a
power cut on A: the same `setsid nohup` line; it skips the move if `logs/campaigns/lodo-v1-D.txt` exists on A.
C estimate (measured 2.2 smoke-h/h, 2 jobs): about 21/10 14:00.

**D note:** one PC-MCL job uses about 19 GB, so with `MEM_GB=15` the sweep's free-memory check keeps D at one job at a
time (about 1.6 h per unit, 10 units about 16 h). Two jobs did fit in the 2-epoch test; raising concurrency is optional.

Consolidated report with plan, timeline and grid results: `report.md` (v3.0, 19/10).

## 3. What each run contains

- `lodo-v1` on A/B (no SC, Amendment 4): 11 cells x 4 held-out devices (Meditron, LittC2SE, Litt3200, AKGC417L) x
  seeds 0-4: AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs ce, patchmix_cl, sg_scl, pafa, qlung.
  Cell names `<grid cell>+lodo_<dev>`; recipe = grid config; fixed last epoch (50; BEATs pafa 100).
- `lodo-v1` on C (trained SC, static bound, Amendment 4): 7 cells (AST patchmix_cl, pafa; HTS-AT patchmix_cl,
  sg_scl; BEATs patchmix_cl, pafa, sg_scl), official split seeds 0-4 + the 4 LODO folds. Cells `<cell>+sc+sc_static`
  and `<cell>+sc+sc_static+lodo_<dev>`. Dynamic bound later:
  `PYTHONPATH=. .venv-train/bin/python scripts/grid.py lodo-jobs --smoke logs/grid_smoke_B.json --names C --sc-cells <7 cells> --sc-mode dynamic --run <run>`
  (writes `<run>-C.txt`; pick a run name that does not overwrite `lodo-v1-C.txt`).
- `pcmcl-v1` on D (Amendment 5): `beats+ft+pcmcl` and `beats+ft+pafa_pcmcl`, official split, seeds 0-4, 100 epochs.
  Enters LODO/SC only if its last-epoch Score beats BEATs PAFA (61.7 ± 0.7) by more than one seed SD.
- Job lists come from `scripts/grid.py lodo-jobs` (balances machines by `logs/grid_smoke_{A,B}.json` x fold cost /
  speed; `SC_COST` 1.05 measured).

## 4. PC-MCL port (method `pcmcl`)

- Code: `src/training/patchmix_cl/pcmcl.py` (training items, `four_class_probs`), branches in `engine.py`,
  `modeling.py`, `dataset.py`; configs `experiments/train/grid/pcmcl_beats.yaml`, `pafa_pcmcl_beats.yaml`; test
  `tests/test_pcmcl.py`. Reference: `../repos/PC-MCL`, arXiv 2601.17080 (65.37 ± 0.73 published, best epoch on test).
- Per epoch: every training cycle (3-label target normal/crackle/wheeze) + 0.5n concatenations of 2-3 short cycles
  (reference "additive" combinations, label OR) + 0.3n patient-matching pairs (same patient 1; hard negative = the
  reference's first pathology group with two patients, in practice two normal-only patients). Loss BCE + 0.1 x CE of
  the matching head (`projector["ssl"]`). With `pcmcl_pafa: 1`, PAFA losses (`projector["pafa"]`, lambda_pcsl 50,
  lambda_gpal 0.0005) on samples of one patient only (multi-patient concatenations carry patient -1).
- Prediction: crackle and wheeze sigmoids as independent events -> 4-class probabilities (argmax = threshold 0.5 rule).
- Settings not in the paper: PAFA's BEATs setting (batch 32, lr 5e-5, 100 epochs, no SpecAugment), 10 s input. The
  train loader runs in the main process (`num_workers` 0; the front end uses the GPU). SC on concatenations is not
  implemented (assert). Checks: 2 epochs of the real schedule gave test Score 59.9 (pcmcl) and 51.0 (pafa_pcmcl).

## 5. Next steps (in order)

1. Resolve the C memory issue with the user (section 2).
2. Monitor (no monitor session runs now). `scripts/grid_status.sh` and `scripts/grid.py audit` hard-code `grid-v1` and
   machines A/B: parametrise them by run and add C and D before handing to a monitor, or check by hand:
   per machine, count `outputs/train/<run>/<cell>/seed<s>/report.json` for the list's units, `grep -c FAILED` and
   `grep -c 'out of memory'` in the sweep log, `tail logs/gpu-<M>.csv`, free disk (A has about 80 GB).
3. When D is done: table for `pcmcl-v1` (Sp, Se, Score, HS, 2-class, worst device, coverage; last and best epoch),
   paired against grid `beats+ft+pafa` and `beats+ft+ce`; apply the Amendment 5 rule.
4. When A/B are done (L1): L2 inference arms with `src.evaluation.lodo_ttsc` (test-time SC static and dynamic, A2,
   TTA-EQ); L3 analysis with `src.evaluation.lodo` (H2 statistics, set size next to |Delta|) and per-fold Sp/Se/Score
   tables, method vs ce on the same encoder; report in `docs/reports/lodo-v1-results.md`.
5. When C is done: SC vs no-SC paired by seed (official split vs grid units; LODO vs A/B units); then SC dynamic.
6. Optional: ensembles E1-E4 (rules fixed in Amendment 4: equal-weight softmax mean per fold and seed; needs a small
   `src/evaluation/ensemble.py` writing unit-format outputs), distillation only if E1 beats the best single cell.
7. Choose the final AuscultTrust configuration with the user; then KAUH positive controls, screening view, phantom.

## 6. Decisions of the user (19/10)

- LODO cells as in section 3 (BEATs ce replaced ce_pafa_cfg). 80/20 split stage dropped. Ensembles/distillation optional.
- SC cells as in section 3; static bound primary, dynamic later. C limited to 30% of its GPU.
- PC-MCL and PAFA + PC-MCL on BEATs on D now (not after LODO).
- Deleted on request (A and B): checkpoints of smoke-*, screen-ast-cv5, grid-v1-rerun, grid-v1-resumed,
  grid-v1-sgscl-ref, repro-wave1*, lung-srad (outputs kept).

## 7. Gotchas

- Kill processes by PID only, never `pkill -f` (over ssh the pattern matches the ssh shell). Never edit a running bash
  script in place. Never reuse a unit with another config (`check_same_config` asserts). No test-set tuning except in
  columns labelled optimistic; claims in docs carry [Fact]/[Interpretation]/... tags.
- The sweep starts a job only if summed `MEM_GB` reservations fit `--gpu-gb` and the GPU has that much free.
  `MEM_GB` comes from A's smoke peaks; real use can be higher on other cards (C).
- `--epochs 1` smoke runs compress the cosine schedule (learning rate near zero): their scores mean nothing; use
  `--stop_epoch N --report_epochs N` to check learning.
- SG-SCL uses `grl_schedule: progress`; QLung skips silent frames; frozen PAFA equals frozen CE.
- Code on B and C predates the `pcmcl` changes (functionally identical for the other methods); D and A have the
  current code. Sync with `rsync -a --exclude __pycache__ src experiments scripts tests <host>:<repo>/` when needed,
  not while a list depends on old behaviour.
- `report.json` holds per-epoch history (train loss/accuracy since 16/10); `test_preds.npz` every epoch's test softmax;
  LODO units also `cal_preds.npz`; SC units `sc_state.npz`.
