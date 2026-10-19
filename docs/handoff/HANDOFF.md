# Handoff (19/10/2026 19:00): LODO v1, SC and PC-MCL running on five machines

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
LODO without SC (A, B, E), trained SC static (C; shared with D after PC-MCL), PC-MCL trials (D).

## 2. Machines and runs (state 19/10 19:00)

Repo path on all: `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust`; data, model weights, `../repos` and `.venv-train`
were rsynced from A to C, D and E (ICBHI audio lives in `data/raw/icbhi`; `data/processed` holds symlinks to it).
SSH from A by key to all. Logs per job list `<L>`: scheduler `logs/<L>.log`, jobs `logs/<L>/<n>.log`; GPU every minute
`logs/gpu-<M>.csv`. Results `outputs/train/<run>/<cell>/seed<s>/` (`report.json` written last = unit done).

| Machine | Host | GPU | Active job list (run `lodo-v1` unless stated) | Units | Est. end |
|---|---|---|---|---|---|
| A | local | RTX 5000 Ada 32 GB | `lodo-v1-A2` (no SC); starts when A's last old job ends | 36 | about 20/10 18:00 |
| B | `namdp15@10.148.183.8` | RTX 5090 32 GB | `lodo-v1-B2` (no SC); starts when B's old jobs end | 89 | about 20/10 18:00 |
| E | `namdp15@10.148.182.179` | RTX 5090 32 GB | `lodo-v1-E2` (no SC, units from A/B lists) | 88 | about 20/10 18:00 |
| C | `namdp15@10.148.183.119` | RTX PRO 6000 96 GB | **dropped 20/10 02:00** (see "C dropped" below); `lodo-v1-C` stopped, 14 of 175 SC units done | 14 | - |
| D | `namdp15@10.148.183.171` | RTX 5090 32 GB | `pcmcl-v1-D` (run `pcmcl-v1`) | 10 | about 20/10 10:45 |

Old lists `lodo-v1-A.txt`, `lodo-v1-B.txt` are kept but their schedulers were stopped on purpose: their unfinished units
are in the `*2` lists. Their last running jobs ("old jobs") finish normally. LODO without SC ends about 20/10 18:00
(±20%). Trained SC (161 units left) is paused until the user confirms the SC cell list after the LODO and PC-MCL
results.

**C dropped (20/10 02:00, user decision).** Our scheduler and jobs on C were killed twice by the machine's user
(`anhdh35`, hostname `VMO-ANHDH35-D`): 19/10 23:52 and 20/10 01:30. Nothing of ours runs there. The C+D dispatcher
(PID 3446984) was stopped before it acted. On C the 5 cut-off units (3 of the second kill) were moved to
`<outputs|checkpoints>/train/lodo-v1-interrupted/` and `logs/campaigns/lodo-v1-C.moved` written. C's 14 finished SC
units (outputs and checkpoints), the interrupted copies, the job logs `logs/lodo-v1-C/`, `logs/lodo-v1-C.log*` and
`logs/gpu-C.csv` were copied to A and verified by checksum (0 differing files). Remaining SC units: the unfinished lines
of `logs/campaigns/lodo-v1-C.txt` (cut-off units rerun from epoch 0). Plan agreed with the user (option 2), to start only
after the user confirms the cell list: D after PC-MCL, then A, B, E once their LODO lists end; the dispatcher needs a new
list tag (not `*2`, used by A/B/E) and must start a machine only after its current list has ended. HTS-AT weights are
not yet on D; SC was smoke-tested on A only (`smoke-lodo-v1`).

**Share dispatchers** (`scripts/lodo_move.py`, running on A; see its docstring). They stop source schedulers by PID
(running jobs go on), keep running/interrupted units on their machine, give the unstarted units, longest first, to the
machine that ends them first (measured throughput), write `logs/campaigns/<run>-<M>2.txt`, start each machine's new
scheduler (tag -<M>2) only after that machine's old jobs have ended, restart a scheduler that dies without its final
line, and rsync the other machines' `outputs/train/lodo-v1/` to A at the end. Plan saved in
`logs/campaigns/lodo-v1-share-<machines>.json`; rerunning the same command resumes from it.
- A+B+E (LODO no SC): PID 3011935, log `logs/lodo-v1-share-ABE.log`, plan applied 19/10 18:42 (A 36, B 89, E 88).
- No SC dispatcher runs (C dropped; SC paused, see above). Earlier C plans (C+D share, move all to D) never acted.
- The A+B+E dispatcher was started from the identical copy in `.claude/worktrees/rebalance/scripts/` with
  `AT_REPO=<this repo>`; restart it (after a power cut on A) with the command in the docstring of `scripts/lodo_move.py`.

**Machine E:** the user's `spatialdb_service` (about 6 GB of GPU) was stopped on request (19/10 18:47, by PID); E uses
the full GPU (scheduler budget 30.8 GB). 1-epoch smoke of one unit per encoder (`smoke-lodo-e`) passed.

**Scheduler restart by hand** (only if a dispatcher is not running; the dispatchers do it themselves):
`RUN_ID=<run> SWEEP_TAG=-<tag> setsid nohup .venv-train/bin/python scripts/sweep.py logs/campaigns/<run>-<tag>.txt --workers 4 [--gpu-gb 34 on C] --ram-gb 6 >> logs/<run>-<tag>.log 2>&1 < /dev/null &`
(`<tag>` = A2, B2, C2, D2, E2; `pcmcl-v1-D`: `--workers 2`). Over ssh start it with `ssh host 'bash -s' < script.sh`, else
the ssh client stays attached. Finished units are skipped; interrupted ones resume from `last.pth` (RNG state not
restored: rerun such units cleanly at the end if exact runs matter, as `scripts/grid_rerun.py` did for the grid).

**C's GPU share:** the user allows about 30% of C's 96 GB (3-4 jobs). Since 20/10 01:20 C's scheduler runs
`--workers 4 --gpu-gb 34` (reservations: 3 BEATs jobs at 11 GB, or 4 with HTS-AT units); actual use about 8.3 GB per BEATs
job (3 jobs about 25 GB). No hard per-process cap: on 19/10 16:49 one job grew to 35.6 GB, so check `logs/gpu-C.csv`
(report if our jobs together pass about 35 GB).

**D note:** one PC-MCL job uses about 19 GB, so `pcmcl-v1-D` runs one job at a time (about 1.9 h per unit).

Consolidated report with plan, timeline and grid results: `report.md` (v3.0, 19/10).

## 2a. Monitoring (for the monitor session)

Run `bash scripts/grid_status.sh` on A (read-only; covers A-E and both dispatchers; about 1 min) every hour and append
one line per run to `docs/handoff/monitor_log.md` (`dd/mm/yyyy HH:MM | ...`). `grid-v1` is finished: stop checking it.

Normal, no action:
- `lodo-v1-A`, `lodo-v1-B` "superseded ... stopped on purpose"; `lodo-v1-A2`/`B2` "not started yet" while that
  machine still trains old units (shown under "training now").
- `lodo-v1-C` "units moved ... stopped on purpose" with 14 done: C is dropped, no SC runs until the user confirms.
  Do not restart anything on C. `pcmcl-v1-D` "NOT RUNNING" after its final line: D idles until then (expected).
- `pcmcl-v1-D` "NOT RUNNING" after its final line `all jobs finished, 0 failed`.
- Out-of-memory requeues: the scheduler retries them; report the count only.

Report to the user at once (do not fix, do not kill, do not restart anything yourself):
- `FAILED` > 0 in any list; any NaN or collapsed job log.
- A `*2` list or `lodo-v1-C` / `pcmcl-v1-D` with "scheduler NOT RUNNING" and no `all jobs finished` line, or a machine
  with an empty "training now" while its active list has units left.
- A dispatcher PID missing on A before its log says `ALL DONE`; a dispatcher log line with `Traceback`, `ended with units
  missing` or `restarting`.
- `ssh failed` twice in a row for a machine (power cut or network).
- Disk free below 20 GB on any machine; done counts that do not grow for 3 hours.
- Milestones to report: PC-MCL done (about 20/10 10:45; then ask the user about the SC cell list); A+B+E done
  (about 20/10 18:00); `ALL DONE` lines. Our processes on C or any machine killed by
  someone else (scheduler missing, no final line): report at once.

## 3. What each run contains

- `lodo-v1` on A/B/E (no SC, Amendment 4): 11 cells x 4 held-out devices (Meditron, LittC2SE, Litt3200, AKGC417L) x
  seeds 0-4: AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs ce, patchmix_cl, sg_scl, pafa, qlung.
  Cell names `<grid cell>+lodo_<dev>`; recipe = grid config; fixed last epoch (50; BEATs pafa 100).
- `lodo-v1` on C, shared with D after PC-MCL (trained SC, static bound, Amendment 4): 7 cells (AST patchmix_cl, pafa; HTS-AT patchmix_cl,
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
2. Monitor: section 2a (`scripts/grid_status.sh` covers every machine and both dispatchers).
3. When D is done: table for `pcmcl-v1` (Sp, Se, Score, HS, 2-class, worst device, coverage; last and best epoch),
   paired against grid `beats+ft+pafa` and `beats+ft+ce`; apply the Amendment 5 rule.
4. When A/B/E are done (L1): L2 inference arms with `src.evaluation.lodo_ttsc` (test-time SC static and dynamic, A2,
   TTA-EQ); L3 analysis with `src.evaluation.lodo` (H2 statistics, set size next to |Delta|) and per-fold Sp/Se/Score
   tables, method vs ce on the same encoder; report in `docs/reports/lodo-v1-results.md`.
5. When C is done: SC vs no-SC paired by seed (official split vs grid units; LODO vs A/B/E units); then SC dynamic.
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
