# Handoff (19/10/2026): grid v1 done, LODO v1 next

For the next working session. Read this, then `docs/updates/lodo-plan-v1.md`. Repo rules are in `CLAUDE.md`.
The grid monitoring handoff is archived as `docs/handoff/HANDOFF_2026-10-15-grid.md`.

## Where things are

- Grid v1 (70 method x encoder cells, 350 units, ICBHI official split) is complete and audited.
  Report: `docs/reports/grid-v1-results.md`. Tables: `outputs/train/grid-v1/table_{last,best_test}.md`.
  Plan of record for the grid: `docs/updates/stage-plan-v1.md`.
- Both machines are idle. No sweep, dispatcher or monitor runs.
  - A: this machine, RTX 5000 Ada 32 GB. B: `ssh namdp15@10.148.183.8`, RTX 5090 32 GB. Repo path
    `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust` on both. A holds all grid outputs (B's synced); checkpoints stay on
    the machine that trained them.
- Code synced to B with: `rsync -a --exclude __pycache__ src experiments scripts tests namdp15@10.148.183.8:Workspace/Personal/KHKT-CVA-2026/auscult-trust/`
- Tests: `MPLBACKEND=Agg PYTHONPATH=. .venv-train/bin/python -m pytest -q tests` (50 pass). DASS runs in `.venv-dass`
  (grid job lines carry the interpreter).

## Decisions taken by the user

- LODO cells: AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs patchmix_cl, sg_scl, pafa, ce_pafa_cfg,
  qlung. No DASS, no OPERA-CT. The draft lists also hold BEATs ce (control) and trained SC static on 3 cells: both
  proposed, pending the user's step-1 confirmation (remove with `grid.py lodo-jobs` arguments if declined).
- Ensembles (E1-E4) and distillation are optional, after the primary LODO analysis. The 80/20 step is dropped.

## Next steps

State 19/10 13:10 (Amendment 4 registered, `docs/updates/prereg-v9-amendment-4.md`):
- **Machine C** (`ssh namdp15@10.148.183.119`, key login, RTX PRO 6000 96 GB, same repo path; data, models, repos and
  `.venv-train` synced from A; ICBHI raw audio in `data/raw/icbhi` because `data/processed` holds symlinks): trained SC,
  static bound, 7 cells (AST patchmix_cl, pafa; HTS-AT patchmix_cl, sg_scl; BEATs patchmix_cl, pafa, sg_scl), official
  split + 4 LODO folds, 175 units, `logs/campaigns/lodo-v1-C.txt`. Running since 13:05:
  `RUN_ID=lodo-v1 SWEEP_TAG=-C ... sweep.py logs/campaigns/lodo-v1-C.txt --workers 3 --gpu-gb 29 --ram-gb 6`, log
  `logs/lodo-v1-C.log`, GPU log `logs/gpu-C.csv`. The user allowed up to 30% of C's GPU: memory budget 29 GB (compute
  share is not capped). Dynamic-bound SC comes after (`grid.py lodo-jobs --sc-mode dynamic --names C ...`).
- **Machines A and B**: LODO without SC, 11 cells (AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs ce,
  patchmix_cl, sg_scl, pafa, qlung), 220 units, `logs/campaigns/lodo-v1-{A,B}.txt`, about 44 h. NOT started: waiting
  for the user's go. Start: GPU logger, then
  `RUN_ID=lodo-v1 SWEEP_TAG=-<M> setsid nohup .venv-train/bin/python scripts/sweep.py logs/campaigns/lodo-v1-<M>.txt --workers 4 --ram-gb 6 > logs/lodo-v1-<M>.log 2>&1 < /dev/null &`
  (sync code and the B list to B first; over ssh use `ssh ... 'bash -s' < script` so the client detaches).
- Smoke runs passed on A, B and C (`smoke-lodo-v1`, `-b`, `-c`).
- Monitoring: `scripts/grid_status.sh` and `grid.py audit` hard-code `grid-v1` and machines A/B; parametrise by run and
  add C before handing to a monitor.
- Then stages L2-L3 of `docs/updates/lodo-plan-v1.md`; ensembles and distillation optional. The final AuscultTrust
  configuration is chosen after LODO and SC (BEATs so far).
- A and B LODO started 19/10 13:40 (`logs/lodo-v1-{A,B}.log`, GPU logs `logs/gpu-{A,B}.csv`).
- PC-MCL ported (method `pcmcl`, `src/training/patchmix_cl/pcmcl.py`, `experiments/train/grid/pcmcl_beats.yaml`, test
  `tests/test_pcmcl.py`); real-data item check and a GPU train/eval step passed on A. PENDING until LODO v1 is done:
  then a 1-epoch smoke (measure peak memory for MEM_GB and time), register `docs/updates/prereg-v9-amendment-5.md`, run
  `beats+ft+pcmcl` seeds 0-4 on the official split (run `pcmcl-v1`). The pcmcl code is on A only (B and C run the
  previous, functionally identical code for the other methods); sync before running it elsewhere.
- Deleted 19/10 on user request (A and B): checkpoints of smoke-*, screen-ast-cv5, grid-v1-rerun (identical copies in
  grid-v1), grid-v1-resumed, grid-v1-sgscl-ref, repro-wave1*, lung-srad. Outputs of those runs are kept.

## Gotchas learned in the grid

- Kill processes by PID only (never `pkill -f`: over ssh the pattern matches the ssh shell). Never edit a running bash
  script in place. Never reuse a unit with another config (`check_same_config` asserts).
- The sweep starts a job only when the summed `MEM_GB` reservations fit the GPU (`--gpu-gb`, default total - 1 GB).
  `MEM_GB = max(4, smoke peak x 1.25 + 1.5)`. Fewer than 4 jobs running is normal.
- Sweeps do not survive a reboot. After a power cut: back up `logs/<run>-<M>.log`, restart the same command; units
  resume from `last.pth`. Resume does not restore the RNG state: rerun resumed units cleanly if exact runs matter
  (`scripts/grid_rerun.py` did this for the grid).
- SG-SCL uses `grl_schedule: progress` (the reference schedule collapses BEATs). QLung skips silent frames (reference NaN).
- Frozen PAFA = frozen CE (projector-only losses).
- `report.json` is written before `last.pth` is removed; its history carries train loss and accuracy per epoch.
- No test-set tuning except in columns labelled optimistic. Claims in docs use [Fact]/[Interpretation]/... tags.
