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

- LODO cells (11): AST ce, pafa, patchmix_cl; HTS-AT ce, patchmix_cl, sg_scl; BEATs patchmix_cl, sg_scl, pafa,
  ce_pafa_cfg, qlung. No DASS, no OPERA-CT. BEATs CE not included (open).
- Ensemble rule fixed before LODO: E1 = equal-weight mean softmax of the 11 cells per fold and seed (primary); E2 BEATs
  5; E3/E4 AST 3, HTS-AT 3. Then distillation into one BEATs model.

## Next steps (stage L0 of `docs/updates/lodo-plan-v1.md`)

1. Write the LODO job lists: per cell, the grid job line (`scripts/grid.py` `cells()` gives configs and extra args) with
   `experiments/train/lodo_<dev>.yaml` appended to `--config`, cell `<cell>+lodo_<dev>`, `RUN_ID=lodo-v1`, seeds 0-4,
   `--selection fixed --report_epochs <last epoch>`. Balance A/B by `logs/grid_smoke_{A,B}.json` run hours x fold factor
   (Meditron 0.83, LittC2SE 0.94, Litt3200 0.94, AKGC417L 0.70) and measured speed (A 1.17, B 1.73). A small `grid.py`
   subcommand is the natural place (reuse `write_jobs`).
2. Smoke: one epoch (`--stop_epoch 1`) of one cell per encoder and method family on each fold, on both machines. Check
   PAFA (patient ids), SG-SCL (3 source devices), QLung (quality cache on the LODO split), 5 s cycles, `cal_preds.npz`.
3. Start the GPU logger on both machines (command in the plan), then the sweeps:
   `RUN_ID=lodo-v1 SWEEP_TAG=-<M> setsid nohup .venv-train/bin/python scripts/sweep.py logs/campaigns/lodo-v1-<M>.txt --workers 4 --ram-gb 6 > logs/lodo-v1-<M>.log 2>&1 < /dev/null &`
4. Monitoring: adapt `scripts/grid_status.sh` and `scripts/grid.py audit` to run `lodo-v1` (both hard-code `grid-v1`),
   or pass the run as an argument. A Haiku monitor can follow the archived grid handoff with those names.
5. After L1: stages L2-L6 of the plan.

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
