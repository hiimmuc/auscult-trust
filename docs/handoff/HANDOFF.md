# Handoff: monitoring the grid-v1 sweep (from 15/10/2026)

This file is for the session that monitors training. Read it fully before acting. The plan is in
`docs/updates/stage-plan-v1.md`; the previous handoff is `docs/handoff/HANDOFF_2026-10-14.md` (those campaigns were stopped).

## What runs

- Run `grid-v1`: 70 cells (method x encoder x fine-tuned/frozen) x seeds 0-4 = 350 training units on the ICBHI official
  60/40 split.
- Two machines, each with one sweep that starts up to 4 jobs at a time on its GPU. The repo path is
  `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust` on both.
  - **A** (this machine, RTX 5000 Ada): job list `logs/campaigns/grid-v1-A.txt` (120 units since 16/10), sweep log `logs/grid-v1-A.log`,
    one log per job in `logs/grid-v1-A/<n>.log`.
  - **B** (`ssh namdp15@10.148.183.8`, RTX 5090): `logs/campaigns/grid-v1-B.txt` (230 units since 16/10), `logs/grid-v1-B.log`,
    `logs/grid-v1-B/<n>.log`.
- Order on both machines: the AST cells and PAFA on BEATs first, then BEATs, HTS-AT and OPERA-CT, then DASS last. DASS
  fine-tuning runs take 2.5-3.5 h each; the others take 0.1-1.5 h.
- Expected end: about 19/10/2026 evening (lists rebalanced 16/10; the job logs before that are in
  `logs/grid-v1-{A,B}.v1/`).
- A unit is finished when `outputs/train/grid-v1/<cell>/seed<s>/report.json` exists. An interrupted unit resumes from
  `checkpoints/train/grid-v1/<cell>/seed<s>/last.pth` on the same machine.

## Each check (every 30-60 minutes)

1. Run `bash scripts/grid_status.sh` on A. It prints, for both machines:
   - units done / total
   - whether the sweep is alive
   - the units running now
   - jobs the sweep gave up on (FAILED) and out-of-memory requeues
   - job logs with a NaN loss or Score
   - GPU use and free disk
2. Every 3rd check, also run the audit on both machines (read-only, seconds):
   `PYTHONPATH=. .venv-train/bin/python scripts/grid.py audit --list logs/campaigns/grid-v1-<M>.txt` (B through ssh in its
   repo). Report any line other than `... 0 problems` to the user.
3. If everything is normal, write one line in `docs/handoff/monitor_log.md`: date and time, done counts A/B, anything odd.
   Do nothing else.

## Allowed actions (only these)

- **A sweep is not running and its units are not all done.** Restart it with the same command. Finished units are
  skipped and interrupted ones resume.
  - A: `cd ~/Workspace/Personal/KHKT-CVA-2026/auscult-trust && RUN_ID=grid-v1 SWEEP_TAG=-A setsid nohup .venv-train/bin/python scripts/sweep.py logs/campaigns/grid-v1-A.txt --workers 4 --ram-gb 6 > logs/grid-v1-A.log 2>&1 < /dev/null &`
  - B (run on B through ssh): same command with `-B` and `grid-v1-B.txt` / `grid-v1-B.log`.
  - Before restarting, copy the old sweep log to `logs/grid-v1-<M>.<date>.log` so it is not overwritten.
- **A job is FAILED.** Read the last 30 lines of its job log (the job number is in the FAILED line).
  - If the cause is memory (`out of memory`, exit 137) or a lost connection: restart that machine's sweep once all its
    other jobs have finished. The restart picks up unfinished units.
  - Any other error: do not retry. Report it to the user with the exact error line.
- **Reruns and the end of the grid are automatic.** 13 units were resumed from `last.pth` after interruptions, so they
  are rerun without interruption under run `grid-v1-rerun` (lists `logs/campaigns/grid-v1-rerun-{A,B}.txt`). A dispatcher
  on A, `scripts/grid_rerun.py` (log `logs/grid-v1-rerun-dispatch.log`), waits until one machine's grid units are all done,
  splits the reruns between the machines (plan in `logs/campaigns/grid-v1-rerun-plan.json`), starts them, then moves the
  resumed units to `grid-v1-resumed/`, copies the reruns in, audits, syncs B to A and builds the tables. Its last line is
  `ALL DONE` when the tables exist. Do not run the audit/rsync/table steps yourself and do not start rerun sweeps by hand.
  - Each check: the dispatcher must be alive (`ps -eo pid,args | grep [g]rid_rerun.py` on A). If it is not and its log does
    not end with `ALL DONE`, restart it (it resumes from the saved plan):
    `cd ~/Workspace/Personal/KHKT-CVA-2026/auscult-trust && setsid nohup .venv-train/bin/python scripts/grid_rerun.py >> logs/grid-v1-rerun-dispatch.log 2>&1 < /dev/null &`
  - A grid sweep that is not running while its grid units are all done is normal from then on; do not restart it.
  - The rerun sweeps log to `logs/grid-v1-rerun-{A,B}.log`, job logs in `logs/grid-v1-rerun-{A,B}/`.
  - The log ends with an error line ("audit found problems", "not clean", a traceback): report it to the user and stop.
  - `ALL DONE`: tell the user. Do not choose the "best" cells: the user decides the rule.

## Never

- Never edit anything in `src/`, `experiments/`, `scripts/` or the job lists, and never start other training.
- Never kill processes by pattern (`pkill -f`, `killall`). Over ssh the pattern matches the ssh shell itself. If a process
  must stop, kill it by PID, and only after asking the user.
- Never delete outputs, checkpoints or logs.
- Never run two sweeps of the same job list at once (check the sweep pid in the status output first).

## Report to the user, without acting

- A FAILED job that is not out of memory.
- Any out-of-memory requeue (count above 0 in the status output). The memory reservations should prevent them.
- A collapsed job log (status line "job logs collapsed"): test Se below 1% from epoch 3 on, i.e. the model predicts one
  class. Seen with SG-SCL on BEATs under the reference reversal schedule; the grid now uses the ramp.
- A NaN loss or NaN Score in a job log. QLung handles silent frames, but report it anyway.
- The same sweep dying twice.
- Less than 30 GB free disk on either machine.
- B unreachable for more than 2 checks in a row.
- Progress far slower than expected: fewer than about 60 units done per machine by 17/10 evening.

## Context the monitor may need

- The 2-class Sp equals the 4-class Sp by construction; it is not a bug.
- The sweep only starts a job when its reservation (per-job `MEM_GB=` prefix, smoke peak x 1.25 + 1.5 GB) plus the
  reservations of all units running now fit in the GPU total minus 1 GB, and that much memory is free. Since 15/10 evening
  no out-of-memory is expected; report any `out of memory` line to the user (the sweep still requeues it). Jobs waiting
  while the GPU is busy, and fewer than 4 jobs running, are normal.
- QLung reference bug: in the reference code, 71 AKGC417L cycles (14 train, 57 test) get a NaN quality score. Those
  recordings start with 4,096 digital-zero samples. Our port leaves the silent frames out, so no NaN is expected.
