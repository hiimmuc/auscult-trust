# Handoff: state on 2026-10-08 17:30 and what to do next

Read this first when resuming. Everything below is verified at the time stated; running jobs keep going without a Claude session.

## Goal and deadline

- **Results report due 2026-10-09 17:00. English Markdown only** (no docx/pdf). School round talk and poster on 2026-10-17.
- **Scope agreed with the user (cut):** Part 1 encoder comparison (done), Part 2 proposed methods with final refit (running), leave-one-device-out (LODO, AST only, running), KAUH positive controls A and B2 with frozen AST models. **Out of scope, mark [Pending] in the report:** RespireNet, pretrained Lung-SRAD/Patch-Mix CL checkpoints (our AST Patch-Mix CL baseline and plain DASS are the proxies, say so), phantom recordings, SC arms for other encoders, KAUH test B1 and C.
- Existing report to extend: `docs/report_2026-10-08.md` (Part 1 tables, paper comparison, screening results, status). Final report = this file updated with the refit, LODO and KAUH sections; name it `docs/report_2026-10-09.md`.

## Machines (both RTX 32 GB, repo path `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust`)

- **local** (this machine, RTX 5000 Ada): ~350 s per epoch per job with 4 jobs; no DASS kernel needed any more.
- **remote** `namdp15@10.148.183.8` (RTX 5090, 24 cores, 125 GB): ~2x faster. Password login only (password given by the user in chat, not stored here); ask the user to run `! ssh-copy-id namdp15@10.148.183.8` once so a new session can use keys. `10.148.182.97` (8 GB) is synced but deliberately unused. The old SSH helper lived in the previous session's scratchpad and is gone.
- Remote needs the same code: edit locally, then `rsync -a --exclude __pycache__ src experiments scripts tests namdp15@10.148.183.8:Workspace/Personal/KHKT-CVA-2026/auscult-trust/`. Remote venvs: `.venv-train` (everything), `.venv-dass` (DASS only, `LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64`).
- Spectrogram caches in `data/cache/train_images/` on each machine (built once; AST `ast_all_*` is the LODO cache).

## What is running (schedulers survive the session; children are independent)

| Campaign | RUN_ID | Where | Job list | Log |
|---|---|---|---|---|
| Final refit, seeds 0-4, 4 arms (baseline epoch 2, freq_mixstyle 3, sc+sc_static 14, sc_gain 2), 20 runs | `screen-ast` | local 10 jobs, remote 10 jobs | `logs/screen-ast/jobs-final-local4.txt`, remote `jobs-final-remote-main.txt` | `logs/screen-ast-final-local7.log`, remote `logs/screen-ast-final-remote8.log` |
| LODO, AST, 4 devices x {baseline, freq_mixstyle, sc+sc_static} x seeds 0-4, 60 runs (stop at the report epoch) | `lodo-ast` | local 20, remote 40 | `logs/lodo-ast/jobs-local.txt`, `jobs-remote.txt` | `logs/lodo-local.log`, remote `logs/lodo-remote.log` |
| Final refit extra seeds 5-9 (baseline, sc+sc_static) for gate G | `screen-ast` | remote only | `logs/screen-ast/jobs-final-remote-late.txt` | starts automatically when the remote LODO scheduler exits (`/tmp/late.sh` waiter) |

Expected finish (estimates): refit seeds 0-4 about midnight, LODO about 02:00-05:00, seeds 5-9 later (drop them if short of time and say so).
Check progress: `ls outputs/train/screen-ast/*/seed*/report.json | wc -l` (20 for seeds 0-4 across both machines), `ls outputs/train/lodo-ast/*/seed*/report.json | wc -l` (60 across both).
Resume after a crash (same RUN_ID, finished units are skipped, interrupted ones resume from `last.pth`): rerun the same `scripts/sweep.py` command with the same job list, e.g. `SWEEP_TAG=-x RUN_ID=screen-ast nohup .venv-train/bin/python scripts/sweep.py logs/screen-ast/jobs-final-local4.txt --workers 4 --mem-gb 7 --ram-gb 8 > logs/x.log 2>&1 &`. Never `pkill -f` a pattern in your own command line; kill schedulers by PID (children keep running).

## Steps still to do, in order

1. **Gather results on one machine** (needed by the summaries): `rsync -a --ignore-existing -e ssh namdp15@10.148.183.8:Workspace/Personal/KHKT-CVA-2026/auscult-trust/outputs/train/screen-ast/ outputs/train/screen-ast/` and the same for `outputs/train/lodo-ast/`. Checkpoints stay where they are (KAUH inference runs on the machine that holds them).
2. **Stage 1 report:** `RUN_ID=screen-ast bash scripts/run.sh report` -> `outputs/train/screen-ast/final.md` (final table of the four arms, paired differences, non-inferiority check of AuscultTrust vs baseline, margin -1.5 Score points). With only seeds 0-4 the check is over 5 seeds (registered: 10).
3. **LODO analysis:** `.venv/bin/python -m src.evaluation.lodo --run lodo-ast` -> `outputs/train/lodo-ast/lodo_summary.md` (coverage per held-out device, null gate, H2 statistic D for AuscultTrust and Freq-MixStyle vs baseline, also without AKGC417L).
4. **KAUH positive controls:** on each machine holding refit checkpoints: `RUN_ID=screen-ast OUT=final bash scripts/run_kauh.sh` (conditions p0, p0_sc = isolation contrast, auscult), copy `outputs/kauh/final/` together, then `bash scripts/run_kauh.sh analyse` -> `outputs/kauh/final/kauh_summary.md` (A: flip rate and TV distance across Bell/Diaphragm/Extended; B2: conformal coverage per ordered filter pair, D statistic). The one-seed smoke test gave flip rate 0.169 without SC and 0.097 with SC (pipeline check only).
5. **Write the report** (`docs/report_2026-10-09.md`): update Part 1 (unchanged, already complete), Part 2 with the refit table and non-inferiority check, new LODO and KAUH sections (method, tables, honest reading, caveats), the scope cut with [Pending] items, reproducibility. Claims tagged [Fact]/[Interpretation]/[Pending]. Commit.

## Key numbers already in `docs/report_2026-10-08.md`

Part 1 table for six encoders (AST/HTS-AT/DASS fine-tune 60.5-61.6 Score best-on-test, last-epoch ~3 points lower; frozen 49.6-57.3), paper comparison (AST Patch-Mix 59.5 vs 59.46; Patch-Mix CL 61.4 vs 62.37), screening: AuscultTrust = SC with static bound (CV Score 60.51 at epoch 14, baseline 58.97 at epoch 2, gap +1.54, 3 folds only, caveats listed there).

## Code added in this session (all committed on `main`, 35+ tests pass)

`scripts/run.sh` (steps `screen summary final report reproduce resume lodo`, DRY mode), `scripts/sweep.py` (memory-gated parallel scheduler, retries, skip finished/running units), `scripts/run_kauh.sh`, `src/training/patchmix_cl/encoders/` (AST, HTS-AT, OPERA-CT, CLAP, HeAR, DASS adapters with Patch-Mix), LODO training path (`--holdout_device`, `lodo_split` in `src/processing/splits.py`, calibration and test softmax saved), `src/evaluation/lodo.py`, `src/evaluation/kauh_infer.py`, `src/evaluation/kauh.py`. Decision log: `docs/decision_logs/2026-10-07-training-sweep.md`.

## Known risks

- Local GPU is slow; if the refit lags, move jobs to the remote (edit job lists, restart schedulers by PID).
- The LODO and KAUH analyses have only been tested on synthetic data and one smoke unit; check the first real outputs for sanity (coverage near 0.9 on source devices, null p95 plausible).
- Best CV epochs are very early (2-3) for several arms: the fixed-epoch refit reads the epoch-2/3 checkpoint of a 50-epoch schedule (registered rule, not changed).
