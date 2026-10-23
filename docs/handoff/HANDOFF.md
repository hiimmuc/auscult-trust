# Handoff (23/10/2026 16:00): final model chosen, all runs done, results report due 25/10

For the next session. Read this file, then `CLAUDE.md` (repo rules), `report.md` v5 (results by plan stage),
`docs/decision_logs/2026-10-23-final-model.md` (final decision) and `docs/reports/final-pafa-static-execution.md`
(frozen evaluation). The proposal is v10 (`docs/updates/research-proposal-v10-en.md`); the registered study is v9 with
pre-registration amendments 1-5. Archived handoffs: `HANDOFF_2026-10-19.md` (LODO v1, SC, PC-MCL runs on five
machines, monitoring rules, incidents), `HANDOFF_2026-10-15-grid.md`, `HANDOFF_2026-10-14.md`, `HANDOFF_2026-10-08.md`.
Session style asked by the user: terse replies, minimal code, one runnable check per change; edits through a git
worktree, then into the main checkout; commit and push to `origin main` with a `Claude-Session:` line.

## 1. State

- [Fact] Final decision (user, 23/10): baseline = BEATs + PAFA, fine-tuned, no SC (`grid-v1/beats+ft+pafa`);
  AuscultTrust = the same recipe trained and evaluated with static SC (`lodo-v1/beats+ft+pafa+sc+sc_static`); fixed
  epoch 100, seeds 0-4, serving seed 0. Exploratory extensions E1 (BEATs PAFA, SG-SCL, Patch-Mix CL) and E1+E3 (plus
  HTS-AT Patch-Mix CL, SG-SCL), equal weights, with and without static SC.
- [Fact] Every training run is finished: grid-v1 350 units, lodo-v1 1,400 (220 no SC, 880 inference arms, 175 SC
  static, 125 SC dynamic + PAFA + PC-MCL with SC), pcmcl-v1 10, plus the registered AST study (lodo-ast 260,
  lodo-bench 95). 0 failed units. No process of ours runs on any machine.
- [Fact] Frozen evaluation `final-pafa-static` done 23/10 15:32: 122 commands passed, 0 failed. KAUH-A flip rate
  0.445 -> 0.209 (AuscultTrust), KAUH-B2 D = +0.113 [+0.072, +0.153] on 2 tested pairs: both positive controls pass.
  Screening view and B1 in the execution report. Seed-0 bundle `app/bundle-final-pafa-static/` validated.
- [Fact] H2 on ICBHI LODO is not testable (no fold above the null gate); the phantom is the remaining primary H2 test.

## 2. Where things are

- All outputs of machines A, B, D, E are on A in this repository (`outputs/train/<run>/<cell>/seed<s>/`; copied with
  `rsync --ignore-existing` on 23/10). Final evaluation: `outputs/train/final-pafa-static/`,
  `outputs/kauh/final-pafa-static/`, `outputs/screening/final-pafa-static/`; logs `logs/final-pafa-static*`.
- Checkpoints: A holds `grid-v1` (its half), the BEATs PAFA family of `lodo-v1` (120 units) and the ensemble members
  fetched for the final evaluation (SHA-256 checked, `logs/final-pafa-static/fetch_members.log`). Other checkpoints stay
  where they were trained (B: grid-v1 other half, lodo-ast, lodo-bench, lodo-v1; D, E: lodo-v1). A has about 100 GB free.
- Machines (ssh by key from A, repo at `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust`): B `namdp15@10.148.183.8`,
  D `namdp15@10.148.183.171`, E `namdp15@10.148.182.179`. Machine C (`10.148.183.119`) is dropped: never start anything
  there. B, D and E run older or newer code than A only for paths the finished runs did not use; sync with
  `rsync -a --exclude __pycache__ src experiments scripts tests <host>:<repo>/` before new runs.
- Demo app: `app/README.md` (Cloudflare Pages front end, GPU API through Tailscale Funnel).

## 3. Next steps (in order)

1. Results report for the submission of 25/10/2026, built from `report.md` v5 (sections 1, 6-16, 17).
2. Optional before the cut-off (31/01/2027): registered ensembles E1-E4 of Amendment 4; `k rate` with few target
   cycles; distillation of an ensemble into one BEATs model; AST/HTS-AT with SC dynamic (about 8 h on B, D, E, A) only if
   the user asks.
3. Phantom (H1, H3, co-primary H2): hardware, capture, remove-and-replace ceiling. Weights and s_ref are frozen
   (`freeze.json`) and must not change before the first recording.

## 4. Commands

- Status of runs on every machine: `bash scripts/grid_status.sh` (lists tags up to 8).
- Tables: `bash scripts/lodo_h2_all.sh`; `scripts/lodo_tables.py`, `scripts/ensemble.py`, `scripts/ensemble_sesp.py`,
  `scripts/ensemble_e1_dynamic.py` (prefix `PYTHONPATH=. .venv-train/bin/python`).
- Frozen evaluation: `scripts/final_evaluation.py`, `scripts/final_ensembles.py` (see `docs/QUICKSTART.md` section 7).
- Tests: `.venv/bin/python -m pytest -q tests` (model-building tests with `.venv-train`, see `docs/SETUP.md`).

## 5. Gotchas

- Kill processes by PID only, never `pkill -f` (over ssh the pattern matches the ssh shell).
- A cut-off unit always reruns from epoch 0 (move partial outputs and checkpoints to `<run>-interrupted`); never resume
  from `last.pth` for reported units.
- Another agent (main checkout, not a Claude session) also edits the docs; check `git status` and file times before
  editing shared files.
- From a git worktree, set `AT_DATA` and `AT_REPOS` to the wrapper's `data/` and `repos/`, and link `.venv`,
  `.venv-train`, `outputs` from the main checkout.
- `--epochs 1` smoke runs compress the cosine schedule: their scores mean nothing.
