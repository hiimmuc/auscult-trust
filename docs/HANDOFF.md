# Handoff: state on 2026-10-14 and what runs

Read this first when resuming. The plan being executed is the 2026-10-14 implementation plan (two machines); its registered parts are `docs/prereg-v9-amendment.md` (correction note at the end) and `docs/prereg-v9-amendment-2.md`. The previous handoff (2026-10-08) is `docs/HANDOFF_2026-10-08.md`.

## Machines (repo path `~/Workspace/Personal/KHKT-CVA-2026/auscult-trust` on both)

- **A** (local, RTX 5000 Ada 32 GB): code, tests, inference, analysis, light training.
- **B** (`namdp15@10.148.183.8`, RTX 5090 32 GB): heavy training. SSH key login works. B has its own `.git` copy (not synced); code goes A -> B with
  `rsync -a --exclude __pycache__ src experiments scripts tests CODE_COMMIT namdp15@10.148.183.8:Workspace/Personal/KHKT-CVA-2026/auscult-trust/`
  (`git rev-parse HEAD > CODE_COMMIT` first). Results B -> A: `rsync -a --ignore-existing` of `outputs/train/<RUN_ID>/` (and checkpoints when needed).
- B's `.venv-train` got pip (`ensurepip`) and pytest on 14/10. Reference repos added: `repos/PAFA` (BEATs module, PAFA loss) on both machines; BEATs weights `data/models/beats/`; ESC-50 in `data/raw/esc50/` (A only).

## Fixed on 14/10

- `db_scale` of AST/DASS was 4.34x too large (Kaldi fbank is a natural log): the "6 dB" gain of `sc_gain` was ~26 dB. Decision log `docs/decision_logs/2026-10-14-db-scale.md`. Old `sc_gain` numbers are a labelled deviation; rerun as campaign C0.

## What runs (schedulers survive the session; restart any of them with the same command, finished units are skipped)

| Where | Process | Log | Content |
|---|---|---|---|
| B | `logs/c0_chain.sh` | `logs/c0_chain.out`, `logs/screen-ast-c0-screen.log`, `logs/screen-ast-c0-refit.log` | C0: `sc_gain` 3-fold screen (RUN_ID `screen-ast-c0`, both bounds) -> summary -> refit `sc_gain_v2[+sc_static]` seeds 0-4 in `screen-ast` |
| B | sweep `logs/lodo-ast/jobs-e14.txt` | `logs/lodo-ast-e14.log` | C1b: `baseline_e14+lodo_<dev>` seeds 0-4 |
| B | `scripts/queue.sh wait:c0_chain.sh wait:jobs-e14.txt smoke c4_screen c4_gate c4_final c2_ast c2_dass c3 c5 c6` | `logs/queue-b.log` | smoke runs, then C4 (`sc_cons` vs `sc_gain_v2`, 5-fold, stop rule in `scripts/c4_gate.py`), C2 AST/DASS LODO epoch 50 (`lodo-bench`), C3 BEATs, C5 batch 32, C6 PAFA |
| A | sweep `logs/screen-ast/jobs-c1a.txt` | `logs/screen-ast-c1a.log` | C1a: `baseline_e14` seeds 0-9 (official split) |
| A | `scripts/run_lodo_infer.sh` | `logs/lodo-infer.log` | LODO inference arms (`baseline_plain`, `baseline_ttsc` primary, `baseline_ttsc_all`, `baseline_a2`, `baseline_ttsc_a2`, `baseline_tta8`, `sc_plain`, `sc_a2`, `sc_tta8`) and the F4 target-sample sweep |
| A | `scripts/queue.sh wait:jobs-c1a.txt kauh:p0_e14... c2_htsat c2f` | `logs/queue-a.log` | KAUH for the epoch control, C2 HTS-AT, C2f frozen AST / OPERA-CT |
| A | `scripts/queue.sh wait:jobs-c1a.txt screen:... export:... kauh:...` | `logs/queue-a2.log` | screening view ICBHI windows, A5 inductive export, KAUH comparators (A2, TTA-EQ) |

Job lists: `bash scripts/campaigns.sh <name>` writes `logs/campaigns/<name>.txt` (each line carries its RUN_ID). Kill schedulers by PID only.

## Analyses to run when the training is in (all on A, after syncing outputs from B)

- C0: `outputs/train/screen-ast-c0/screen.json`; refit table: `summary final` reads only the registered arms, so compare `sc_gain_v2*` directly (e.g. `summary gate --base_cell baseline --base_epoch 2 --arm_cell <sc_gain_v2 cell> --arm_epoch <E>`).
- Gate G sensitivity: `python -m src.training.patchmix_cl.summary gate --run screen-ast` (AuscultTrust vs `baseline_e14`, ten seeds).
- LODO: `python -m src.evaluation.lodo --run lodo-ast --base baseline --arms sc+sc_static freq_mixstyle baseline_ttsc baseline_ttsc_all baseline_a2 baseline_tta8 sc_tta8 sc_a2 baseline_e14 --out lodo_summary_v2`; epoch-matched: `--base baseline_e14 --arms sc+sc_static --out lodo_summary_e14`. Benchmark: `--run lodo-bench --base ast_baseline --arms htsat_baseline dass_baseline beats_baseline ast_frozen opera_ct_frozen pafa_ast`.
- Probe: `python -m src.evaluation.probe --run lodo-ast --arms baseline_plain sc_plain`.
- F4: `outputs/train/lodo-ast/target_n/sc+sc_static+lodo_<dev>/seed<s>.json`.
- KAUH: `python -m src.evaluation.kauh --dir outputs/kauh/final --base p0 --arms p0_sc auscult p0_e14 p0_e14_sc p0_a2 p0_sc_a2 p0_tta8 auscult_tta8 auscult_a2`.
- Screening / B1: `python -m src.evaluation.screening analyse --pairs p0:p0 p0:p0_sc auscult:auscult p0_e14:p0_e14 p0_e14:p0_e14_sc`.
- Noise stress (S1): `python -m src.evaluation.snr_stress infer outputs/train/screen-ast/<cell>/seed<s> --out outputs/snr/<arm>` then `analyse --out outputs/snr --arms baseline sc sc_gain_v2`.
- SC diagnostics: `scripts/sc_diagnostics.py` (done: `outputs/sc_diag/summary.json`, sigma_N10 = 2.23 dB).
- Report: `docs/report_2026-11-15.md`.
