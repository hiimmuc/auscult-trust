# Quick start: train and compare the classifier arms

Prerequisite: [SETUP.md](SETUP.md) done (envs, data, tests pass). Run everything from the repo root.

Sections 1-6 describe the registered AST workflow (`scripts/run.sh`). Section 7 reproduces the final model chosen on
23/10/2026: baseline = BEATs + PAFA without SC, AuscultTrust = the same recipe with trained static SC.

## 1. Campaign

One campaign = one `RUN_ID`. Export the same id again to resume: finished units are skipped, interrupted ones continue from `last.pth`.

```bash
export RUN_ID=$(date +%Y%m%d-%H%M%S)
bash scripts/run.sh screen      # 3-fold grouped CV of baseline, sc, sc_gain, freq_mixstyle (SC arms: both sc_mode)
bash scripts/run.sh summary     # ranking, CV epoch + sc_mode per arm, AuscultTrust arm -> outputs/train/$RUN_ID/screen.json
bash scripts/run.sh final       # refit at CV epoch on all training patients, test once
bash scripts/run.sh report      # outputs/train/$RUN_ID/final.md
```

| Step | What happens | Output |
| --- | --- | --- |
| `screen [arm ...]` | CV on training patients only. Test set untouched. | per-fold reports |
| `summary` | AuscultTrust = `sc` or `sc_gain` by mean CV Score (tie: `sc`). `sc_mode` kept by higher CV Score (tie: `dynamic`). | `screen.json` |
| `final` | Seeds 0-9 for `baseline` and AuscultTrust, 0-4 for the others. | per-seed reports, checkpoints |
| `report` | Table, per-device Sp/Se/Score/HS, paired differences, non-inferiority check (gate G). | `final.md` |
| `reproduce [cell ...]` | Published protocol: best test epoch, 5 seeds. Optimistic, literature column only. | per-seed reports |

Switches (environment variables):

| Variable | Default | Effect |
| --- | --- | --- |
| `RUN_ID` | timestamp | Campaign id |
| `PARALLEL=N` | off | N jobs at once on the GPU (`scripts/sweep.py`) |
| `MEM_GB` | 8 | GPU memory a job needs before it starts |
| `FOLDS` | 3 | CV folds |
| `PY` | `.venv-train/bin/python` | Training interpreter |
| `EXTRA_ARGS` | empty | Appended to every training call |

## 2. Arms

| Arm | Config | Cell name |
| --- | --- | --- |
| P0 Patch-Mix CL | `baseline.yaml` | `baseline` |
| P1 SC | `sc.yaml` | `sc` |
| P1+P3 SC + random gain | `sc_gain.yaml` | `sc_gain` |
| P4 Freq-MixStyle | `freq_mixstyle.yaml` | `freq_mixstyle` |
| override | `sc_static.yaml` | `sc+sc_static` |

Cell name = config stems joined by `+`, without `base`. Details: `experiments/train/README.md`.

## 3. One run

```bash
# CV fold
PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.main \
  --config experiments/train/base.yaml,experiments/train/sc.yaml --cv_folds 3 --cv_fold 0 --seed 0

# final refit at a fixed epoch
PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.main \
  --config experiments/train/base.yaml,experiments/train/sc.yaml --selection fixed --report_epochs E --seed 0
```

## 4. Freeze a model for the conformal evaluation

```text
PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.export_probs \
  outputs/train/$RUN_ID/<cell>/seed0 --ckpt report_epoch_<E>.pth
```

Writes softmax, embeddings and `export/model.sha256`. Hand-off to Stage 2: frozen checkpoint + SHA-256, saved `s_ref` and coefficients, exported softmax. Conformal code reads the export through `src/evaluation/export.py`.

## 5. Where things go

```text
outputs/<exp>/<run_id>/<cell>/<unit>/       report, predictions, sc_state.npz, train_args.json
checkpoints/<exp>/<run_id>/<cell>/<unit>/   weights, last.pth
```

`<exp>` = `train`; `<unit>` = `cv<k>` or `seed<s>`. A finished unit is skipped. A different config in an existing unit is refused. Never write elsewhere.

Every SC run saves `sc_state.npz` (reference spectrum, per-device spectra, coefficients, `sc_mode`) and `s_ref_sha256`.

## 6. Common problems

| Symptom | Fix |
| --- | --- |
| Interrupted campaign | `export RUN_ID=<same id>`, rerun the same command. |
| GPU out of memory with `PARALLEL` | Raise `MEM_GB` or lower `PARALLEL`. |
| "different config" refusal | New `RUN_ID`, or new cell. Do not reuse a unit. |
| Unit shows `FAILED: <cell>` | Run that unit alone (section 3) to read the traceback. |

## 7. Final model: BEATs + PAFA, baseline vs AuscultTrust (static SC)

Configs compose left to right; the cell name is the stems joined by `+`. Fixed epoch 100, seeds 0-4.

```bash
C=experiments/train
# baseline (official split): cell beats+ft+pafa
PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.main \
  --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml \
  --cell beats+ft+pafa --selection fixed --report_epochs 100 --seed 0
# AuscultTrust (official split): cell beats+ft+pafa+sc+sc_static
PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.main \
  --config $C/base.yaml,$C/encoder_beats.yaml,$C/grid/pafa_beats.yaml,$C/sc.yaml,$C/sc_static.yaml \
  --cell beats+ft+pafa+sc+sc_static --selection fixed --report_epochs 100 --seed 0
# leave-one-device-out: append $C/lodo_<device>.yaml and +lodo_<device> to the cell (meditron, littc2se, litt3200, akgc417l)
```

Many units at once: `scripts/grid.py lodo-jobs` writes job lists, `scripts/sweep.py` runs one list per machine (budget by
`MEM_GB`), `bash scripts/grid_status.sh` checks every machine. A unit cut off by a crash reruns from epoch 0: move its
partial output and checkpoint folders to `<run>-interrupted` first.

Analysis and the frozen evaluation (CPU, from the stored predictions):

```bash
bash scripts/lodo_h2_all.sh                                          # null gate and H2 per cell
PYTHONPATH=. .venv-train/bin/python scripts/lodo_tables.py           # LODO, inference-arm and trained-SC tables
PYTHONPATH=. .venv-train/bin/python scripts/ensemble.py              # E1, E2, E3 and unions, equal vs fitted weights
PYTHONPATH=. .venv-train/bin/python scripts/ensemble_sesp.py         # Se at PAFA's Sp (k rate), best epoch on test
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. .venv-train/bin/python -u scripts/final_evaluation.py  # freeze, KAUH A/B2/B1, screening, bundle
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. .venv-train/bin/python -u scripts/final_ensembles.py   # E1 and E1+E3 on KAUH and screening
```

Results: `report.md`; frozen evaluation: `docs/reports/final-pafa-static-execution.md`; decision:
`docs/decision_logs/2026-10-23-final-model.md`.
