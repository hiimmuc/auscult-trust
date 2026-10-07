# Quick start: train and compare the classifier arms

Prerequisite: [SETUP.md](SETUP.md) done (envs, data, tests pass). Run everything from the repo root.

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

```bash
PYTHONPATH=. .venv-train/bin/python -m src.training.patchmix_cl.export_probs \
  outputs/train/$RUN_ID/<cell>/seed0 --ckpt report_epoch_<E>.pth
```

Writes softmax, embeddings and `export/model.sha256`. Hand-off to Stage 2: frozen checkpoint + SHA-256, saved `s_ref` and coefficients, exported softmax. Conformal code reads the export through `src/evaluation/export.py`.

## 5. Where things go

```
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
