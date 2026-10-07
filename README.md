# auscult-trust

AuscultTrust: spectrum correction (SC) + an AST classifier trained with Patch-Mix CL + a split-conformal layer, for lung-sound
classification that has to survive a change of stethoscope (KHKT 2026-2027).

Question: does SC let a benchmark-level lung-sound classifier keep its predictions and its conformal coverage when the
stethoscope changes, without labels from the new stethoscope?

## Setup and run

```
bash scripts/download_data.sh            # ICBHI, KAUH, AST weights, reference repository (see the file header for sources)
python scripts/prepare_data.py           # fix ICBHI file names, build the training input folder, check the datasets
uv venv .venv && uv pip install -r requirements.txt                      # analysis and tests
uv venv .venv-train --python 3.10 && uv pip install -r requirements-train.txt   # training (needs timm==0.4.5)
bash scripts/run.sh screen               # then: summary, final, report (the file header explains each step)
uv run --no-project --python .venv/bin/python python -m pytest -q tests
```
More in `docs/SETUP.md` and `docs/QUICKSTART.md`. Every run gets its own `run_id` directory; nothing is overwritten.

## Layout

```
src/
  paths.py, runs.py        repo and data locations; run directories outputs|checkpoints/<exp>/<run_id>/<cell>/<unit>
  processing/              ICBHI and KAUH loaders, patient-disjoint splits, spectrum correction and gain augmentation
  training/patchmix_cl/    dataset, training loop, reports, export of frozen predictions; the model and loss come from
                           the reference repository (upstream.py)
  evaluation/              metrics, coverage under shift, conformal layer, reader of exported predictions
experiments/train/         YAML: base recipe + one file per arm
scripts/                   run.sh, download_data.sh, prepare_data.py
tests/  notebooks/  docs/  data/ (README only)  checkpoints/ outputs/ (gitignored)
```
Next to the repo (the wrapper directory, not versioned): `data/`, `repos/` (read-only reference repositories), `docs/`
(proposals), `archive/` (earlier experiments).
