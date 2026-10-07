# auscult-trust

AuscultTrust: spectrum correction (SC) + AST (Patch-Mix CL recipe) + split-conformal layer for lung-sound classification
under stethoscope shift (KHKT 2026-2027). Main line: proposal v9, `docs/research-proposal-v9-en.md`; pre-registration
`docs/prereg-v8.md` + `docs/prereg-v9-amendment.md`. Working rules: `CLAUDE.md`. Setup: `docs/SETUP.md`, run: `docs/QUICKSTART.md`.

Question: does SC let a benchmark-level lung-sound classifier keep its predictions and its conformal coverage when the
stethoscope changes, without labels from the new stethoscope?

## Layout

```text
src/
  paths.py, runs.py        repo/data locations; run directories (outputs|checkpoints/<exp>/<run_id>/<cell>/<unit>)
  processing/              ICBHI/KAUH loaders, patient-disjoint splits, spectrum correction (SC, A2, random bin gain)
  training/patchmix_cl/    Stage 1: config, dataset, model, engine, main, export_probs, summary
  evaluation/              metrics, coverage/shift metrics, conformal layer, Stage 2 export reader
experiments/stage1/        YAML: base recipe + one file per arm
scripts/                   run_stage1.sh, migrate_legacy_layout.py
tests/  notebooks/  docs/  data/ (README only)  checkpoints/ (gitignored)  outputs/ (gitignored)
```

Next to the repo (wrapper directory, not versioned): `data/`, `repos/` (read-only reference repos), `docs/` (proposals), `archive/`
(Phase 0 code, see `archive/README.md`).

## Quick commands

```bash
uv venv .venv && uv pip install -r requirements.txt             # analysis, tests, Stage 2
uv run --no-project --python .venv/bin/python python -m pytest -q tests
bash scripts/run_stage1.sh screen        # Stage 1 screening, then: summary, final, report
```

Every run is a new `run_id` directory; nothing is overwritten (`src/runs.py`).
