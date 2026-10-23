# Repository Guidelines

## Project Structure & Module Organization

`src/processing/` handles audio, datasets, patient splits, and spectrum correction; `src/training/patchmix_cl/` implements training/export; `src/evaluation/` contains metrics and conformal analysis. Training recipes live in `experiments/train/`, automation in `scripts/`, and Python tests in `tests/`. `api/` serves FastAPI inference; `app/web/` contains the Vite/TypeScript UI and browser checks. Consult `docs/SETUP.md`, `docs/QUICKSTART.md`, `report.md`, `docs/handoff/HANDOFF.md` and `app/README.md` for setup, reproduction commands, results, current state, and serving details. The current proposal is `docs/updates/research-proposal-v10-en.md`; the registered study is v9.

The parent directory is an unversioned wrapper: `../data/` holds datasets, `../repos/` holds read-only upstream references, and `../archive/` contains frozen work. Keep generated results in ignored `outputs/`, `checkpoints/`, and `logs/`.

## Build, Test, and Development Commands

Run from this repository unless specified:

- `uv pip install --python .venv/bin/python -r requirements.txt`: install analysis/test dependencies after creating `.venv`.
- `uv pip install --python .venv-train/bin/python -r requirements-train.txt`: install training dependencies into a separate Python 3.10 environment.
- `.venv/bin/python -m pytest -q tests`: run the research test suite.
- `bash scripts/run.sh screen`: start CV screening; subsequent campaign stages are `summary`, `final`, and `report`. Preserve `RUN_ID` when resuming.
- `PYTHONPATH=. .venv-train/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000`: serve a prepared model bundle; install `api/requirements.txt` first.
- `cd app/web && npm ci && npm run dev`: start the UI; `npm run build` type-checks and builds it.

## Coding Style & Naming Conventions

Follow nearby code: four-space Python indentation, snake_case functions/modules, and two-space TypeScript indentation with camelCase identifiers. Write comments and documentation in English. Prefer small changes and existing helpers. No repository-wide formatter or lint command is configured; the frontend build checks TypeScript.

## Testing Guidelines

Use pytest files named `test_*.py` and functions named `test_*`. Add regression coverage for changed numerical behavior and patient-split logic. API contract checks run with `.venv-train/bin/python -m pytest -q api/test_contract.py`; real-model tests require a bundle and GPU. Browser checks are documented in `app/README.md`. No numerical coverage threshold is configured.

## Commit & Pull Request Guidelines

History mixes descriptive subjects with prefixes such as `docs:` and `api:`. Use a concise, scoped subject. PRs should explain behavior changes, validation commands/results, and relevant issues or protocol decisions; include screenshots for UI changes.

## Research & Agent Rules

Preserve raw data and committed splits. Tune only on training-patient CV; retain negative results and log seeds, configuration, and artifact hashes. Never commit weights or generated results. Read `CLAUDE.md` for research conventions. When a CodeGraph index is available, query it before locating or reading code; never index automatically.
