#!/usr/bin/env bash
# Research units, API contracts, and TypeScript/build; no datasets, weights, or deployment.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv-train/bin/python}
"$PY" -m src.references --only PAFA ADD-RSC patch-mix_contrastive_learning QLung SG-SCL
"$PY" -m pytest -q
npm --prefix app/web ci
npm --prefix app/web run build
git diff --check
