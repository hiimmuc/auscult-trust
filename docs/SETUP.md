# Setup

Directory layout: the repo `auscult-trust/` sits inside a wrapper directory with `data/` (or `$AT_DATA`), `repos/`, `archive/`.

1. Analysis env (tests, Stage 2): `uv venv .venv && uv pip install -r requirements.txt` (torch cu128 line in that file).
2. Training env (Stage 1): AST needs `timm==0.4.5`, so a second env on Python 3.10:
   `uv venv .venv-train --python 3.10 && uv pip install --python .venv-train/bin/python -r requirements-train.txt`.
3. Data under `../data` (`data/README.md`): `raw/icbhi`, `raw/kauh`, `processed/patchmix_icbhi` (`audio_test_data/` symlinks to the
   ICBHI wav and txt files, `official_split.txt`), `models/ast/audioset_10_10_0.4593.pth` (AST AudioSet weights, mAP 0.4593).
4. Legacy artefacts from before the restructure: `python scripts/migrate_legacy_layout.py` (dry run), then `--apply`.

Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests`.
