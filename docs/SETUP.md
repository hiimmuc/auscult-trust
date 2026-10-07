# Setup

The repo `auscult-trust/` sits inside a wrapper directory with `data/` (or the folder in `AT_DATA`), `repos/` (or `AT_REPOS`) and `archive/`.

1. Data and reference code: `bash scripts/download_data.sh`, then `python scripts/prepare_data.py`.
   Sources: ICBHI from the official site (https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip),
   the Harvard Dataverse mirror (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HT6PKI) or Kaggle; KAUH from Mendeley Data
   (https://data.mendeley.com/datasets/jwyy9np4gv/3); AST weights from the AST authors; the Patch-Mix CL repository from GitHub.
2. Analysis env (tests, evaluation): `uv venv .venv && uv pip install -r requirements.txt` (the torch line in that file selects the CUDA 12.8 build).
3. Training env: the reference AST code needs `timm==0.4.5`, so a second env on Python 3.10:
   `uv venv .venv-train --python 3.10 && uv pip install --python .venv-train/bin/python -r requirements-train.txt`.

Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests`.
