# Setup

Install the environments, download the data, check the data. Then go to [QUICKSTART.md](QUICKSTART.md).

## 1. Directory layout

`auscult-trust/` sits inside a wrapper directory. Data and reference code live next to the repo, not inside it.

```
KHKT-CVA-2026/                  wrapper (not versioned)
├── auscult-trust/              this repo
├── data/                       AT_DATA overrides            raw/ processed/ splits/ models/ cache/
├── repos/                      AT_REPOS overrides, read-only OPERA, patch-mix_contrastive_learning, SG-SCL, ...
├── archive/                    Phase 0 code (frozen)
└── docs/                       proposals
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `AT_DATA` | `../data` | Dataset and model-weight root (`src/paths.py`) |
| `AT_REPOS` | `../repos` | Reference repositories (read-only, imported by path) |

`DATA_DIR` and `REPOS_DIR` do the same job inside `scripts/download_data.sh`.

## 2. Requirements

- Linux, `uv`, `git`, `curl`, about 10 GB free disk for the core data (ICBHI zip is 1.98 GB).
- NVIDIA GPU for training. Reference machine: RTX 5000 Ada. The `torch` line in `requirements.txt` selects the CUDA 12.8 build.
- Python 3.10 for the training env (the reference AST code needs `timm==0.4.5`).

## 3. Environments

Two envs. Do not merge them.

```bash
# analysis, evaluation, tests
uv venv .venv
uv pip install -r requirements.txt

# training (Python 3.10, timm==0.4.5)
uv venv .venv-train --python 3.10
uv pip install --python .venv-train/bin/python -r requirements-train.txt
```

| Env | Used for |
| --- | --- |
| `.venv` | pytest, `src/evaluation`, notebooks |
| `.venv-train` | `src.training.patchmix_cl.main`, `export_probs`, `scripts/run.sh` (via `PY`) |

## 4. Data and reference code

```bash
bash scripts/download_data.sh            # ICBHI, KAUH, encoder weights, reference repositories
python scripts/prepare_data.py           # idempotent
```

`download_data.sh` skips complete files and resumes partial ones. `--hf-lung` also fetches HF_Lung_V1 (large, Phase 0 only).

| Item | Source |
| --- | --- |
| ICBHI 2017 | Official site (expired certificate: zip checked by md5), [Harvard Dataverse mirror](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HT6PKI), Kaggle |
| KAUH | [Mendeley Data jwyy9np4gv v3](https://data.mendeley.com/datasets/jwyy9np4gv/3) |
| AST AudioSet weights | AST authors |
| Patch-Mix CL, OPERA, SG-SCL | GitHub, cloned to `repos/` |
| Optional encoders (OPERA-CT, CLAP, HeAR, HTS-AT) | Hugging Face. HeAR is gated: request access, then `hf auth login`. HTS-AT weights are a manual download, see the header of `scripts/download_data.sh` |

`prepare_data.py` does:

1. Renames 91 ICBHI recordings that carry the wrong stethoscope name (`filename_differences.txt`) to `Meditron`.
2. Builds `data/processed/patchmix_icbhi/` with `audio_test_data/` (symlinks) and `official_split.txt` (official 60/40 split from the Patch-Mix CL repo).
3. Checks KAUH: 336 wav = 112 patients x 3 filters.

Expected after setup: ICBHI 920 wav, 6898 cycles, 126 patients; KAUH 336 wav.

## 5. Verify

```bash
uv run --no-project --python .venv/bin/python python -m pytest -q tests
```

Single test: append `tests/test_training.py::test_summary_counts_match_metrics`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `timm` import or version error in training | Use `.venv-train`, not `.venv`. Needs `timm==0.4.5`. |
| ICBHI download fails on certificate | Use the Dataverse mirror (script falls back automatically). |
| `official_split.txt` not found | Patch-Mix CL repo missing from `repos/`: rerun `download_data.sh`. |
| Missing package | Install it with `uv pip install`, do not work around it. |
| CPU-heavy extraction too slow or hogging | Prefix `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`. |
