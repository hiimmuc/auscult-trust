# auscult-trust

Code for the lung-sound reliability study (KHKT 2026–2027). The full spec is in `../docs/research-proposal-lung-sound-v4-en.md`; the rules for working in this repo are in `CLAUDE.md`.

The study asks whether breath phase and short time windows help per sound class (RQ1), which class breaks under recording-device shift (RQ2), whether test-time adaptation breaks conformal coverage (RQ3), whether auscultation adds information beyond vital signs (RQ4), and whether abstention tracks clinician disagreement (RQ5).

## Current scope

Only Tier 0 (public data) work exists: the 2x2 ablation of short-window branch × phase pooling on ICBHI (RQ1, hypotheses H1 and H2). Conformal variants, TTA and the metrics are implemented and unit-tested, but no experiment using them has run on real data yet. Tier 1 (phantom) and Tier 2 (hospital patients) are not started, and Tier 2 data must not be touched before pre-registration.

First ablation run (5 seeds, one patient split of 88/19/19 patients, OPERA-CT, 20 epochs, no tuning), per-run json in `experiments/ablation_*`:
no cell beats the FM-only baseline by more than its seed SD on ICBHI Score or per-class recall/F1 (ICBHI Score 0.506 ± 0.017 for FM only, 0.515 ± 0.019 with the branch). Phase pooling scored lower (0.487 ± 0.023), and the shuffled-phase control (0.507 ± 0.032) is indistinguishable from the real mask. The baseline is well below published ICBHI Scores (about 0.62), so it does not yet meet the proposal's reproduce-within-1-point check. The phase detector is weak (HF_Lung test frame F1: none 0.85, inspiration 0.68, expiration 0.29), which limits what H2 can show.

## Setup

Uses `uv`. The GPU driver supports CUDA 12.8, so the default torch wheel does not work; `requirements.txt` has the install command for the cu128 build.

```bash
uv venv .venv
uv pip install -r requirements.txt
# then the torch/torchvision cu128 builds (see comments in requirements.txt):
uv pip install "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
```

OPERA reference code is expected at `../repos/OPERA` with the checkpoint `cks/model/encoder-operaCT.ckpt` (Hugging Face: `evelyn0414/OPERA`). Patch-Mix CL and SG-SCL baseline code is cloned next to it in `../repos` for reference only.

## Data

All data lives in `../data/` (outside this repo).

| Path | Content | Source |
|---|---|---|
| `raw/icbhi/ICBHI_final_database/` | 920 recordings, 6898 annotated cycles, 126 patients, plus `raw/icbhi/ICBHI_challenge_train_test.txt` | bhichallenge.med.auth.gr (official; its TLS certificate has expired, so it was fetched with certificate checks off and no checksum could be verified) |
| `raw/hf_lung_v1/{train,test}/` | 9765 15 s recordings with `*_label.txt` | gitlab.com/techsupportHF/HF_Lung_V1 |
| `raw/kauh/` | 336 recordings (112 subjects × Bell/Diaphragm/Extended), annotation xlsx | Mendeley Data, doi:10.17632/jwyy9np4gv.3 (sha256 verified) |
| `cache/opera_ct/` | frozen OPERA-CT frame embeddings per ICBHI cycle | `src.extract` |
| `cache/opera_ct_hflung/` | embeddings and phase labels for HF_Lung windows | `src.phase` |
| `splits/` | versioned, hashed patient-disjoint split files | `src.make_split` |

Rules (from `CLAUDE.md`): splits are always patient-disjoint; `raw/` and committed split files are never modified; datasets are never joined across different people.

## Pipeline

Run everything from the repo root.

```bash
# 1. patient-disjoint ICBHI split (refuses to overwrite an existing file)
.venv/bin/python -m src.make_split configs/tier0_icbhi_baseline.yaml

# 2. cache frozen OPERA-CT embeddings for every ICBHI cycle (~20 min on CPU)
.venv/bin/python -m src.extract configs/tier0_icbhi_baseline.yaml

# 3. train the phase detector on HF_Lung_V1, then write phase masks into the ICBHI cache
.venv/bin/python -m src.phase configs/phase.yaml

# 4. train one config, or run the full 2x2 ablation with a shuffled-phase control
.venv/bin/python -m src.train  configs/tier0_icbhi_baseline.yaml
.venv/bin/python -m src.ablate configs/tier0_icbhi_baseline.yaml

# tests
.venv/bin/python -m pytest -q tests
```

Every run writes `experiments/<name>/seed<N>.json` with its config, seed, split hash, encoder and metrics. The ablation runs 5 seeds and reports mean ± SD; a gain smaller than the baseline seed SD is not flagged as a result.

## Layout

```
src/data/       audio.py, icbhi.py, hf_lung.py, splits.py
src/models/     opera.py (frozen encoder), cache.py, net.py (branch, phase pooling, head, phase detector)
src/conformal/  split, Mondrian, weighted, label-shift and k-shot conformal sets
src/tta/        Tent-style adaptation of LayerNorm parameters only
src/eval/       metrics (per-class F1/recall, ICBHI Score, ECE, coverage, set size, patient bootstrap), patient aggregation
src/train.py    src/ablate.py    src/extract.py    src/phase.py    src/make_split.py
configs/        one yaml per experiment
experiments/    per-run outputs
tests/
```

## Known limits and TODO

- OPERA-CT frames are 0.25 s long (32 frames per 8 s cycle), so phase masks are coarse. The proposal's 64 ms window refers to OPERA's input window, not this frame rate.
- The phase detector is trained on a 1500-file subset of HF_Lung train (CPU budget). Its test accuracy is printed at the end of `src.phase`.
- The ICBHI device recount from filenames gives 583 LittC2SE train cycles against 594 in the proposal; the 11 extra cycles are not in the official split file. Patients per device (AKGC417L 32, Litt3200 11, LittC2SE 23, Meditron 64) sum to more than the 126 unique patients, so some patients appear on several devices.
- Not yet implemented: non-OPERA control encoder (AST AudioSet or CLAP), KAUH loader, device-held-out experiments, the calibration-aware TTA variant, and the Tier 1/Tier 2 analyses.
- The ICBHI zip passes its CRC check but has no published checksum to compare with.
