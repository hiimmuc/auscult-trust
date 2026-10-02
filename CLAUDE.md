# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status
Scope now: Phase 0 + Tier 0 2x2 ablation (H1, H2) on ICBHI. Data dir `../data` is empty so far; nothing ran on real data. Proposal: `../docs/research-proposal-lung-sound-v4-en.md`. Reference repos (OPERA, Patch-Mix CL, SG-SCL) cloned in `../repos`.
TODO: OPERA/AST/CLAP encoder wrappers, HF_Lung phase detector + loader, KAUH loader, ICBHI patient split file.

## Commands
Env uses `uv`: `uv venv .venv && uv pip install -r requirements.txt`.
- Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests` (single: append `tests/test_core.py::test_icbhi_score`)
- One config: `.venv/bin/python -m src.train configs/tier0_icbhi_baseline.yaml`
- 2x2 ablation + shuffled-phase control: `.venv/bin/python -m src.ablate configs/tier0_icbhi_baseline.yaml` (phase cells skipped until phase masks are cached)

## Code map
Run from repo root, imports are `src.*`. Flow: `data/icbhi.py` cycle table -> `data/splits.py` patient-disjoint split file -> `models/cache.py` frozen-encoder embeddings in `../data/cache/<encoder>/` -> `models/net.py` (`LungModel` flags `use_branch`, `use_phase`, `shuffle_phase`) -> `train.py` / `ablate.py` -> `conformal/`, `tta/`, `eval/` -> `runlog.py` writes `experiments/<name>/seed<N>.json`. Code docstrings: Google style.

## Project
Lung-sound reliability study (KHKT 2026–2027). Full spec: `../docs/research-proposal-lung-sound-v4-en.md`.

Core questions:
- RQ1: Do breath phase and short time windows help, per class (crackle vs wheeze)?
- RQ2: Which class breaks under recording-device shift?
- RQ3: Does TTA break conformal coverage, and which correction restores it?
- RQ4: Does auscultation add information beyond vital signs? Report ΔAUROC with CI, not a win/lose claim.
- RQ5: Does abstention track clinician disagreement?

## Data tiers (hard rules)
| Tier | Data | Use |
|---|---|---|
| 0 | ICBHI, HF_Lung_V1, KAUH | Train, validate, calibrate, shift experiments |
| 1 | Lung phantom recordings | Evaluate only. No model selection. No conformal calibration for humans |
| 2 | Hospital patients | Test only. Locked. Opened once, per pre-registration |

- Never read, load, or inspect Tier 2 data unless the task explicitly says the pre-registered analysis is running.
- Splits are patient-disjoint. Always.
- KAUH: one recording exists in 3 filter renderings (Bell/Diaphragm/Extended). All renderings of a patient go in the same split.
- ICBHI: device is confounded with class prior and site. Device-held-out splits ignore the official split, hold out a whole device, keep patients disjoint.
- Never join datasets across different people (e.g., ICBHI audio + MIMIC vitals).

## Model
- Encoder: OPERA-CT, frozen. Extract embeddings once, cache them.
- Leakage control: OPERA was pretrained on ICBHI and HF_Lung. Every Tier 0 shift result must also be run with a non-OPERA encoder (AST AudioSet or CLAP).
- Trainable parts: short-window branch, phase-aware pooling, head. Nothing else.
- TTA updates only normalisation parameters of branch and head. After TTA, recompute nonconformity scores with the adapted model.
- Conformal variants: split, Mondrian, weighted (domain-classifier density ratio), label-shift, k-shot recalibration.
- Patient-level aggregation is fixed: max and mean of p(crackle), p(wheeze) per phase across sites. Not learned.

## Experiment rules
- ≥5 seeds. Report mean ± SD. A gain smaller than seed SD is not a result.
- Report per-class F1/recall, ICBHI Score, ECE, coverage, set size.
- Patient-level bootstrap for CIs. Never cycle-level.
- Log config, seed, data hash, and encoder for every run.
- Negative results are kept and reported.

## Repo layout (proposed)
```
data/        raw/ (read-only), cache/ (embeddings), splits/ (versioned split files)
src/         data/, models/, conformal/, tta/, eval/
configs/     one file per experiment
experiments/ outputs per run
docs/        proposal, pre-registration, phantom protocol
```

## Working conventions
- Simplest change that works. No speculative abstractions.
- Do not modify `data/raw/` or committed split files.
- Do not invent dataset statistics, citations, or results. Mark unknowns as TODO.
- Claims in docs use tags: [Fact], [Hypothesis], [Assumption].
- Code comments and docs in English.

## Open items
See proposal Appendix B (B1–B7). Recount ICBHI per-device patients and cycles from filenames before using any device counts.
