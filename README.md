# auscult-trust

Code for the lung-sound device-shift study (KHKT 2026–2027). Spec: `../docs/research-proposal-AB-gain-decomposition-v7-en.md`. Rules for working in this repo: `CLAUDE.md`. Published reference numbers: `../docs/papers/anchors.md`. Results of the archived v4 experiments: `report.md`.

Question: what fraction of stethoscope-induced shift in log-mel features is a per-frequency gain, and does correcting it restore conformal coverage?

## Pipeline

```
data/        icbhi.py (cycle table), kauh.py, splits.py (patient-disjoint split files), audio.py
encoders/    one interface for OPERA-CT, OPERA-CE, AST, CLAP, HeAR: wave -> (32, D) frames per cached layer
features.py  cache of layers (+ tokens for ViT encoders) per encoder: python -m src.features configs/extract_<enc>.yaml
auscult_trust/  ladder rungs on cached features: L0 probe, L1 head, L2 last-k blocks, L3 LoRA; ConformalLayer
conformal/ eval/ shift/   conformal V1-V5, metrics, shift metrics, gain corrections A1/A2
legacy/      archived v4 branch/phase stack (not extended)
```

Run the ladder for one encoder (resumable): `bash scripts/run_ladder.sh opera_ct|opera_ce|clap|ast|hear`.
Tests: `uv run --no-project --python .venv/bin/python python -m pytest -q tests`.
