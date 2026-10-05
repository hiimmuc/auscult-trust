"""Archived branch/phase stack of proposal v4 (short-window branch, phase pooling, MLP head on cached frames, Tent on it).

Kept for the reported negative results in `report.md`; not extended and not part of the live pipeline
(`src.features`, `src.auscult_trust`). It reads the npz caches in `data/cache/<encoder>_v2` through `src.legacy.cache`.
Configs: `configs/legacy/`. Run as `python -m src.legacy.<module> configs/legacy/<config>.yaml`.
"""
