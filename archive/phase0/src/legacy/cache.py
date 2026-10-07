"""Reader of the archived per-cycle npz cache of the branch/phase experiments (`data/cache/<encoder>/<key>.npz`).

These caches were written by an earlier extractor (frame embeddings plus the band-passed cycle for the short-window
branch, optional phase mask) and are kept as read-only artefacts. The live pipeline is `src.features`.
"""
from pathlib import Path

import numpy as np

from src.features import cache_key


def load_cycle(cache_dir, row):
    """Read one cached cycle.

    Args:
        cache_dir: Cache directory of one encoder.
        row: Cycle table row.

    Returns:
        Tuple (emb (N, D) float32, wave (T,) float32, phase (N,) int64).
        Phase is all zeros when none was cached.
    """
    d = np.load(Path(cache_dir) / f"{cache_key(row)}.npz")
    emb = d["emb"].astype(np.float32)
    phase = d["phase"].astype(np.int64) if "phase" in d else np.zeros(len(emb), np.int64)
    return emb, d["wave"].astype(np.float32), phase
