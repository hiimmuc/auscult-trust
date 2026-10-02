"""Embedding cache. Encoder runs once per cycle; results live in `data/cache/<encoder>/`."""
import hashlib
from pathlib import Path

import numpy as np

from src.data.audio import bandpass, cycle_wave, load_wav


def cache_key(row):
    """Stable file name for one cycle.

    Args:
        row: Cycle table row with `stem`, `start`, `end`.

    Returns:
        String key.
    """
    return hashlib.md5(f"{row.stem}|{row.start}|{row.end}".encode()).hexdigest()


def extract(df, encoder, name, cache_root="../data/cache", phase_fn=None):
    """Embed every cycle once and save `{emb, wave}` as npz. Existing files are skipped.

    Args:
        df: Cycle table (see `src.data.icbhi.cycle_table`).
        encoder: Callable (wave 1-D float32 at 16 kHz) -> (N, D) frame embeddings.
            TODO: OPERA-CT, AST AudioSet and CLAP wrappers (weights/API not yet checked).
            Reference code: `../repos/OPERA`.
        phase_fn: Optional callable (N, D) embeddings -> (N,) phase mask in {0, 1, 2}.
            TODO: detector trained on HF_Lung_V1. Without it no phase is cached.
        name: Encoder name, used as cache sub-directory.
        cache_root: Cache directory root.

    Returns:
        Cache directory path.
    """
    out = Path(cache_root) / name
    out.mkdir(parents=True, exist_ok=True)
    recs = {}
    for r in df.itertuples():
        f = out / f"{cache_key(r)}.npz"
        if f.exists():
            continue
        if r.wav not in recs:
            recs = {r.wav: bandpass(load_wav(r.wav))}  # keep one recording in memory
        w = cycle_wave(recs[r.wav], r.start, r.end)
        emb = encoder(w)
        extra = {"phase": phase_fn(emb).astype(np.int8)} if phase_fn else {}
        np.savez(f, emb=emb.astype(np.float16), wave=w.astype(np.float16), **extra)
    return out


def load_cycle(cache_dir, row):
    """Read one cached cycle.

    Args:
        cache_dir: Directory returned by `extract`.
        row: Cycle table row.

    Returns:
        Tuple (emb (N, D) float32, wave (T,) float32, phase (N,) int64).
        Phase is all zeros when none was cached.
    """
    d = np.load(Path(cache_dir) / f"{cache_key(row)}.npz")
    emb = d["emb"].astype(np.float32)
    phase = d["phase"].astype(np.int64) if "phase" in d else np.zeros(len(emb), np.int64)
    return emb, d["wave"].astype(np.float32), phase
