"""Cache multi-layer frame features (and token sequences for fine-tunable ViTs) of a frozen encoder. Resumable.

Per cycle (rows in `cycle_table` order) it stores, in `<cache_root>/<name>/`:
  layers.npy  (N, L, 32, D) float16  frames after the blocks in `keep_layers` (the last layer goes through the encoder's
                                     final LayerNorm); L = 1 for encoders that only expose their last layer
  tokens.npy  (N, T, D) float16      ViT encoders only: token sequence after block `token_layer`, padded time columns
                                     removed; the input of the blocks that `src.auscult_trust` may fine-tune
  keys.json   cache key per row; done.npy  bool per row (progress, written every 128 rows)
  meta.json   encoder name, `keep_layers`, `token_layer` (read back by `src.auscult_trust.data.FeatureData`)
Re-running the same command continues at the first row not marked done. Bump `name` whenever preprocessing changes,
so old features are never reused silently.

Config key `dataset: kauh` (with `kauh_root`) caches the 8 s windows of `data.kauh.windows` instead of ICBHI cycles.

Usage: python -m src.features configs/extract_ast.yaml [--limit N]   (N: smoke test on the first N rows)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src import encoders
from src.data.audio import cycle_wave, load_wav
from src.data.icbhi import cycle_table
from src.data.kauh import file_table as kauh_files
from src.data.kauh import windows as kauh_windows
from src.encoders.base import N_FRAMES


def cache_key(row):
    """Stable key of one cycle, from `stem`, `start`, `end`."""
    return hashlib.md5(f"{row.stem}|{row.start}|{row.end}".encode()).hexdigest()


def open_arrays(out, n, shape_l, shape_t):
    """Open (or create) the memmapped arrays and progress flags of a cache directory. `shape_t` None: no token array."""
    out.mkdir(parents=True, exist_ok=True)

    def memmap(f, shape):
        return np.lib.format.open_memmap(out / f, mode="r+" if (out / f).exists() else "w+", dtype=np.float16, shape=(n, *shape))
    layers = memmap("layers.npy", shape_l)
    tokens = memmap("tokens.npy", shape_t) if shape_t else None
    done = np.load(out / "done.npy") if (out / "done.npy").exists() else np.zeros(n, bool)
    return layers, tokens, done


def _flush(arrays, done, path):
    for a in arrays:
        a.flush()
    np.save(path, done)


def extract(cfg, limit=None):
    """Embed every row of the cycle table once and write the cache of `cfg`. Rows already marked done are skipped."""
    df = kauh_windows(kauh_files(cfg["kauh_root"])) if cfg.get("dataset") == "kauh" else cycle_table(cfg["icbhi_root"])
    keys = [cache_key(r) for r in df.itertuples()]
    out = Path(cfg["cache_root"]) / cfg["name"]
    if (out / "keys.json").exists():
        assert json.loads((out / "keys.json").read_text()) == keys, "cache rows do not match the cycle table"
    enc = encoders.load(cfg["encoder"])
    keep, tl = list(cfg.get("keep_layers", [enc.n_layers])), cfg.get("token_layer")
    layers, tokens, done = open_arrays(out, len(df), (len(keep), N_FRAMES, enc.dim), enc.token_shape)
    arrays = [a for a in (layers, tokens) if a is not None]
    (out / "keys.json").write_text(json.dumps(keys))
    (out / "meta.json").write_text(json.dumps({"encoder": cfg["encoder"], "keep_layers": keep, "token_layer": tl}))
    raw, n = {}, limit or len(df)
    for i, r in enumerate(df.itertuples()):
        if i >= n or done[i]:
            continue
        if r.wav not in raw:
            raw = {r.wav: load_wav(r.wav)}  # keep one recording in memory
        fr, tok = enc.run(cycle_wave(raw[r.wav], r.start, r.end), set(keep), tl)
        layers[i] = torch.stack([fr[k] for k in keep]).cpu().numpy().astype(np.float16)
        if tokens is not None:
            tokens[i] = tok.cpu().numpy().astype(np.float16)
        done[i] = True
        if i % 128 == 127:
            _flush(arrays, done, out / "done.npy")
            print(f"{done.sum()}/{len(df)} rows", flush=True)
    _flush(arrays, done, out / "done.npy")
    print(f"finished: {done.sum()}/{len(df)} rows done in {out}")


if __name__ == "__main__":
    extract(yaml.safe_load(Path(sys.argv[1]).read_text()), int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None)
