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

Config key `a2: {source_devices: [...]}` applies A2 (per-mel-bin moment matching of the encoder's log-mel input) after A1: statistics are
estimated per device (the domain boundary) on all unlabelled clips, source devices pool to the source statistics, and every row is
mapped from its own device's statistics onto them. Needs `Encoder.logmel` (AST, OPERA-CT/CE); other encoders raise. Bump `name`.

Config key `correction: {source_devices: [...], reference: arithmetic|geometric, n_fft, hop}` applies A1 spectrum correction (src.shift.correction) to
every ICBHI waveform before encoding: a coefficient table is built once from all devices present in the cycle table,
referenced against `source_devices`, and each row is corrected with its own device's coefficient. Calibration rows
(source devices) and test rows (other devices) each get their own coefficient, so they are corrected toward the same
reference separately. ICBHI only (KAUH windows have no `device` column). Always bump `name` when adding or changing
`correction`.

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
from src.shift.correction import apply_spectrum_correction, isa_match, mean_spectrum, spectrum_coefficients


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


def _a2_stats(df, enc, source_devices, wave_of, max_frames=798):
    """Per-bin (mean, sd) of the encoder's log-mel input per device, and pooled over the source devices.

    Streaming sums (no clip kept in memory). `wave_of(row)` returns the waveform the encoder will see (A1 applied).
    """
    acc = {}
    for r in df.itertuples():
        x = enc.logmel(wave_of(r))[:max_frames].astype(np.float64)
        a = acc.setdefault(r.device, [0.0, 0.0, 0])
        a[0], a[1], a[2] = a[0] + x.sum(0), a[1] + (x ** 2).sum(0), a[2] + len(x)
    fin = lambda a: (a[0] / a[2], np.sqrt(np.maximum(a[1] / a[2] - (a[0] / a[2]) ** 2, 0)) + 1e-6)
    src = [acc[d] for d in source_devices]
    return {d: fin(a) for d, a in acc.items()}, fin([sum(a[i] for a in src) for i in range(3)])


def _flush(arrays, done, path):
    for a in arrays:
        a.flush()
    np.save(path, done)


def _device_coefficients(df, cfg):
    """Compute A1 coefficients per device in `df` from one mean spectrum per device (full cycle waveforms).

    Args:
        df: Cycle table with `wav`, `start`, `end`, `device` columns.
        cfg: Extraction config; reads `cfg["correction"]["source_devices"]`, `n_fft`, `hop`.

    Returns:
        Tuple (dict device -> coefficients array, n_fft, hop).
    """
    cc = cfg["correction"]
    n_fft, hop = cc.get("n_fft", 1024), cc.get("hop", 512)
    raw = {}
    waves_by_device = {}
    for r in df.itertuples():
        if r.wav not in raw:
            raw = {r.wav: load_wav(r.wav)}
        waves_by_device.setdefault(r.device, []).append(cycle_wave(raw[r.wav], r.start, r.end))
    spectra = {d: mean_spectrum(ws, n_fft, hop) for d, ws in waves_by_device.items()}
    return spectrum_coefficients(spectra, source_devices=cc.get("source_devices"), reference=cc.get("reference", "arithmetic")), n_fft, hop


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
    coef, n_fft, hop = (None, None, None)
    if cfg.get("correction") and cfg.get("dataset") != "kauh":
        coef, n_fft, hop = _device_coefficients(df, cfg)
    raw, n = {}, limit or len(df)

    def wave_of(r):
        nonlocal raw
        if r.wav not in raw:
            raw = {r.wav: load_wav(r.wav)}  # keep one recording in memory
        w = cycle_wave(raw[r.wav], r.start, r.end)
        return apply_spectrum_correction(w, coef[r.device], n_fft, hop) if coef is not None else w

    dev_stats = src_stats = None
    if cfg.get("a2") and cfg.get("dataset") != "kauh":
        dev_stats, src_stats = _a2_stats(df, enc, cfg["a2"]["source_devices"], wave_of)
    for i, r in enumerate(df.itertuples()):
        if i >= n or done[i]:
            continue
        wave = wave_of(r)
        if dev_stats is not None:  # A2 after A1: each device's log-mel statistics matched onto the pooled source statistics
            enc.a2 = lambda x, d=r.device: isa_match(x, dev_stats[d], src_stats)
        fr, tok = enc.run(wave, set(keep), tl)
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
