"""Train the breath-phase detector on frozen OPERA-CT frame embeddings of HF_Lung_V1,
then write predicted phase masks into the ICBHI embedding cache.

Usage: python -m src.phase configs/phase.yaml
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from src.data.audio import SR, bandpass, crop_pad, load_wav
from src.data.hf_lung import file_table, frame_phase, read_labels
from src.data.icbhi import cycle_table
from src.data.splits import split_patients
from src.models.cache import cache_key
from src.models.net import PhaseDetector
from src.models.opera import OperaCT

WINDOWS = (0.0, 7.0)  # two 8 s windows cover a 15 s recording


def embed_hf(rows, encoder, out):
    """Embed HF_Lung windows once and cache `{emb, phase}`. Existing files are skipped.

    Args:
        rows: Output of `file_table`.
        encoder: Callable wave -> (N, D) embeddings.
        out: Cache directory.

    Returns:
        List of npz paths, aligned with `rows` x `WINDOWS`.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in rows:
        stem = Path(r["wav"]).stem
        x = None
        for w in WINDOWS:
            f = out / f"{stem}_{int(w)}.npz"
            paths.append(f)
            if f.exists():
                continue
            x = bandpass(load_wav(r["wav"])) if x is None else x
            emb = encoder(crop_pad(x[int(w * SR):], 8 * SR))
            np.savez(f, emb=emb.astype(np.float16),
                     phase=frame_phase(read_labels(r["label"]), w, n_frames=len(emb)).astype(np.int8))
    return paths


def load(paths):
    """Stack cached HF_Lung windows.

    Args:
        paths: npz paths from `embed_hf`.

    Returns:
        Tuple (emb (n, N, D) float32, phase (n, N) int64).
    """
    d = [np.load(p) for p in paths]
    return (np.stack([x["emb"] for x in d]).astype(np.float32),
            np.stack([x["phase"] for x in d]).astype(np.int64))


def train_detector(emb, phase, epochs=30, lr=1e-3, seed=0):
    """Fit a `PhaseDetector` on windows of frame embeddings.

    Args:
        emb: (n, N, D) embeddings.
        phase: (n, N) labels in {0, 1, 2}.
        epochs: Passes over the windows.
        lr: AdamW learning rate.
        seed: RNG seed.

    Returns:
        Trained `PhaseDetector` in eval mode.
    """
    torch.manual_seed(seed)
    x, y = torch.tensor(emb), torch.tensor(phase)
    m = PhaseDetector(x.shape[-1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(epochs):
        for i in torch.randperm(len(x)).split(64):
            opt.zero_grad()
            F.cross_entropy(m(x[i]).flatten(0, 1), y[i].flatten()).backward()
            opt.step()
    return m.eval()


@torch.no_grad()
def predict_phase(det, emb):
    """Predict per-frame phase.

    Args:
        det: Trained `PhaseDetector`.
        emb: (N, D) or (n, N, D) float32 embeddings.

    Returns:
        Integer array of shape emb.shape[:-1].
    """
    return det(torch.tensor(emb)).argmax(-1).numpy()


def add_phase_to_cache(df, cache_dir, det):
    """Write predicted phase masks into existing ICBHI cache files.

    Args:
        df: ICBHI cycle table.
        cache_dir: Embedding cache directory.
        det: Trained `PhaseDetector`.
    """
    for r in df.itertuples():
        f = Path(cache_dir) / f"{cache_key(r)}.npz"
        d = dict(np.load(f))
        d["phase"] = predict_phase(det, d["emb"].astype(np.float32)).astype(np.int8)
        np.savez(f, **d)


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1]))
    enc = OperaCT()
    rows = file_table(cfg["hf_root"])
    rng = np.random.default_rng(0)
    train = [r for r in rows if r["official"] == "train"]
    test = [r for r in rows if r["official"] == "test"]
    train = [train[i] for i in rng.permutation(len(train))[:cfg["n_train"]]]
    test = [test[i] for i in rng.permutation(len(test))[:cfg["n_test"]]]
    cache = Path(cfg["cache_root"]) / "opera_ct_hflung"
    xtr, ytr = load(embed_hf(train, enc, cache))
    xte, yte = load(embed_hf(test, enc, cache))
    det = train_detector(xtr, ytr)
    pred = predict_phase(det, xte)
    print("HF_Lung test frame acc %.3f, F1 none/insp/exp:" % (pred == yte).mean(),
          f1_score(yte.ravel(), pred.ravel(), average=None).round(3))
    torch.save(det.state_dict(), Path(cfg["cache_root"]) / "phase_detector.pt")
    if cfg.get("apply_to_icbhi"):
        add_phase_to_cache(cycle_table(cfg["icbhi_root"]), Path(cfg["cache_root"]) / "opera_ct", det)
