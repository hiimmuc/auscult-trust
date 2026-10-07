"""Train the breath-phase detector on frozen-encoder frame embeddings of HF_Lung_V1,
then write predicted phase masks into the ICBHI embedding cache.

HF_Lung_V1 labels expiration in only about half of the breaths (per 15 s file: 3.5 inspiration vs
1.9 expiration segments, counted on every 10th file), so unlabelled expiration sits in class
"none" and expiration F1 is low. ICBHI masks therefore use p(inspiration) only: each annotated
cycle gets one inspiration and one expiration segment (`decode_cycle`).

Usage: python -m src.legacy.phase configs/legacy/phase.yaml
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from src import encoders
from src.data.audio import CYCLE_SEC, SR, crop_pad, load_wav
from src.legacy.hf_lung import file_table, frame_phase, read_labels
from src.data.icbhi import cycle_table
from src.features import cache_key
from src.legacy.net import PhaseDetector, device
from src.runlog import run_dir, start_summary, write_meta

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
            x = load_wav(r["wav"]) if x is None else x  # raw input, as for ICBHI
            emb = encoder(crop_pad(x[int(w * SR) :], 8 * SR))
            np.savez(
                f,
                emb=emb.astype(np.float16),
                phase=frame_phase(read_labels(r["label"]), w, n_frames=len(emb)).astype(np.int8),
            )
    return paths


def load(paths):
    """Stack cached HF_Lung windows.

    Args:
        paths: npz paths from `embed_hf`.

    Returns:
        Tuple (emb (n, N, D) float32, phase (n, N) int64).
    """
    d = [np.load(p) for p in paths]
    return (
        np.stack([x["emb"] for x in d]).astype(np.float32),
        np.stack([x["phase"] for x in d]).astype(np.int64),
    )


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
    x, y = torch.tensor(emb, device=device), torch.tensor(phase, device=device)
    m = PhaseDetector(x.shape[-1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(epochs):
        for i in torch.randperm(len(x), device=device).split(64):
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
    return det(torch.tensor(emb, device=device)).argmax(-1).cpu().numpy()


@torch.no_grad()
def insp_prob(det, emb, bs=1024):
    """p(inspiration) per frame.

    Args:
        det: Trained `PhaseDetector`.
        emb: (n, N, D) float32 embeddings.
        bs: Batch size.

    Returns:
        (n, N) array.
    """
    return np.concatenate([det(torch.tensor(emb[i:i + bs], device=device)).softmax(-1)[..., 1].cpu().numpy()
                           for i in range(0, len(emb), bs)])


def decode_cycle(p_insp, k):
    """Phase mask for one annotated cycle: one inspiration and one expiration segment.

    The boundary and the order (I then E, or E then I) maximise the log-likelihood of
    p(inspiration) over the k real frames. The mask is tiled like the repeat-padded audio.
    [Assumption] an ICBHI cycle holds one inspiration and one expiration; the order is not assumed.

    Args:
        p_insp: (N,) p(inspiration) per encoder frame.
        k: Number of frames covered by the real cycle (the rest is repeat padding).

    Returns:
        (N,) int mask, 1 = inspiration, 2 = expiration.
    """
    n = len(p_insp)
    k = int(np.clip(k, 1, n))
    if k == 1:
        return np.full(n, 1 if p_insp[0] >= 0.5 else 2)
    ci = np.r_[0, np.cumsum(np.log(p_insp[:k] + 1e-6))]
    ce = np.r_[0, np.cumsum(np.log(1 - p_insp[:k] + 1e-6))]
    b = np.arange(1, k)
    ie, ei = ci[b] + ce[k] - ce[b], ce[b] + ci[k] - ci[b]
    first = 1 if ie.max() >= ei.max() else 2
    cut = b[np.argmax(ie if first == 1 else ei)]
    return np.resize(np.where(np.arange(k) < cut, first, 3 - first), n)


def add_phase_to_cache(df, cache_dir, det):
    """Write decoded phase masks into existing ICBHI cache files.

    Args:
        df: ICBHI cycle table.
        cache_dir: Embedding cache directory.
        det: Trained `PhaseDetector`.

    Returns:
        Fraction of cycles decoded as inspiration first (sanity check of the cycle convention).
    """
    first_insp = []
    for r in df.itertuples():
        f = Path(cache_dir) / f"{cache_key(r)}.npz"
        d = dict(np.load(f))
        p = insp_prob(det, d["emb"][None].astype(np.float32))[0]
        k = round(min(r.end - r.start, CYCLE_SEC) / CYCLE_SEC * len(p))
        d["phase"] = decode_cycle(p, k).astype(np.int8)
        first_insp.append(d["phase"][0] == 1)
        np.savez(f, **d)
    return float(np.mean(first_insp))


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1]))
    exp = Path(sys.argv[1]).stem
    start_summary(exp)
    write_meta(exp, cfg)
    enc = encoders.load(cfg["encoder"].removesuffix("_v2"))  # legacy cache names end in `_v2`
    rows = file_table(cfg["hf_root"])
    rng = np.random.default_rng(0)
    train = [r for r in rows if r["official"] == "train"]
    test = [r for r in rows if r["official"] == "test"]
    train = [train[i] for i in rng.permutation(len(train))[: cfg.get("n_train")]]  # None = all
    test = [test[i] for i in rng.permutation(len(test))[: cfg.get("n_test")]]
    cache = Path(cfg["cache_root"]) / f"{cfg['encoder']}_hflung"
    xtr, ytr = load(embed_hf(train, enc, cache))
    xte, yte = load(embed_hf(test, enc, cache))
    det = train_detector(xtr, ytr)
    pred = predict_phase(det, xte)
    f1 = f1_score(yte.ravel(), pred.ravel(), average=None)
    acc = float((pred == yte).mean())
    print("HF_Lung test frame acc %.3f, F1 none/insp/exp:" % acc, f1.round(3))
    torch.save({k: v.cpu() for k, v in det.state_dict().items()}, run_dir(exp, "checkpoints") / "detector.pt")
    (run_dir(exp) / "metrics.json").write_text(
        json.dumps(
            {
                "n_train_files": len(train),
                "n_test_files": len(test),
                "test_frame_acc": acc,
                "test_f1_none_insp_exp": f1.tolist(),
            },
            indent=1,
        )
    )
    if cfg.get("apply_to_icbhi"):
        print(
            "ICBHI cycles decoded inspiration-first: %.3f"
            % add_phase_to_cache(
                cycle_table(cfg["icbhi_root"]), Path(cfg["cache_root"]) / cfg["encoder"], det
            )
        )
