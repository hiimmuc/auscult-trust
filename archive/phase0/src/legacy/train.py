"""Train and evaluate one config on cached embeddings.

Usage: python -m src.legacy.train configs/legacy/tier0_icbhi_baseline.yaml
"""

import copy
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from src.conformal.conformal import lac_scores, predict_sets, split_threshold
from src.data.icbhi import cycle_table
from src.data.splits import load_split, split_frames
from src.eval.metrics import class_report, coverage, ece, icbhi_score, mean_sd
from src.eval.shift import set_stats
from src.legacy.cache import load_cycle
from src.legacy.net import LungModel, device
from src.runlog import ckpt_path, log_run, start_summary


@lru_cache(maxsize=4)  # train, val, test of one cache
def load_cache(cache_dir, stems_starts_ends):
    """Load cached cycles once into device tensors (float16), shared across seeds and cells.

    Args:
        cache_dir: Embedding cache directory (str, for hashing).
        stems_starts_ends: Tuple of (stem, start, end) per row.

    Returns:
        Dict with `emb` (n, N, D), `wave` (n, T), `phase` (n, N) tensors.
    """
    rows = [SimpleNamespace(stem=s, start=a, end=b) for s, a, b in stems_starts_ends]
    emb, wave, phase = map(np.stack, zip(*[load_cycle(cache_dir, r) for r in rows]))
    return {
        "emb": torch.tensor(emb, dtype=torch.float16, device=device),
        "wave": torch.tensor(wave, dtype=torch.float16, device=device),
        "phase": torch.tensor(phase, device=device),
    }


def tensors(cache_dir, df):
    """Device tensors for the rows of `df`.

    Args:
        cache_dir: Embedding cache directory.
        df: Cycle table subset.

    Returns:
        Tuple (dict of model inputs, (n,) label tensor).
    """
    d = load_cache(str(cache_dir), tuple(zip(df.stem, df.start, df.end)))
    return d, torch.tensor(df.label.values, device=device)


def take(data, idx):
    """Rows `idx` of every input tensor, floating tensors upcast to float32."""
    return {k: v[idx].float() if v.is_floating_point() else v[idx] for k, v in data.items()}


@torch.no_grad()
def predict(model, data, bs=256):
    """Softmax probabilities for every row of `data`.

    Args:
        model: `LungModel`.
        data: Dict of input tensors from `tensors`.
        bs: Batch size.

    Returns:
        (n, K) numpy array.
    """
    model.eval()
    n = len(data["emb"])
    return np.concatenate([model(**take(data, slice(i, i + bs))).softmax(-1).cpu().numpy() for i in range(0, n, bs)])


def fit_eval(cfg, seed, data, return_model=False, ckpt=None):
    """Train one seed, keep the epoch with the best val ICBHI Score, evaluate on test.

    Split conformal is calibrated on val. Val also picks the epoch; LAC scores play no part in
    that choice. # ponytail: same val for both, carve a separate calibration set if coverage drifts.

    Args:
        cfg: Config dict with `use_branch`, `use_phase`, `lr`, `wd`, `epochs`, `bs`, `alpha`, optional
            `shuffle_phase`, `branch_win`, `class_weight`.
        seed: Random seed.
        data: Dict `train`/`val`/`test` -> (dict of input tensors, (n,) label tensor).
        return_model: Also return the trained model.
        ckpt: Path to save the best-epoch head weights (`runlog.ckpt_path`), or None.

    Returns:
        Tuple (metrics dict, (n_test, K) test probabilities), plus the model when `return_model`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    (xtr, ytr), (xv, yv), (xt, yt) = data["train"], data["val"], data["test"]
    yv, yt = yv.cpu().numpy(), yt.cpu().numpy()
    model = LungModel(xtr["emb"].shape[-1], use_branch=cfg["use_branch"], use_phase=cfg["use_phase"],
                      shuffle_phase=cfg.get("shuffle_phase", False), branch_win=cfg.get("branch_win", 80)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    w = None
    if cfg.get("class_weight"):
        cnt = torch.bincount(ytr, minlength=4).float()
        w = cnt.sum() / (4 * cnt.clamp(min=1))
    best, state = -1.0, None
    for _ in range(cfg["epochs"]):
        model.train()
        for idx in torch.randperm(len(ytr), device=device).split(cfg["bs"]):
            opt.zero_grad()
            F.cross_entropy(model(**take(xtr, idx)), ytr[idx], weight=w).backward()
            opt.step()
        score = icbhi_score(yv, predict(model, xv).argmax(1))
        if score > best:
            best, state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    if ckpt:
        torch.save({"state": state, "cfg": cfg, "seed": seed}, ckpt)
    pv, pt = predict(model, xv), predict(model, xt)
    sets = predict_sets(pt, split_threshold(lac_scores(pv, yv), cfg["alpha"]))
    rep = class_report(yt, pt.argmax(1))
    m = {"icbhi_score": icbhi_score(yt, pt.argmax(1)), "val_icbhi_score": best, "ece": ece(pt, yt),
         "recall": rep["recall"].tolist(), "f1": rep["f1"].tolist(),
         "coverage": coverage(sets, yt), "set_size": set_stats(sets)["size"],
         "test_pred": pt.argmax(1).tolist()}  # rows in test order, for paired CIs later
    return (m, pt, model) if return_model else (m, pt)


def run(cfg, seed, return_probs=False, ckpt=None):
    """ICBHI run: load the cached split, then `fit_eval`.

    Args:
        cfg: Config dict (see `configs/`).
        seed: Random seed.
        return_probs: Also return test probabilities and the test table.
        ckpt: Path to save the head weights, or None.

    Returns:
        Metrics dict, or (metrics, test probs, test DataFrame) when `return_probs`.
    """
    part = split_frames(cycle_table(cfg["icbhi_root"]), load_split(cfg["split_file"])[0])
    cache = Path(cfg["cache_root"]) / cfg["encoder"]
    m, pt = fit_eval(cfg, seed, {k: tensors(cache, v) for k, v in part.items()}, ckpt=ckpt)
    return (m, pt, part["test"]) if return_probs else m


if __name__ == "__main__":
    path = Path(sys.argv[1])
    cfg = yaml.safe_load(path.read_text())
    start_summary(path.stem)
    h = load_split(cfg["split_file"])[1]
    res = [run(cfg, s, ckpt=ckpt_path(path.stem, "run", s)) for s in cfg["seeds"]]
    for s, m in zip(cfg["seeds"], res):
        log_run(path.stem, "run", s, cfg, h, cfg["encoder"], m)
    print("icbhi_score mean±sd: %.4f ± %.4f" % mean_sd([m["icbhi_score"] for m in res]))
