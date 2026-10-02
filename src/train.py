"""Train and evaluate one config on cached embeddings.

Usage: python -m src.train configs/tier0_icbhi_baseline.yaml
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.conformal.conformal import lac_scores, predict_sets, split_threshold
from src.data.icbhi import cycle_table
from src.data.splits import load_split
from src.eval.metrics import class_report, coverage, ece, icbhi_score, mean_sd, set_size
from src.models.cache import load_cycle
from src.models.net import LungModel
from src.runlog import log_run


def load_batch(cache_dir, rows, device):
    """Load cached cycles into tensors.

    Args:
        cache_dir: Embedding cache directory.
        rows: List of cycle table rows.
        device: Torch device.

    Returns:
        Tuple (kwargs dict for `LungModel.forward`, (B,) label tensor).
    """
    emb, wave, phase = zip(*[load_cycle(cache_dir, r) for r in rows])
    b = {k: torch.tensor(np.stack(v)).to(device)
         for k, v in (("emb", emb), ("wave", wave), ("phase", phase))}
    return b, torch.tensor([r.label for r in rows]).to(device)


@torch.no_grad()
def predict(model, cache_dir, df, device, bs=64):
    """Softmax probabilities for every row of `df`.

    Args:
        model: Trained `LungModel`.
        cache_dir: Embedding cache directory.
        df: Cycle table subset.
        device: Torch device.
        bs: Batch size.

    Returns:
        (n, K) numpy array.
    """
    model.eval()
    rows = list(df.itertuples())
    return np.concatenate([model(**load_batch(cache_dir, rows[i:i + bs], device)[0]).softmax(-1).cpu().numpy()
                           for i in range(0, len(rows), bs)])


def run(cfg, seed):
    """Train one seed and evaluate on test with split conformal calibrated on val.

    Args:
        cfg: Config dict (see `configs/`).
        seed: Random seed.

    Returns:
        Metrics dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = cycle_table(cfg["icbhi_root"])
    split, h = load_split(cfg["split_file"])
    part = {k: df[df.patient.isin(v)].reset_index(drop=True) for k, v in split.items()}
    cache = Path(cfg["cache_root"]) / cfg["encoder"]
    emb_dim = load_cycle(cache, next(part["train"].itertuples()))[0].shape[1]
    model = LungModel(emb_dim, use_branch=cfg["use_branch"], use_phase=cfg["use_phase"],
                      shuffle_phase=cfg.get("shuffle_phase", False)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    rows = list(part["train"].itertuples())
    for _ in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(len(rows))
        for i in range(0, len(rows), cfg["bs"]):
            b, y = load_batch(cache, [rows[j] for j in perm[i:i + cfg["bs"]]], dev)
            opt.zero_grad()
            F.cross_entropy(model(**b), y).backward()
            opt.step()
    pv, pt = (predict(model, cache, part[k], dev) for k in ("val", "test"))
    yv, yt = part["val"].label.values, part["test"].label.values
    thr = split_threshold(lac_scores(pv, yv), cfg["alpha"])
    sets = predict_sets(pt, thr)
    rep = class_report(yt, pt.argmax(1))
    return {"icbhi_score": icbhi_score(yt, pt.argmax(1)), "ece": ece(pt, yt),
            "recall": rep["recall"].tolist(), "f1": rep["f1"].tolist(),
            "coverage": coverage(sets, yt), "set_size": set_size(sets)}


if __name__ == "__main__":
    path = Path(sys.argv[1])
    cfg = yaml.safe_load(path.read_text())
    h = load_split(cfg["split_file"])[1]
    res = [run(cfg, s) for s in cfg["seeds"]]
    for s, m in zip(cfg["seeds"], res):
        log_run(path.stem, cfg, s, h, cfg["encoder"], m)
    print("icbhi_score mean±sd: %.4f ± %.4f" % mean_sd([m["icbhi_score"] for m in res]))
