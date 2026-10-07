"""H2 mechanism control on HF_Lung_V1: phase pooling with true vs predicted phase (proposal 4.2).

HF_Lung has frame-level phase labels and adventitious-sound labels, so phase pooling can be tested
without detector error. Each 8 s window (`src.legacy.phase.WINDOWS`) gets the 4 ICBHI classes:
crackle = any D event overlapping the window, CAS = any Wheeze/Stridor/Rhonchi event overlapping it.
The phase mask is inspiration (1) vs rest (2): expiration is labelled in only about half of the
breaths (see `src.legacy.phase`), so "rest" stands in for expiration. Predicted masks are
p(inspiration) > 0.5 from a detector that never saw the window: 2-fold cross-fit over subject
groups on official train, full-train detector on official test.

Usage: python -m src.legacy.hf_ablate configs/legacy/hf_phase_ablation.yaml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.legacy.ablate import report
from src.data.audio import CYCLE_SEC
from src.legacy.hf_lung import file_table, read_labels
from src.data.splits import load_split, save_split, split_patients
from src.legacy.net import device
from src.legacy.phase import WINDOWS, insp_prob, train_detector
from src.runlog import ckpt_path, log_run, start_summary
from src.legacy.train import fit_eval

CELLS = {"fm": (False, None, False), "fm+true_phase": (True, "true", False),
         "fm+pred_phase": (True, "pred", False), "fm+shuffled_true_phase": (True, "true", True)}
PAIRS = [("fm+pred_phase", "fm+true_phase")]  # true minus predicted: cost of detector error
CAS = ("Wheeze", "Stridor", "Rhonchi")


def window_label(labels, start, sec=CYCLE_SEC):
    """4-class label of one window: 0 normal, 1 crackle, 2 CAS (wheeze-like), 3 both.

    Args:
        labels: Output of `read_labels`.
        start: Window start in seconds.
        sec: Window length in seconds.

    Returns:
        Int label.
    """
    def hit(types):
        return any(t in types and s < start + sec and e > start for t, s, e in labels)
    return int(hit(("D",))) + 2 * int(hit(CAS))


def window_table(root):
    """One row per cached HF_Lung window.

    Args:
        root: HF_Lung_V1 directory.

    Returns:
        DataFrame with `stem, w, patient` (date group), `official, label`.
    """
    rows = []
    for r in file_table(root):
        lab = read_labels(r["label"])
        rows += [{"stem": Path(r["wav"]).stem, "w": int(w), "patient": r["group"], "official": r["official"],
                  "label": window_label(lab, w)} for w in WINDOWS]
    return pd.DataFrame(rows)


def crossfit_insp(emb, phase, groups, train, seed=0):
    """p(inspiration) for every window from a detector that did not train on it.

    Args:
        emb: (n, N, D) float32 embeddings.
        phase: (n, N) true labels {0, 1, 2}.
        groups: (n,) subject groups.
        train: (n,) bool, official-train windows (2-fold cross-fit). Others get the full-train detector.
        seed: Fold assignment seed.

    Returns:
        (n, N) array.
    """
    g = np.unique(groups[train])
    fold = dict(zip(g, np.random.default_rng(seed).permutation(len(g)) % 2))
    f = np.array([fold.get(x, -1) for x in groups])
    p = np.zeros(phase.shape, np.float32)
    for k in (0, 1):
        det = train_detector(emb[train & (f != k)], phase[train & (f != k)])
        p[f == k] = insp_prob(det, emb[f == k])
    p[~train] = insp_prob(train_detector(emb[train], phase[train]), emb[~train])
    return p


if __name__ == "__main__":
    path = Path(sys.argv[1])
    cfg = yaml.safe_load(path.read_text())
    start_summary(path.stem)
    df = window_table(cfg["hf_root"])
    if not Path(cfg["split_file"]).exists():
        s = split_patients(df.patient[df.official == "train"], (0.85, 0.15), 0, ("train", "val"))
        save_split({**s, "test": sorted(set(df.patient[df.official == "test"]))}, cfg["split_file"])
    split, h = load_split(cfg["split_file"])
    cache = Path(cfg["cache_root"]) / f"{cfg['encoder']}_hflung"
    emb, phase = [], []
    for r in df.itertuples():
        with np.load(cache / f"{r.stem}_{r.w}.npz") as x:
            emb.append(x["emb"])
            phase.append(x["phase"])
    emb, phase = np.stack(emb).astype(np.float32), np.stack(phase).astype(np.int64)
    p = crossfit_insp(emb, phase, df.patient.values, (df.official == "train").values)
    pred_ok = ((p > 0.5) == (phase == 1))[df.official.values == "test"].mean()
    print(f"HF_Lung test frame accuracy, predicted vs true inspiration-vs-rest mask: {pred_ok:.3f}")
    masks = {"true": np.where(phase == 1, 1, 2), "pred": np.where(p > 0.5, 1, 2), None: phase}
    idx = {k: np.flatnonzero(df.patient.isin(v).values) for k, v in split.items()}
    results, preds = {}, {}
    for cell, (ph, src, shuf) in CELLS.items():
        data = {k: ({"emb": torch.tensor(emb[i], dtype=torch.float16, device=device),
                     "phase": torch.tensor(masks[src][i], device=device)},
                    torch.tensor(df.label.values[i], device=device)) for k, i in idx.items()}
        c = {**cfg, "use_branch": False, "use_phase": ph, "shuffle_phase": shuf}
        out = [fit_eval(c, s, data, ckpt=ckpt_path(path.stem, cell, s)) for s in cfg["seeds"]]
        results[cell] = [m for m, _ in out]
        preds[cell] = np.stack([pt.argmax(1) for _, pt in out], 1)
        for s, m in zip(cfg["seeds"], results[cell]):
            log_run(path.stem, cell, s, c, h, cfg["encoder"], m)
        del data
    test = df.iloc[idx["test"]]
    report(results, preds, test.label.values, test.patient.values, PAIRS)
