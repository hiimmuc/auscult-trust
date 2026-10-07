"""Stage 2 entry for a frozen full model: read the softmax export of `baselines/patchmix_cl/export_probs.py` and run E1.

E1 (device held out): calibration and evaluation patients come from the official test patients of the other devices,
split 20 times (stratified by device); the held-out device's test cycles are the shifted target. Patients recorded on the
held-out device and another one are dropped from the other devices, so patients stay disjoint.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.conformal import lac_scores, predict_sets, split_threshold
from src.processing.icbhi import DEVICES
from src.processing.splits import calibration_eval_splits
from src.evaluation.metrics import coverage
from src.evaluation.shift import prior_matched_coverage


def load_export(folder, split="test", weights=None):
    """Read `<folder>/<split>.npz`; when `weights` is given, check it against `<folder>/model.sha256`.

    Returns:
        Dict `probs`, `emb`, `labels`, `device` (names), `patient`.
    """
    folder = Path(folder)
    if weights is not None:
        h = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        assert h == (folder / "model.sha256").read_text().split()[0], "weights differ from the exported model"
    d = np.load(folder / f"{split}.npz", allow_pickle=True)
    return {"probs": d["probs"], "emb": d["emb"], "labels": d["labels"],
            "device": np.array(DEVICES)[d["device"]], "patient": d["patient"]}


def e1_coverage(exp, held, alpha=0.1, n_splits=20, seed=0):
    """V1 split-conformal coverage of a frozen model on a held-out device, over `n_splits` calibration/evaluation re-splits.

    Args:
        exp: Dict from `load_export` (test split).
        held: Held-out device name.
        alpha: Miscoverage level.
        n_splits: Number of re-splits of the other devices' test patients.
        seed: Base seed of the re-splits.

    Returns:
        Dict of (n_splits,) arrays `cov_in` (evaluation patients of the other devices), `cov_target` (held-out device),
        `cov_target_prior` (target coverage re-weighted to the calibration class mix), plus `n_target`, `n_dropped`.
    """
    df = pd.DataFrame({"patient": exp["patient"], "device": exp["device"]})
    shared = set(df.loc[df.device == held, "patient"])
    keep = (df.device != held) & ~df.patient.isin(shared)
    target = (df.device == held).to_numpy()
    out = {"cov_in": [], "cov_target": [], "cov_target_prior": []}
    y, probs = exp["labels"], exp["probs"]
    for s in calibration_eval_splits(df[keep], n_splits, seed=seed):
        cal = df.patient.isin(s["calibration"]).to_numpy() & keep.to_numpy()
        ev = df.patient.isin(s["evaluation"]).to_numpy() & keep.to_numpy()
        thr = split_threshold(lac_scores(probs[cal], y[cal]), alpha)
        prior = np.bincount(y[cal], minlength=probs.shape[1]) / cal.sum()
        sets_t = predict_sets(probs[target], thr)
        out["cov_in"].append(coverage(predict_sets(probs[ev], thr), y[ev]))
        out["cov_target"].append(coverage(sets_t, y[target]))
        out["cov_target_prior"].append(prior_matched_coverage(sets_t, y[target], prior))
    return {**{k: np.array(v) for k, v in out.items()}, "n_target": int(target.sum()),
            "n_dropped": int((~keep & ~target).sum())}
