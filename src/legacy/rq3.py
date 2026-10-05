"""RQ3 / H4 on Tier 0: conformal coverage under the official-split shift, with and without Tent TTA.

Plain FM (G1: H1 and H2 not shown). Source calibration = val. Target = official ICBHI test, split by
patient: `k_patients` labelled patients calibrate V5 (k-shot) only; every variant is evaluated on the
remaining patients. Their inputs, unlabelled, also feed V3 (density ratio), V4 (label-shift weights)
and Tent. After TTA all calibration scores are recomputed with the adapted model.

Usage: python -m src.legacy.rq3 configs/legacy/rq3_conformal.yaml
"""
import sys
from pathlib import Path

import numpy as np
import yaml

from src.conformal.conformal import (bbse_weights, density_ratio, kshot_threshold, label_shift_threshold, lac_scores,
                                     mondrian_threshold, predict_sets, split_threshold,
                                     weighted_covariate_threshold)
from src.data.icbhi import cycle_table
from src.data.splits import load_split, split_frames
from src.eval.metrics import coverage, ece, icbhi_score, mean_sd
from src.eval.shift import set_stats
from src.runlog import ckpt_path, log_run, start_summary
from src.legacy.train import fit_eval, predict, take, tensors
from src.legacy.tent import tent_adapt

VARIANTS = ("split", "mondrian", "weighted", "label_shift", "kshot")


def thresholds(pv, yv, pk, yk, pe, w_cal, w_eval, alpha, n_classes=4):
    """Thresholds of conformal variants V1-V5.

    Args:
        pv: (m, K) source calibration probabilities.
        yv: (m,) source calibration labels.
        pk: (k, K) labelled target probabilities (V5).
        yk: (k,) their labels.
        pe: (n, K) unlabelled target evaluation probabilities.
        w_cal: (m,) density ratio at calibration points.
        w_eval: (n,) density ratio at evaluation points.
        alpha: Miscoverage level.
        n_classes: K.

    Returns:
        Dict variant -> threshold for `predict_sets`.
    """
    sv = lac_scores(pv, yv)
    w = bbse_weights(pv.argmax(1), yv, pe.argmax(1), n_classes)
    return {"split": split_threshold(sv, alpha),
            "mondrian": mondrian_threshold(sv, yv, alpha, n_classes),
            "weighted": weighted_covariate_threshold(sv, w_cal, w_eval, alpha),
            "label_shift": label_shift_threshold(sv, yv, w, alpha),
            "kshot": kshot_threshold(pk, yk, alpha)}


def set_metrics(sets, y, n_classes=4):
    """Coverage, mean set size and worst class-conditional coverage.

    Args:
        sets: (n, K) boolean membership.
        y: (n,) labels.
        n_classes: K.

    Returns:
        Dict.
    """
    return {"coverage": coverage(sets, y), "set_size": set_stats(sets)["size"],
            "worst_class_cov": min(coverage(sets[y == k], y[y == k]) for k in range(n_classes) if (y == k).any())}


def batches(x, bs):
    """Split input tensors into float batches for `tent_adapt`."""
    return [take(x, slice(i, i + bs)) for i in range(0, len(x["emb"]), bs)]


def run_seed(cfg, seed, ckpt=None):
    """One seed: train plain FM, then conformal variants without TTA and after Tent at each lr.

    Args:
        cfg: Config dict.
        seed: Random seed; also draws the k-shot target patients.
        ckpt: Path to save the trained FM head, or None.

    Returns:
        Dict mode -> {icbhi_score, ece, <variant>: set metrics}. Modes: `none`, `tent_lr<lr>`.
    """
    part = split_frames(cycle_table(cfg["icbhi_root"]), load_split(cfg["split_file"])[0])
    test = part.pop("test")
    kp = np.random.default_rng(seed).choice(sorted(set(test.patient)), cfg["k_patients"], replace=False)
    part["kcal"], part["eval"] = test[test.patient.isin(kp)], test[~test.patient.isin(kp)]
    cache = Path(cfg["cache_root"]) / cfg["encoder"]
    data = {k: tensors(cache, v) for k, v in part.items()}
    _, _, model = fit_eval(cfg, seed, {"train": data["train"], "val": data["val"], "test": data["eval"]},
                           return_model=True, ckpt=ckpt)
    yv, yk, ye = (part[k].label.values for k in ("val", "kcal", "eval"))
    ev, ee = (data[k][0]["emb"].float().mean(1).cpu().numpy() for k in ("val", "eval"))  # frozen encoder
    w = density_ratio(ev, ee, np.vstack([ev, ee]))
    w_cal, w_eval = w[:len(ev)], w[len(ev):]
    out = {}
    for mode, lr in [("none", None)] + [(f"tent_lr{lr:g}", lr) for lr in cfg["tent_lrs"]]:
        mdl = model if lr is None else tent_adapt(model, batches(data["eval"][0], cfg["bs"]) * cfg["tent_epochs"], lr=lr)
        pv, pk, pe = (predict(mdl, data[k][0]) for k in ("val", "kcal", "eval"))
        thr = thresholds(pv, yv, pk, yk, pe, w_cal, w_eval, cfg["alpha"])
        out[mode] = {"icbhi_score": icbhi_score(ye, pe.argmax(1)), "ece": ece(pe, ye),
                     **{v: set_metrics(predict_sets(pe, thr[v]), ye) for v in VARIANTS}}
    return out


if __name__ == "__main__":
    path = Path(sys.argv[1])
    cfg = yaml.safe_load(path.read_text())
    start_summary(path.stem)
    h = load_split(cfg["split_file"])[1]
    res = [run_seed(cfg, s, ckpt_path(path.stem, "fm", s)) for s in cfg["seeds"]]
    for s, r in zip(cfg["seeds"], res):
        log_run(path.stem, "fm", s, cfg, h, cfg["encoder"], r)
    print(f"nominal coverage {1 - cfg['alpha']:.2f}; mean ± SD over {len(res)} seeds")
    for mode in res[0]:
        f = lambda get: "%.3f±%.3f" % mean_sd([get(r[mode]) for r in res])
        print(f"{mode}: icbhi_score={f(lambda r: r['icbhi_score'])} ece={f(lambda r: r['ece'])}")
        for v in VARIANTS:
            print(f"  {v:12s}", " ".join(f"{k}={f(lambda r: r[v][k])}" for k in ("coverage", "set_size", "worst_class_cov")))
