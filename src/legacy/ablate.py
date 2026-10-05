"""2x2 ablation (branch x phase) plus shuffled-phase control, per class, over seeds (H1, H2).

Usage: python -m src.legacy.ablate configs/legacy/tier0_icbhi_baseline.yaml [cell ...]
Optional cell names run a subset; `fm` always runs as the reference.
Phase cells need phase masks in the cache (`src.legacy.phase`). They are skipped otherwise.
"""
import sys
from pathlib import Path

import numpy as np
import yaml

from src.data.icbhi import CLASSES, cycle_table
from src.data.splits import load_split
from src.eval.metrics import class_report, icbhi_score, mean_sd, patient_bootstrap
from src.legacy.cache import load_cycle
from src.runlog import ckpt_path, log_run, start_summary
from src.legacy.train import run

CELLS = {"fm": (False, False, False), "fm+branch": (True, False, False),
         "fm+phase": (False, True, False), "fm+branch+phase": (True, True, False),
         "fm+branch+shuffled_phase": (True, True, True),
         "fm+long_branch": (True, False, False)}  # H1 control: 64 ms branch window, same capacity
BRANCH_WIN = {"fm+long_branch": 1024}
PAIRS = [("fm+long_branch", "fm+branch")]  # extra paired CIs (a, b): b - a. H1: does the short window matter?
PERF = ("icbhi_score",) + tuple(f"{k}_{c}" for c in CLASSES for k in ("recall", "f1"))  # higher is better


def has_phase(cfg):
    """Check whether the first train cycle has a cached phase mask.

    Args:
        cfg: Config dict.

    Returns:
        True if phase masks exist in the cache.
    """
    df = cycle_table(cfg["icbhi_root"])
    split, _ = load_split(cfg["split_file"])
    row = next(df[df.patient.isin(split["train"])].itertuples())
    return bool(load_cycle(Path(cfg["cache_root"]) / cfg["encoder"], row)[2].any())


def summarize(results):
    """Mean ± SD over seeds per cell and metric, and gain over the `fm` baseline.

    Args:
        results: Dict cell -> list of per-seed metric dicts from `src.train.run`.

    Returns:
        Dict cell -> {metric: (mean, sd)} with per-class recall/F1 as `recall_<class>`, `f1_<class>`,
        plus `real_gain` {metric: bool} for performance metrics: True when the gain over `fm`
        is positive and larger than the baseline seed SD (proposal G1). ECE, coverage and set size
        are not gains and are not flagged.
    """
    out = {}
    for cell, runs in results.items():
        m = {k: mean_sd([r[k] for r in runs]) for k in ("icbhi_score", "ece", "coverage", "set_size")}
        for i, c in enumerate(CLASSES):
            for k in ("recall", "f1"):
                m[f"{k}_{c}"] = mean_sd([r[k][i] for r in runs])
        out[cell] = m
    base = out["fm"]
    for cell, m in out.items():
        m["real_gain"] = {k: bool(m[k][0] - base[k][0] > base[k][1]) for k in PERF}
    return out


def paired_diff_ci(y, patients, pred_a, pred_b, metric, n_boot=1000):
    """Patient-bootstrap 95% CI of the seed-averaged metric difference b - a on one test set.

    Captures test-patient sampling variance, which the seed SD leaves out.

    Args:
        y: (n,) labels.
        patients: (n,) patient ids.
        pred_a: (n, S) predicted labels of the reference cell, one column per seed.
        pred_b: (n, S) predicted labels of the compared cell.
        metric: Callable (y, pred) -> float.
        n_boot: Resamples.

    Returns:
        Tuple (estimate, ci_low, ci_high).
    """
    def diff(y, a, b):
        return np.mean([metric(y, b[:, s]) - metric(y, a[:, s]) for s in range(a.shape[1])])
    return patient_bootstrap(diff, patients, n_boot=n_boot, y=y, a=pred_a, b=pred_b)


def per_class_f1(k):
    """F1 of class k as a (y, pred) -> float metric."""
    return lambda y, pred: class_report(y, pred)["f1"][k]


def report(results, preds, y, patients, pairs=()):
    """Print mean ± SD per cell, gains over seed SD, and paired patient-bootstrap CIs.

    Args:
        results: Dict cell -> list of per-seed metric dicts. Must contain `fm`.
        preds: Dict cell -> (n, S) test predictions, one column per seed.
        y: (n,) test labels.
        patients: (n,) test patient (or subject-group) ids.
        pairs: Extra (a, b) cell pairs; prints the CI of b - a.
    """
    metrics = {"icbhi_score": icbhi_score, "f1_crackle": per_class_f1(1), "f1_wheeze": per_class_f1(2)}

    def ci(a, b):
        return {k: "%+.3f [%+.3f, %+.3f]" % paired_diff_ci(y, patients, preds[a], preds[b], f)
                for k, f in metrics.items()}
    for cell, m in summarize(results).items():
        print(cell, " ".join(f"{k}={m[k][0]:.3f}±{m[k][1]:.3f}"
                             for k in ("icbhi_score", "recall_crackle", "recall_wheeze", "f1_crackle", "f1_wheeze")))
        print("  gain > baseline SD:", [k for k, v in m["real_gain"].items() if v])
        if cell != "fm":
            print("  vs fm, patient-bootstrap 95% CI:", ci("fm", cell))
    for a, b in pairs:
        if a in preds and b in preds:
            print(f"{b} vs {a}, patient-bootstrap 95% CI:", ci(a, b))


if __name__ == "__main__":
    path = Path(sys.argv[1])
    cfg = yaml.safe_load(path.read_text())
    start_summary(path.stem)
    h = load_split(cfg["split_file"])[1]
    phase_ok = has_phase(cfg)
    results, preds = {}, {}
    chosen = set(sys.argv[2:]) | {"fm"} if sys.argv[2:] else set(CELLS)
    for cell, (br, ph, shuf) in CELLS.items():
        if cell not in chosen:
            continue
        if ph and not phase_ok:
            print(f"skip {cell}: no cached phase masks (TODO detector)")
            continue
        c = {**cfg, "use_branch": br, "use_phase": ph, "shuffle_phase": shuf, "branch_win": BRANCH_WIN.get(cell, 80)}
        out = [run(c, s, return_probs=True, ckpt=ckpt_path(path.stem, cell, s)) for s in cfg["seeds"]]
        results[cell] = [m for m, _, _ in out]
        preds[cell] = np.stack([p.argmax(1) for _, p, _ in out], 1)
        test = out[0][2]
        for s, m in zip(cfg["seeds"], results[cell]):
            log_run(path.stem, cell, s, c, h, cfg["encoder"], m)
    report(results, preds, test.label.values, test.patient.values, PAIRS)
