"""Device and class probes on the pooled embeddings of LODO units (how much device information the encoder keeps).

    python -m src.evaluation.probe --run lodo-ast --arms baseline_plain sc_plain

Per unit (`<arm>+lodo_<dev>/seed<s>/emb.npz` from `src.evaluation.lodo_ttsc`): embeddings of the calibration cycles (source
devices) and of the held-out device, so all four devices are present. Devices with at least 10 patients are kept; cycles are
subsampled to a balanced device x class table (the smallest cell size, seeded). Logistic regression (standardised features),
patient-grouped 5-fold CV. Reported: device balanced accuracy and log-loss, class balanced accuracy, and two references:
chance (1 / number of devices) and a probe that sees only the one-hot class label (device information carried by the class
mix alone).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.processing.icbhi import DEVICES

MIN_PATIENTS = 10


def load(unit):
    e = np.load(unit / "emb.npz")
    c, t = np.load(unit / "cal_preds.npz"), np.load(unit / "test_preds.npz")
    return (
        np.concatenate([e["cal"], e["test"]]).astype(np.float32),
        np.concatenate([c["labels"], t["labels"]]),
        np.concatenate([c["device"], t["device"]]),
        np.concatenate([c["patient"], t["patient"]]),
    )


def balanced_subsample(y, dev, seed):
    rng = np.random.default_rng(seed)
    cells = {(d, k): np.flatnonzero((dev == d) & (y == k)) for d in np.unique(dev) for k in np.unique(y)}
    cells = {k: v for k, v in cells.items() if len(v)}
    n = min(len(v) for v in cells.values())
    return np.sort(np.concatenate([rng.choice(v, n, replace=False) for v in cells.values()]))


def cv_probe(X, target, groups, n_classes, folds=5):
    """Patient-grouped CV of a logistic-regression probe: (balanced accuracy, log-loss) over out-of-fold predictions."""
    proba = np.zeros((len(target), n_classes))
    labels = np.arange(n_classes)
    for tr, te in GroupKFold(folds).split(X, target, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        m.fit(X[tr], target[tr])
        p = np.zeros((len(te), n_classes))
        p[:, m.classes_] = m.predict_proba(X[te])
        proba[te] = p
    return float(balanced_accuracy_score(target, proba.argmax(1))), float(log_loss(target, np.clip(proba, 1e-7, 1), labels=labels))


def probe_unit(unit, seed=0):
    X, y, dev, pat = load(unit)
    keep_dev = [d for d in np.unique(dev) if len(np.unique(pat[dev == d])) >= MIN_PATIENTS]
    if len(keep_dev) < 2:  # a device probe needs two devices; the source devices of a unit hold only its calibration patients
        return None
    m = np.isin(dev, keep_dev)
    X, y, dev, pat = X[m], y[m], dev[m], pat[m]
    idx = balanced_subsample(y, dev, seed)
    X, y, dev, pat = X[idx], y[idx], dev[idx], pat[idx]
    dmap = {d: i for i, d in enumerate(keep_dev)}
    dt = np.array([dmap[d] for d in dev])
    onehot = np.eye(4)[y]
    out = {"devices": [DEVICES[d] for d in keep_dev], "n": int(len(y)), "chance": 1 / len(keep_dev)}
    out["device_bacc"], out["device_logloss"] = cv_probe(X, dt, pat, len(keep_dev))
    out["label_only_device_bacc"], out["label_only_device_logloss"] = cv_probe(onehot, dt, pat, len(keep_dev))
    out["class_bacc"], out["class_logloss"] = cv_probe(X, y, pat, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="lodo-ast")
    ap.add_argument("--arms", nargs="+", default=["baseline_plain", "sc_plain"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    root = Path("outputs/train") / a.run
    res, L = {}, ["| Held-out fold | Arm | Seeds | Devices | Device bal. acc | Device log-loss | Label-only device bal. acc | Chance | Class bal. acc |",
                  "|---|---|---|---|---|---|---|---|---|"]
    for dev in DEVICES:
        for arm in a.arms:
            rows = [probe_unit(root / "{}+lodo_{}".format(arm, dev.lower()) / "seed{}".format(s), s) for s in a.seeds
                    if (root / "{}+lodo_{}".format(arm, dev.lower()) / "seed{}".format(s) / "emb.npz").exists()]
            if None in rows:
                L.append("| {} | {} | - | fewer than two devices with {} patients | n/a | | | | |".format(dev, arm, MIN_PATIENTS))
                continue
            if not rows:
                continue
            res["{}|{}".format(dev, arm)] = rows
            mean = lambda k: np.mean([r[k] for r in rows])
            sd = lambda k: np.std([r[k] for r in rows], ddof=1) if len(rows) > 1 else 0.0
            L.append("| {} | {} | {} | {} | {:.3f} ± {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} ± {:.3f} |".format(
                dev, arm, len(rows), ", ".join(rows[0]["devices"]), mean("device_bacc"), sd("device_bacc"), mean("device_logloss"),
                mean("label_only_device_bacc"), mean("chance"), mean("class_bacc"), sd("class_bacc")))
    md = "\n".join(L) + "\n"
    (root / "probe_summary.md").write_text(md)
    (root / "probe_summary.json").write_text(json.dumps(res, indent=1))
    print(md)


if __name__ == "__main__":
    main()
