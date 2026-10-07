"""Metrics. All CIs are patient-level bootstrap, never cycle-level."""
import numpy as np
from sklearn.metrics import f1_score, recall_score


def class_report(y, pred, n_classes=4):
    """Per-class recall and F1.

    Args:
        y: (n,) true labels.
        pred: (n,) predicted labels.
        n_classes: Number of classes K.

    Returns:
        Dict with `recall` and `f1`, each a (K,) array.
    """
    lab = list(range(n_classes))
    return {"recall": recall_score(y, pred, labels=lab, average=None, zero_division=0),
            "f1": f1_score(y, pred, labels=lab, average=None, zero_division=0)}


def icbhi_score(y, pred):
    """ICBHI Score = (Sp + Se) / 2 with class 0 = normal.

    Sp = normal recall. Se = correct abnormal cycles / all abnormal cycles.

    Args:
        y: (n,) true labels, 0 = normal.
        pred: (n,) predicted labels.

    Returns:
        Float in [0, 1].
    """
    n, a = y == 0, y != 0
    sp = (pred[n] == 0).mean() if n.any() else 0.0
    se = ((pred[a] == y[a]).mean()) if a.any() else 0.0
    return (sp + se) / 2


def ece(probs, y, bins=15):
    """Expected calibration error on top-label confidence.

    Args:
        probs: (n, K) probabilities.
        y: (n,) true labels.
        bins: Number of equal-width bins.

    Returns:
        Float ECE.
    """
    conf, pred = probs.max(1), probs.argmax(1)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs((pred[b == i] == y[b == i]).mean() - conf[b == i].mean()) * (b == i).mean()
               for i in range(bins) if (b == i).any())


def coverage(sets, y):
    """Fraction of prediction sets containing the true label.

    Args:
        sets: (n, K) boolean membership matrix.
        y: (n,) true labels.

    Returns:
        Float.
    """
    return sets[np.arange(len(y)), y].mean()


def patient_bootstrap(metric, patients, n_boot=1000, seed=0, **arrays):
    """Patient-level bootstrap 95% CI. Resamples whole patients with replacement.

    Args:
        metric: Callable taking the same keyword arrays, returning a float.
        patients: (n,) patient id per row.
        n_boot: Number of resamples.
        seed: RNG seed.
        **arrays: (n, ...) arrays passed through to `metric`, e.g. `y=..., pred=...`.

    Returns:
        Tuple (estimate, ci_low, ci_high).
    """
    patients = np.asarray(patients)
    rows = {p: np.flatnonzero(patients == p) for p in np.unique(patients)}
    ids = list(rows)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[ids[i]] for i in rng.choice(len(ids), len(ids))])
        vals.append(metric(**{k: v[idx] for k, v in arrays.items()}))
    return metric(**arrays), *np.percentile(vals, [2.5, 97.5])


def mean_sd(values):
    """Mean and sample SD over seeds.

    Args:
        values: Per-seed metric values.

    Returns:
        Tuple (mean, sd). A gain below `sd` is not a result.
    """
    v = np.asarray(values, dtype=float)
    return v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0


def sp_se(y, pred):
    """Specificity (normal recall) and sensitivity (correct abnormal / all abnormal), class 0 = normal.

    Returns:
        Tuple (sp, se); 0.0 for a missing group.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    n, a = y == 0, y != 0
    return (float((pred[n] == 0).mean()) if n.any() else 0.0,
            float((pred[a] == y[a]).mean()) if a.any() else 0.0)


def hs(y, pred):
    """Harmonic mean of Sp and Se (HS Score)."""
    sp, se = sp_se(y, pred)
    return 2 * sp * se / (sp + se) if sp + se else 0.0


def macro_f1(y, pred, n_classes=4):
    """Macro-averaged F1 over `n_classes` (absent classes count as 0)."""
    return float(class_report(np.asarray(y), np.asarray(pred), n_classes)["f1"].mean())


def by_device(y, pred, devices, n_classes=4):
    """Sp, Se, ICBHI Score, HS and macro-F1 per device, plus the pooled row under key `all`.

    Args:
        y: (n,) true labels.
        pred: (n,) predicted labels.
        devices: (n,) device name per row.
        n_classes: K.

    Returns:
        Dict device -> dict `n`, `sp`, `se`, `score`, `hs`, `macro_f1`, per-class `recall` and the 2-class view `two_cls_{se,score,hs}`
        (the K-class predictions collapsed to normal/abnormal; not a separately trained model).
    """
    y, pred, devices = np.asarray(y), np.asarray(pred), np.asarray(devices)
    out = {}
    for d in [*np.unique(devices), "all"]:
        m = np.ones(len(y), bool) if d == "all" else devices == d
        sp, se = sp_se(y[m], pred[m])
        se2 = float((pred[m][y[m] != 0] > 0).mean()) if (y[m] != 0).any() else 0.0  # abnormal predicted as any abnormal class
        out[str(d)] = {"n": int(m.sum()), "sp": sp, "se": se, "score": (sp + se) / 2, "hs": hs(y[m], pred[m]),
                       "macro_f1": macro_f1(y[m], pred[m], n_classes), "recall": class_report(y[m], pred[m], n_classes)["recall"].tolist(),
                       "two_cls_se": se2,
                       "two_cls_score": (sp + se2) / 2, "two_cls_hs": 2 * sp * se2 / (sp + se2) if sp + se2 else 0.0}
    return out
