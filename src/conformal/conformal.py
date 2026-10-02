"""Conformal prediction variants V1-V5 (proposal 4.3).

Score is LAC: s(x, y) = 1 - p(y | x). A prediction set is {y : s(x, y) <= threshold}.
Thresholds broadcast to (n, K): scalar (split), (K,) per candidate label
(Mondrian, label-shift) or (n, 1) per test point (covariate-shift weighted).

After TTA, recompute `probs` for the calibration set with the adapted model
before calling any calibrate function. Old scores are invalid.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


def lac_scores(probs, labels):
    """Nonconformity score of the true label.

    Args:
        probs: (n, K) class probabilities.
        labels: (n,) integer labels.

    Returns:
        (n,) scores.
    """
    return 1.0 - probs[np.arange(len(labels)), labels]


def weighted_threshold(cal_scores, cal_w, test_w, alpha):
    """Weighted conformal quantile with the test point's own mass at +inf.

    Args:
        cal_scores: (m,) calibration scores.
        cal_w: (m,) calibration weights.
        test_w: Scalar or array of test weights.
        alpha: Miscoverage level.

    Returns:
        Threshold(s) shaped like `test_w`. `inf` when calibration data is too small.
    """
    order = np.argsort(cal_scores)
    s, cw = cal_scores[order], np.cumsum(cal_w[order])
    target = (1 - alpha) * (cw[-1] + np.asarray(test_w, dtype=float))
    idx = np.searchsorted(cw, target, side="left")
    return np.where(idx >= len(s), np.inf, s[np.minimum(idx, len(s) - 1)])


def split_threshold(scores, alpha):
    """V1 split conformal threshold.

    Args:
        scores: (m,) calibration scores.
        alpha: Miscoverage level.

    Returns:
        Scalar threshold.
    """
    return weighted_threshold(scores, np.ones(len(scores)), 1.0, alpha)


def mondrian_threshold(scores, labels, alpha, n_classes):
    """V2 class-conditional threshold.

    Args:
        scores: (m,) calibration scores.
        labels: (m,) calibration labels.
        alpha: Miscoverage level.
        n_classes: Number of classes K.

    Returns:
        (K,) thresholds. `inf` for a class with too few calibration points.
    """
    return np.array([split_threshold(scores[labels == k], alpha) if (labels == k).any()
                     else np.inf for k in range(n_classes)])


def density_ratio(src_emb, tgt_emb, x):
    """Estimate target/source density ratio with a domain classifier.

    Args:
        src_emb: (n_s, d) source embeddings.
        tgt_emb: (n_t, d) unlabelled target embeddings.
        x: (n, d) points to weight.

    Returns:
        (n,) weights p/(1-p), corrected for class sizes.
    """
    X = np.vstack([src_emb, tgt_emb])
    d = np.r_[np.zeros(len(src_emb)), np.ones(len(tgt_emb))]
    p = LogisticRegression(max_iter=1000).fit(X, d).predict_proba(x)[:, 1]
    return np.clip(p / (1 - p + 1e-9), 1e-3, 1e3) * len(src_emb) / len(tgt_emb)


def weighted_covariate_threshold(cal_scores, cal_w, test_w, alpha):
    """V3 weighted conformal under covariate shift.

    Args:
        cal_scores: (m,) calibration scores.
        cal_w: (m,) density ratio at calibration points.
        test_w: (n,) density ratio at test points.
        alpha: Miscoverage level.

    Returns:
        (n, 1) thresholds, one per test point.
    """
    return weighted_threshold(cal_scores, cal_w, test_w, alpha)[:, None]


def bbse_weights(cal_pred, cal_labels, tgt_pred, n_classes):
    """Estimate label-shift weights q(y)/p(y) by black-box shift estimation.

    Args:
        cal_pred: (m,) predicted labels on source calibration data.
        cal_labels: (m,) true labels on source calibration data.
        tgt_pred: (n,) predicted labels on unlabelled target data.
        n_classes: Number of classes K.

    Returns:
        (K,) non-negative weights.
    """
    C = np.zeros((n_classes, n_classes))
    for k in range(n_classes):
        m = cal_labels == k
        if m.any():
            C[:, k] = np.bincount(cal_pred[m], minlength=n_classes) / m.sum()
    mu = np.bincount(tgt_pred, minlength=n_classes) / len(tgt_pred)
    return np.clip(np.linalg.lstsq(C, mu, rcond=None)[0], 0, None)


def label_shift_threshold(cal_scores, cal_labels, class_w, alpha):
    """V4 conformal under label shift.

    Args:
        cal_scores: (m,) calibration scores.
        cal_labels: (m,) calibration labels.
        class_w: (K,) weights q(y)/p(y), e.g. from `bbse_weights`.
        alpha: Miscoverage level.

    Returns:
        (K,) thresholds, one per candidate label.
    """
    return weighted_threshold(cal_scores, class_w[cal_labels], class_w, alpha)


def kshot_threshold(probs_k, labels_k, alpha):
    """V5 recalibration on k labelled target samples.

    Args:
        probs_k: (k, K) probabilities from the (adapted) model on the target samples.
        labels_k: (k,) their labels.
        alpha: Miscoverage level.

    Returns:
        Scalar threshold. Exchangeable within target, so coverage is exact.
    """
    return split_threshold(lac_scores(probs_k, labels_k), alpha)


def predict_sets(probs, threshold):
    """Build prediction sets.

    Args:
        probs: (n, K) class probabilities.
        threshold: Scalar, (K,) or (n, 1) threshold.

    Returns:
        (n, K) boolean membership matrix.
    """
    return (1.0 - probs) <= threshold


def abstain(sets):
    """Abstain when the set is empty or has more than one label.

    Args:
        sets: (n, K) boolean membership matrix.

    Returns:
        (n,) boolean, True = abstain.
    """
    return sets.sum(1) != 1
