"""Split-conformal layer: plain split, k-shot recalibration (k patients), oracle label shift.

Score is LAC: s(x, y) = 1 - p(y | x). A prediction set is {y : s(x, y) <= threshold}.
Thresholds broadcast to (n, K): scalar (split) or (K,) per candidate label (label shift).

After TTA, recompute `probs` for the calibration set with the adapted model
before calling any calibrate function. Old scores are invalid.
"""
import numpy as np


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


def oracle_label_shift_threshold(cal_scores, cal_labels, cal_freq, tgt_freq, alpha):
    """V4-oracle: label-shift conformal with the true class frequencies instead of BBSE estimates.

    Upper bound for V4: the gap to `label_shift_threshold` is the cost of estimating the weights.

    Args:
        cal_scores: (m,) calibration scores.
        cal_labels: (m,) calibration labels.
        cal_freq: (K,) true class frequencies of the calibration data.
        tgt_freq: (K,) true class frequencies of the target test data.
        alpha: Miscoverage level.

    Returns:
        (K,) thresholds, one per candidate label.
    """
    w = np.asarray(tgt_freq, float) / np.maximum(np.asarray(cal_freq, float), 1e-12)
    return label_shift_threshold(cal_scores, cal_labels, w, alpha)


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


def kshot_patients_threshold(probs, labels, patients, k, alpha, seed=0):
    """V5 with k counted in target patients: all rows of k randomly chosen target patients calibrate.

    Rows of one patient are not exchangeable with other patients' rows, so k is a patient count (use 5 or 10).

    Args:
        probs: (n, K) target probabilities.
        labels: (n,) target labels.
        patients: (n,) patient id per row.
        k: Number of labelled target patients.
        alpha: Miscoverage level.
        seed: RNG seed.

    Returns:
        Tuple (threshold, mask) with `mask` (n,) True for calibration rows; evaluate on `~mask`.
    """
    ids = np.unique(patients)
    chosen = np.random.default_rng(seed).choice(ids, min(k, len(ids)), replace=False)
    mask = np.isin(patients, chosen)
    return kshot_threshold(probs[mask], labels[mask], alpha), mask


def predict_sets(probs, threshold):
    """Build prediction sets.

    Args:
        probs: (n, K) class probabilities.
        threshold: Scalar, (K,) or (n, 1) threshold.

    Returns:
        (n, K) boolean membership matrix.
    """
    return (1.0 - probs) <= threshold

