"""Phase 0 functions moved out of the v9 core (see archive/README.md)."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


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
