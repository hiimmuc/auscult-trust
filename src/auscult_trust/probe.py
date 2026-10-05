"""L0 reference rung: multinomial logistic regression on the masked clip mean of the frozen last layer.

L2 penalty, class-weighted, inputs z-scored with train-patient statistics, C chosen by patient-grouped CV
over the dev (train+val) patients. Deterministic, so there are no seeds: uncertainty is a patient-level
bootstrap of the test set.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.auscult_trust.conformal import ConformalLayer
from src.auscult_trust.data import folds
from src.eval.metrics import class_report, icbhi_score, patient_bootstrap

C_GRID = (1e-3, 1e-2, 1e-1, 1, 10, 100)


def clip_features(D):
    """Mean over real (non-repeated) frames of the last cached layer, every cache row. Returns (N, dim) float32."""
    f = D.layers[:, -1].float()  # (N, 32, dim)
    m = D.mask.unsqueeze(-1)
    return ((f * m).sum(1) / m.sum(1)).numpy()


def fit_lr(X, y, C):
    """z-score with train statistics, then fit. Returns a callable X -> class probabilities."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit((X - mu) / sd, y)
    return lambda Z: clf.predict_proba((Z - mu) / sd)


def run_probe(cfg, D, alpha=0.1):
    """CV for C, final fit on all dev patients, test once, conformal on one held-out patient fold.

    Returns:
        Dict with `cv` (score per C), `C`, test `icbhi_score` + bootstrap CI, per-class recall/F1, and
        `split` / `mondrian` conformal results (model trained on 4/5 of dev, calibrated on the rest).
    """
    X, y = clip_features(D), D.y
    fs = folds(D, cfg["cv_folds"])
    cv = {}
    for C in C_GRID:
        sc = [icbhi_score(y[te], fit_lr(X[tr], y[tr], C)(X[te]).argmax(1)) for tr, te in fs]
        cv[C] = (float(np.mean(sc)), float(np.std(sc, ddof=1)))
    best = max(cv, key=lambda c: cv[c][0])
    dev, test = D.dev.row.values, D.test.row.values
    yte = y[test]
    pred = fit_lr(X[dev], y[dev], best)(X[test])
    rep = class_report(yte, pred.argmax(1))
    est, lo, hi = patient_bootstrap(lambda y, p: icbhi_score(y, p), D.test.patient.values, y=yte, p=pred.argmax(1))
    tr, cal = fs[0]
    p_model = fit_lr(X[tr], y[tr], best)
    out = {"cv": {str(c): v for c, v in cv.items()}, "C": best, "icbhi_score": float(est), "ci": [float(lo), float(hi)],
           "recall": rep["recall"].tolist(), "f1": rep["f1"].tolist(), "test_pred": pred.argmax(1).tolist(), "n_cal": len(cal)}
    for v in ("split", "mondrian"):
        out[v] = ConformalLayer(alpha, v).calibrate(p_model(X[cal]), y[cal]).evaluate(p_model(X[test]), yte)
    return out
