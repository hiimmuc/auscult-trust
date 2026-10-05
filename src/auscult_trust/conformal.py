"""Conformal layer of AuscultTrust: calibrated prediction sets on top of the head's softmax.

Variants here need only source calibration data: V1 split, V2 class-conditional (Mondrian), V5 k-shot
recalibration on labelled target samples. The shift-corrected V3 (weighted) and V4 (label shift) live in
`src.conformal.conformal` because they need unlabelled target data.
"""
import numpy as np

from src.conformal.conformal import kshot_threshold, lac_scores, mondrian_threshold, predict_sets, split_threshold
from src.eval.metrics import coverage, ece
from src.eval.shift import coverage_deficit, per_class_coverage, set_stats

VARIANTS = ("split", "mondrian")


class ConformalLayer:
    """Prediction sets {y : 1 - p(y|x) <= threshold} calibrated on held-out patients.

    Args:
        alpha: Miscoverage level (0.1 = 90% nominal).
        variant: `split` (V1, one threshold) or `mondrian` (V2, one threshold per class).
        n_classes: Number of classes K.
    """

    def __init__(self, alpha=0.1, variant="split", n_classes=4):
        assert variant in VARIANTS, variant
        self.alpha, self.variant, self.K = alpha, variant, n_classes
        self.threshold = None

    def calibrate(self, probs, y):
        """Set the threshold from calibration probabilities and labels (calibration patients only)."""
        s = lac_scores(probs, y)
        self.threshold = split_threshold(s, self.alpha) if self.variant == "split" else mondrian_threshold(s, y, self.alpha, self.K)
        return self

    def recalibrate_kshot(self, probs_k, y_k):
        """V5 oracle: replace the threshold by one computed on k labelled target samples (split variant)."""
        self.threshold = kshot_threshold(probs_k, y_k, self.alpha)
        return self

    def sets(self, probs):
        """(n, K) boolean prediction sets."""
        assert self.threshold is not None, "calibrate first"
        return predict_sets(probs, self.threshold)

    def evaluate(self, probs, y):
        """Coverage, deficit, per-class coverage, set statistics and ECE of the sets on (probs, y)."""
        s = self.sets(probs)
        cov = float(coverage(s, y))
        return {"alpha": self.alpha, "variant": self.variant, "coverage": cov, "deficit": float(coverage_deficit(cov, self.alpha)),
                "per_class": per_class_coverage(s, y, self.K, self.alpha), **set_stats(s), "ece": float(ece(probs, y)),
                "threshold": np.asarray(self.threshold).tolist()}
