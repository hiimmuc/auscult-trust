"""Coverage and set-size metrics on a new device: Delta = (1 - alpha) - coverage."""
import numpy as np


def coverage_deficit(cov, alpha):
    """Nominal minus empirical coverage. Positive means under-coverage."""
    return (1 - alpha) - cov




def set_stats(sets):
    """Mean size, singleton rate and empty-set rate of prediction sets.

    Args:
        sets: (n, K) boolean membership matrix.

    Returns:
        Dict `size`, `singleton`, `empty`.
    """
    n = sets.sum(1)
    return {"size": float(n.mean()), "singleton": float((n == 1).mean()), "empty": float((n == 0).mean())}


def binom_ci(p, n, z=1.96):
    """Normal-approximation 95% CI half-width of a proportion. Returns 0 for n = 0."""
    return z * np.sqrt(p * (1 - p) / n) if n else 0.0


def per_class_coverage(sets, y, n_classes, alpha=0.1):
    """Coverage per true class, with n and the binomial half-width at nominal.

    Args:
        sets: (n, K) boolean membership matrix.
        y: (n,) true labels.
        n_classes: K.
        alpha: Miscoverage level (for the nominal CI).

    Returns:
        List of dicts `n`, `coverage`, `half_width`; coverage is NaN for an empty class.
    """
    out = []
    for k in range(n_classes):
        m = y == k
        out.append({"n": int(m.sum()), "coverage": float(sets[m, k].mean()) if m.any() else float("nan"),
                    "half_width": float(binom_ci(1 - alpha, m.sum()))})
    return out










def prior_matched_coverage(sets, y, target_prior):
    """Coverage re-weighted so the class mix equals `target_prior`. Evaluation only, never used to calibrate.

    Separates the device effect from the class-proportion effect: compare with plain coverage on the same rows.

    Args:
        sets: (n, K) boolean membership matrix.
        y: (n,) true labels.
        target_prior: (K,) class proportions to match, e.g. those of the calibration data.

    Returns:
        Float; classes absent from `y` are dropped and the rest renormalised.
    """
    K = sets.shape[1]
    per = np.array([sets[y == k, k].mean() if (y == k).any() else np.nan for k in range(K)])
    w = np.where(np.isnan(per), 0.0, np.asarray(target_prior, float))
    return float((np.nan_to_num(per) * w).sum() / w.sum())








