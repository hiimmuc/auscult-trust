import numpy as np

from src.eval.shift import (closed_fraction, coverage_deficit, decodability, per_class_coverage,
                            permutation_pvalue, set_stats, spearman_block_bootstrap)


def test_deficit_and_closed_fraction():
    assert np.isclose(coverage_deficit(0.80, 0.1), 0.10)
    assert np.isclose(closed_fraction(0.10, 0.04), 0.6)
    assert np.isnan(closed_fraction(0.01, 0.0, min_delta=0.02))


def test_set_stats_and_per_class():
    sets = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], bool)
    s = set_stats(sets)
    assert (s["size"], s["singleton"], s["empty"]) == (1.0, 0.5, 0.25)
    pc = per_class_coverage(sets, np.array([0, 0, 1, 1]), 2)
    assert pc[0]["coverage"] == 1.0 and pc[1]["coverage"] == 0.5 and pc[1]["n"] == 2


def test_decodability_detects_shift_and_chance():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 8)), rng.normal(size=(200, 8))
    ga, gb = np.repeat(np.arange(20), 10), np.repeat(np.arange(20, 40), 10)
    assert decodability(a, b, ga, gb) < 0.65
    assert decodability(a, b + 1.5, ga, gb) > 0.95
    auc, p = permutation_pvalue(a, b + 1.5, ga, gb, n_perm=20)
    assert auc > 0.95 and p < 0.1


def test_spearman_block_bootstrap_positive():
    x = np.arange(12.0)
    rho, lo, hi = spearman_block_bootstrap(x, x + np.random.default_rng(0).normal(0, 0.5, 12), np.repeat([0, 1, 2, 3], 3))
    assert rho > 0.9 and lo > 0


def test_conformal_layer_coverage_exchangeable():
    from src.auscult_trust.conformal import ConformalLayer
    rng = np.random.default_rng(0)
    def draw(n):
        y = rng.integers(0, 4, n)
        logit = rng.normal(size=(n, 4)) + 2.0 * np.eye(4)[y]
        p = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
        return p, y
    pc, yc = draw(3000)
    pt, yt = draw(3000)
    for v in ("split", "mondrian"):
        r = ConformalLayer(0.1, v).calibrate(pc, yc).evaluate(pt, yt)
        assert abs(r["coverage"] - 0.9) < 0.03, (v, r["coverage"])
        assert all(c["coverage"] > 0.8 for c in r["per_class"]) if v == "mondrian" else True


def test_closed_fraction_checked_nan_inside_binomial_halfwidth():
    from src.eval.shift import closed_fraction_checked
    assert np.isnan(closed_fraction_checked(0.89, 0.9, 0.1, 100))  # deficit 0.01 < half-width 0.059
    assert np.isclose(closed_fraction_checked(0.70, 0.85, 0.1, 1000), 1 - 0.05 / 0.20)


def test_prior_matched_coverage_reweights_classes():
    from src.eval.shift import prior_matched_coverage
    y = np.array([0, 0, 0, 0, 1, 1])
    sets = np.zeros((6, 2), bool)
    sets[:4, 0] = True  # class 0 fully covered, class 1 never
    assert np.isclose(prior_matched_coverage(sets, y, [0.5, 0.5]), 0.5)
    assert np.isclose(prior_matched_coverage(sets, y, [0.9, 0.1]), 0.9)


def test_decodability_curve_and_mmd_detect_shift_and_null():
    from src.eval.shift import decodability_curve, mmd_permutation
    rng = np.random.default_rng(0)
    gs, gt = np.repeat(np.arange(20), 10), np.repeat(np.arange(20, 40), 10)
    a, b = rng.normal(size=(200, 16)), rng.normal(size=(200, 16))
    shifted = b + 2.0
    cur = decodability_curve(a, shifted, gs, gt, ns=(20, 100))
    assert cur[100] > 0.9
    assert decodability_curve(a, b, gs, gt, ns=(20, 100))[100] < 0.7
    _, p_shift = mmd_permutation(a, shifted, gs, gt, n_perm=50)
    _, p_null = mmd_permutation(a, b, gs, gt, n_perm=50)
    assert p_shift < 0.1 < p_null
