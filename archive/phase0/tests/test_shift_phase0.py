import numpy as np

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


def test_closed_fraction_checked_nan_inside_binomial_halfwidth():
    from src.eval.shift import closed_fraction_checked
    assert np.isnan(closed_fraction_checked(0.89, 0.9, 0.1, 100))  # deficit 0.01 < half-width 0.059
    assert np.isclose(closed_fraction_checked(0.70, 0.85, 0.1, 1000), 1 - 0.05 / 0.20)


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
