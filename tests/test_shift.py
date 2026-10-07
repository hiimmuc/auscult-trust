import numpy as np
from src.evaluation.shift import (
    coverage_deficit,
    per_class_coverage,
    prior_matched_coverage,
    set_stats,
)


def test_coverage_deficit():
    assert np.isclose(coverage_deficit(0.80, 0.1), 0.10)


def test_set_stats_and_per_class():
    sets = np.array([[1, 0], [1, 1], [0, 0], [0, 1]], bool)
    s = set_stats(sets)
    assert (s["size"], s["singleton"], s["empty"]) == (1.0, 0.5, 0.25)
    pc = per_class_coverage(sets, np.array([0, 0, 1, 1]), 2)
    assert pc[0]["coverage"] == 1.0 and pc[1]["coverage"] == 0.5 and pc[1]["n"] == 2


def test_prior_matched_coverage_reweights_classes():
    y = np.array([0, 0, 0, 0, 1, 1])
    sets = np.zeros((6, 2), bool)
    sets[:4, 0] = True  # class 0 fully covered, class 1 never
    assert np.isclose(prior_matched_coverage(sets, y, [0.5, 0.5]), 0.5)
    assert np.isclose(prior_matched_coverage(sets, y, [0.9, 0.1]), 0.9)
