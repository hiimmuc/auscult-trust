import numpy as np
import pandas as pd
import pytest

from src.conformal.conformal import (bbse_weights, label_shift_threshold, lac_scores, mondrian_threshold,
                                     predict_sets, split_threshold)
from src.data.icbhi import parse_name
from src.data.splits import assert_disjoint, device_holdout, load_split, save_split, split_patients
from src.eval.aggregate import patient_features
from src.eval.metrics import coverage, icbhi_score, patient_bootstrap


def test_parse_name():
    assert parse_name("101_1b1_Al_sc_Meditron") == {"patient": "icbhi_101", "site": "Al", "device": "Meditron"}


def test_splits_disjoint_and_device_holdout(tmp_path):
    s = split_patients([f"p{i}" for i in range(50)] * 3, seed=1)
    assert_disjoint(s)
    df = pd.DataFrame({"patient": ["a", "a", "b", "c", "d", "e"],
                       "device": ["X", "Y", "Y", "X", "Z", "Z"]})
    h = device_holdout(df, "Y", seed=0)
    assert h["test"] == ["a", "b"]
    assert "a" not in h["train"] + h["val"]  # patient on held-out device removed everywhere
    h_ = save_split(s, tmp_path / "s.json")
    assert load_split(tmp_path / "s.json")[1] == h_
    with pytest.raises(FileExistsError):
        save_split(s, tmp_path / "s.json")


def _sim(n, rng, shift=0.0):
    y = rng.integers(0, 4, n)
    logits = rng.normal(size=(n, 4)) + 2.0 * np.eye(4)[y] * (1 - shift)
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return p, y


def test_split_and_mondrian_coverage():
    rng = np.random.default_rng(0)
    pc, yc = _sim(5000, rng)
    pt, yt = _sim(5000, rng)
    sc = lac_scores(pc, yc)
    cov = coverage(predict_sets(pt, split_threshold(sc, 0.1)), yt)
    assert abs(cov - 0.9) < 0.03
    sets = predict_sets(pt, mondrian_threshold(sc, yc, 0.1, 4))
    for k in range(4):
        assert abs(sets[yt == k, k].mean() - 0.9) < 0.04


def test_label_shift_restores_coverage():
    rng = np.random.default_rng(1)
    pc, yc = _sim(20000, rng)
    pt, yt = _sim(20000, rng)
    keep = rng.random(len(yt)) < np.array([0.8, 0.1, 0.05, 0.05])[yt]  # target prior shifts to class 0
    pt, yt = pt[keep], yt[keep]
    w = bbse_weights(pc.argmax(1), yc, pt.argmax(1), 4)
    sc = lac_scores(pc, yc)
    cov_split = coverage(predict_sets(pt, split_threshold(sc, 0.1)), yt)
    cov_ls = coverage(predict_sets(pt, label_shift_threshold(sc, yc, w, 0.1)), yt)
    assert abs(cov_ls - 0.9) <= abs(cov_split - 0.9) + 0.01
    assert abs(cov_ls - 0.9) < 0.03


def test_icbhi_score():
    y = np.array([0, 0, 1, 2])
    assert icbhi_score(y, np.array([0, 1, 1, 1])) == (0.5 + 0.5) / 2


def test_patient_features_shape():
    df = pd.DataFrame({"patient": ["a", "a", "b"], "phase": ["insp", "exp", "insp"],
                       "p_crackle": [0.9, 0.1, 0.4], "p_wheeze": [0.2, 0.3, 0.5]})
    f = patient_features(df)
    assert f.shape == (2, 8)
    assert f.loc["a", "insp_crackle_max"] == 0.9


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    pat = np.repeat(np.arange(40), 5)
    y = rng.integers(0, 2, 200)
    est, lo, hi = patient_bootstrap(lambda y, pred: (y == pred).mean(), pat, n_boot=200, y=y, pred=y.copy())
    assert est == 1.0 and lo <= est <= hi
