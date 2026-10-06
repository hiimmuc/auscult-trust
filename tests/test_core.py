import numpy as np
import pytest

from src.conformal.conformal import (bbse_weights, label_shift_threshold, lac_scores, mondrian_threshold,
                                     predict_sets, split_threshold)
from src.data.icbhi import parse_name
from src.data.splits import assert_disjoint, load_split, save_split, split_patients
from src.eval.metrics import coverage, icbhi_score, patient_bootstrap


def test_parse_name():
    assert parse_name("101_1b1_Al_sc_Meditron") == {"patient": "icbhi_101", "site": "Al", "device": "Meditron"}


def test_splits_disjoint_and_versioned(tmp_path):
    s = split_patients([f"p{i}" for i in range(50)] * 3, seed=1)
    assert_disjoint(s)
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


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    pat = np.repeat(np.arange(40), 5)
    y = rng.integers(0, 2, 200)
    est, lo, hi = patient_bootstrap(lambda y, pred: (y == pred).mean(), pat, n_boot=200, y=y, pred=y.copy())
    assert est == 1.0 and lo <= est <= hi


def test_rq3_variants_cover_when_exchangeable():
    from src.legacy.rq3 import set_metrics, thresholds
    rng = np.random.default_rng(2)
    pv, yv = _sim(4000, rng)
    pk, yk = _sim(4000, rng)
    pe, ye = _sim(4000, rng)
    thr = thresholds(pv, yv, pk, yk, pe, np.ones(4000), np.ones(4000), 0.1)
    for v, t in thr.items():
        assert abs(set_metrics(predict_sets(pe, t), ye)["coverage"] - 0.9) < 0.03, v


def test_device_holdout_official_disjoint_and_seen_flag():
    import pandas as pd
    from src.data.splits import device_holdout_official, patient_device_crosstab
    rows = [("a1", "A", "train"), ("a2", "A", "test"), ("b1", "B", "train"), ("b2", "B", "test"), ("c1", "B", "train"), ("c1", "A", "train")]
    df = pd.DataFrame(rows, columns=["patient", "device", "official"]).assign(stem=lambda d: d.index.astype(str))
    ct = patient_device_crosstab(df)
    assert (ct.loc["c1"] > 0).all()  # c1 recorded on both devices
    r = device_holdout_official(df, "A")
    assert r["n_dropped"] == 1 and set(r["test_unseen"].patient) == {"a2"} and set(r["test_seen"].patient) == {"a1", "c1"}
    assert set(r["train"].patient) == {"b1"} and set(r["cal"].patient) == {"b2"}


def test_calibration_eval_splits_disjoint_and_stratified_by_device():
    import pandas as pd
    from src.data.splits import calibration_eval_splits
    # 4 patients on device A, 1 on device B (rare-device edge case: must not crash or drop B).
    df = pd.DataFrame({
        "patient": ["a1", "a2", "a3", "a4", "b1"],
        "device": ["A", "A", "A", "A", "B"],
    })
    splits = calibration_eval_splits(df, n_splits=20, fracs=(0.5, 0.5), seed=0)
    assert len(splits) == 20
    for s in splits:
        cal, ev = set(s["calibration"]), set(s["evaluation"])
        assert not cal & ev
        assert cal | ev == {"a1", "a2", "a3", "a4", "b1"}
        assert len(cal) >= 1 and len(ev) >= 1  # device B's single patient must land somewhere every time
    # Different seeds across the 20 splits actually move patients around (not 20 copies of the same split).
    assert len({frozenset(s["calibration"]) for s in splits}) > 1


def test_calibration_eval_splits_odd_device_size_is_not_systematically_biased():
    import pandas as pd
    from src.data.splits import calibration_eval_splits
    df = pd.DataFrame({"patient": [f"p{i}" for i in range(3)], "device": ["A"] * 3})
    splits = calibration_eval_splits(df, n_splits=20, fracs=(0.5, 0.5), seed=0)
    cal_sizes = {len(s["calibration"]) for s in splits}
    assert cal_sizes == {1, 2}, f"3-patient device should split both 1/2 and 2/1 across resplits, got sizes {cal_sizes}"


def test_sp_se_hs_macro_f1_by_device():
    from src.eval.metrics import by_device, hs, icbhi_score, macro_f1, sp_se
    y = np.array([0, 0, 1, 2, 3, 0, 1, 1])
    p = np.array([0, 1, 1, 0, 3, 0, 1, 2])
    sp, se = sp_se(y, p)
    assert np.isclose(sp, 2 / 3) and np.isclose(se, 3 / 5)
    assert np.isclose(hs(y, p), 2 * sp * se / (sp + se)) and np.isclose((sp + se) / 2, icbhi_score(y, p))
    assert 0 < macro_f1(y, p) < 1
    r = by_device(y, p, np.array(["A"] * 4 + ["B"] * 4))
    assert set(r) == {"A", "B", "all"} and r["all"]["n"] == 8 and np.isclose(r["all"]["sp"], sp)


def test_v4_oracle_and_v5_patient_k():
    from src.conformal.conformal import (kshot_patients_threshold, label_shift_threshold,
                                         oracle_label_shift_threshold)
    rng = np.random.default_rng(0)
    scores, labels = rng.uniform(size=400), rng.integers(0, 2, 400)
    f = np.bincount(labels) / 400
    same = oracle_label_shift_threshold(scores, labels, f, f, 0.1)
    assert np.allclose(same, label_shift_threshold(scores, labels, np.ones(2), 0.1))
    probs = rng.dirichlet(np.ones(3), 300)
    pats, y = np.repeat(np.arange(30), 10), rng.integers(0, 3, 300)
    thr, mask = kshot_patients_threshold(probs, y, pats, k=5, alpha=0.1)
    assert len(np.unique(pats[mask])) == 5 and mask.sum() == 50 and np.isfinite(thr)
