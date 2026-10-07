import json

import numpy as np
import pytest

from src.processing.splits import cv_split, patient_folds, patient_val_split
from src.training.patchmix_cl import summary
from src.training.patchmix_cl.reporting import cv_epoch, per_device_report, pick_epochs


def test_patient_val_split_is_patient_disjoint_and_seed_stable():
    pats = np.repeat([f"p{i}" for i in range(20)], 5)
    tr, va = patient_val_split(pats, 0.2, seed=1)
    assert not set(pats[tr]) & set(pats[va]) and len(set(pats[va])) == 4 and len(tr) + len(va) == len(pats)
    assert np.array_equal(va, patient_val_split(pats, 0.2, seed=1)[1])
    assert not np.array_equal(va, patient_val_split(pats, 0.2, seed=2)[1])


def test_per_device_report_and_two_class_view():
    y = np.array([0, 0, 1, 2, 3, 0, 1, 1])
    p = np.array([0, 1, 1, 0, 3, 0, 1, 2])
    dev = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    r = per_device_report(y, p, dev)
    assert set(r) == {"all", "Meditron", "AKGC417L"}
    assert np.isclose(r["all"]["sp"], 100 * 2 / 3) and np.isclose(r["all"]["se"], 100 * 3 / 5)
    assert np.isclose(r["all"]["two_cls_se"], 100 * 4 / 5)  # abnormal predicted as any abnormal class
    assert 0 <= r["all"]["macro_f1"] <= 1  # a fraction, not a percent


def test_pick_epochs_screening_and_fixed_modes_and_cv_epoch():
    rep = lambda s: {"all": {"score": s}, "Meditron": {"score": s}}
    hist = [{"epoch": 1, "val": rep(50), "test": rep(60)}, {"epoch": 2, "val": rep(70), "test": rep(55)},
            {"epoch": 3, "val": rep(60), "test": rep(58)}]
    t = pick_epochs(hist)
    assert (t["last"]["epoch"], t["cv_selected"]["epoch"], t["test_best_optimistic"]["epoch"]) == (3, 2, 1)
    screen = [{"epoch": e, "val": rep(v), "test": None} for e, v in [(1, 50), (2, 60), (3, 55)]]
    t = pick_epochs(screen)
    assert t["cv_selected"]["epoch"] == 2 and "test_best_optimistic" not in t
    fixed = [{"epoch": e, "val": None, "test": rep(v)} for e, v in [(1, 50), (2, 40)]]
    t = pick_epochs(fixed)
    assert t["test_best_optimistic"]["epoch"] == 1 and "cv_selected" not in t
    assert cv_epoch([[1, 3, 2], [1, 1, 5]]) == 3  # mean curve [1, 2, 3.5]
    assert cv_epoch([[1, 3, 3], [1, 3, 3, 9]]) == 2  # truncated to the shortest curve, earliest on ties


def test_freq_mixstyle_keeps_shape_and_mixes_stats():
    torch = pytest.importorskip("torch")
    from src.training.patchmix_cl.augment import freq_mixstyle
    torch.manual_seed(0)
    x = torch.randn(8, 1, 50, 16) * torch.linspace(1, 3, 8).view(8, 1, 1, 1) + torch.arange(8).view(8, 1, 1, 1)
    y = freq_mixstyle(x, p=1.0)
    assert y.shape == x.shape and not torch.allclose(x, y)
    assert torch.equal(freq_mixstyle(x, p=0.0), x)


def test_patient_folds_partition_stratified_and_seed_free():
    pats = np.repeat([f"p{i}" for i in range(30)], 4)
    devs = np.repeat([0] * 20 + [2] * 6 + [3] * 4, 4)
    a = patient_folds(pats, devs, 3)
    assert set(a) == set(pats) and set(a.values()) == {0, 1, 2}
    assert a == patient_folds(pats, devs, 3)  # independent of any run seed
    for d in (0, 2, 3):  # every device spread over the folds
        ids = {p for p, dd in zip(pats, devs) if dd == d}
        assert len({a[p] for p in ids}) == min(3, len(ids))
    seen = []
    for k in range(3):
        tr, va = cv_split(pats, devs, 3, k)
        assert not set(pats[tr]) & set(pats[va]) and len(tr) + len(va) == len(pats)
        seen.extend(va.tolist())
    assert sorted(seen) == list(range(len(pats)))  # folds cover every cycle exactly once


def test_summary_counts_match_metrics():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    c = summary._score_counts(y, p, rng.integers(0, 20, 300).astype(str)).sum(0)
    ref = per_device_report(y, p, np.zeros(300, int))["all"]
    assert np.isclose(summary._score_from_counts(c), ref["score"]) and np.isclose(summary._score_from_counts(c, True), ref["two_cls_score"])


def test_summary_screen_and_final_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(summary, "OUTPUTS", tmp_path)
    root = tmp_path / "stage1" / "run1"

    def write(cell, unit, report, preds=None):
        d = root / cell / unit
        d.mkdir(parents=True)
        (d / "stage1_report.json").write_text(json.dumps(report))
        if preds:
            np.savez(d / "test_preds.npz", **preds)
        return d

    curves = {"P0_baseline": [50, 58, 57], "P1_a1_input": [52, 60, 59], "P1_a1_input+sc_static": [52, 59, 58],
              "P1P3_sc_gain": [51, 55, 61], "P1P3_sc_gain+sc_static": [51, 56, 62], "P4_freq_mixstyle": [50, 57, 56]}
    for cell, c in curves.items():
        for k in range(3):
            write(cell, f"cv{k}", {"history": [{"epoch": e + 1, "val_score": s + k, "test_score": None} for e, s in enumerate(c)]})
    summary.screen(type("A", (), {"run": "run1", "folds": 3})())
    scr = json.loads((root / "stage1_screen.json").read_text())
    assert scr["arms"]["P1_a1_input"]["sc_mode"] == "dynamic" and scr["arms"]["P1_a1_input"]["epoch"] == 2  # static is lower
    assert scr["arms"]["P1P3_sc_gain"]["sc_mode"] == "static" and scr["arms"]["P1P3_sc_gain"]["cell"] == "P1P3_sc_gain+sc_static"
    assert scr["arms"]["P0_baseline"]["sc_mode"] is None and scr["ranking"][0] == "P1P3_sc_gain"
    assert scr["auscult_trust"] == "P1P3_sc_gain"  # 62 + 1 > 60 + 1

    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 400)
    pats = np.repeat(np.arange(40), 10).astype(str)
    devices = np.where(np.arange(400) < 200, 0, 3)
    plan = [("P0_baseline", 2, 0.5, 10), ("P1P3_sc_gain+sc_static", 3, 0.8, 10), ("P1_a1_input", 2, 0.6, 5), ("P4_freq_mixstyle", 2, 0.5, 5)]
    for cell, ep, acc, n_seeds in plan:
        for s in range(n_seeds):
            preds = np.stack([np.where(rng.random(400) < acc, y, rng.integers(0, 4, 400)) for _ in range(3)]).astype(np.int8)
            rep = lambda e: per_device_report(y, preds[e - 1], devices)
            write(cell, f"seed{s}", {"fixed": {str(ep): {"epoch": ep, "test": rep(ep)}}, "test_best_optimistic": {"epoch": 3, "test": rep(3)}},
                  {"epochs": np.array([1, 2, 3]), "preds": preds, "labels": y, "device": devices, "patient": pats})
    scr["arms"]["P4_freq_mixstyle"]["epoch"] = 2
    (root / "stage1_screen.json").write_text(json.dumps(scr))
    summary.final(type("A", (), {"run": "run1", "boot": 200})())
    out = (root / "stage1_final.md").read_text()
    assert "P1P3_sc_gain (AuscultTrust)" in out and "Gate G (P1P3_sc_gain vs P0, 10 seeds)" in out and "**pass**" in out
