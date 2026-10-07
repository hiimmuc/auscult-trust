import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
_p = Path(__file__).resolve().parents[1] / "baselines/patchmix_cl/util/stage1.py"
spec = importlib.util.spec_from_file_location("stage1", _p)
stage1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage1)


def test_patient_val_split_is_patient_disjoint_and_seed_stable():
    pats = np.repeat([f"p{i}" for i in range(20)], 5)
    tr, va = stage1.patient_val_split(pats, 0.2, seed=1)
    assert not set(pats[tr]) & set(pats[va]) and len(set(pats[va])) == 4 and len(tr) + len(va) == len(pats)
    assert np.array_equal(va, stage1.patient_val_split(pats, 0.2, seed=1)[1])
    assert not np.array_equal(va, stage1.patient_val_split(pats, 0.2, seed=2)[1])


def test_per_device_report_and_two_class_view():
    y = np.array([0, 0, 1, 2, 3, 0, 1, 1])
    p = np.array([0, 1, 1, 0, 3, 0, 1, 2])
    dev = np.array([0, 0, 0, 0, 3, 3, 3, 3])
    r = stage1.per_device_report(y, p, dev)
    assert set(r) == {"all", "Meditron", "AKGC417L"}
    assert np.isclose(r["all"]["sp"], 100 * 2 / 3) and np.isclose(r["all"]["se"], 100 * 3 / 5)
    assert np.isclose(r["all"]["two_cls_se"], 100 * 4 / 5)  # abnormal predicted as any abnormal class
    assert stage1.worst_device_score(r) == min(r["Meditron"]["score"], r["AKGC417L"]["score"])


def test_pick_epochs_three_numbers_differ_by_rule():
    def rep(s):
        return {"all": {"score": s}, "Meditron": {"score": s}}
    hist = [{"epoch": 1, "val": rep(50), "test": rep(60)}, {"epoch": 2, "val": rep(70), "test": rep(55)},
            {"epoch": 3, "val": rep(60), "test": rep(58)}]
    t = stage1.pick_epochs(hist)
    assert (t["last"]["epoch"], t["cv_selected"]["epoch"], t["test_best_optimistic"]["epoch"]) == (3, 2, 1)


def test_freq_mixstyle_keeps_shape_and_mixes_stats():
    torch.manual_seed(0)
    x = torch.randn(8, 1, 50, 16) * torch.linspace(1, 3, 8).view(8, 1, 1, 1) + torch.arange(8).view(8, 1, 1, 1)
    y = stage1.freq_mixstyle(x, p=1.0)
    assert y.shape == x.shape and not torch.allclose(x, y)
    assert torch.equal(stage1.freq_mixstyle(x, p=0.0), x)


def test_device_bin_norm_removes_per_device_bin_offset():
    rng = np.random.default_rng(0)
    base = [rng.normal(size=(30, 8, 1)) for _ in range(6)]
    off = rng.normal(0, 3, 8)
    imgs = base[:3] + [b + off[None, :, None] for b in base[3:]]
    out, _ = stage1.device_bin_norm(imgs, [0, 0, 0, 1, 1, 1])
    m0 = np.concatenate([o[..., 0] for o in out[:3]]).mean(0)
    m1 = np.concatenate([o[..., 0] for o in out[3:]]).mean(0)
    assert np.allclose(m0, m1, atol=1e-6)


def test_a1_coefficients_geometric_vs_arithmetic_and_invariant_dims():
    spec = {0: np.array([1.0, 4.0]), 1: np.array([4.0, 4.0])}
    assert np.allclose(stage1.a1_coefficients(spec, reference="arithmetic")[0], [2.5, 1.0])
    assert np.allclose(stage1.a1_coefficients(spec, reference="geometric")[0], [2.0, 1.0])
    assert len(stage1.HC_INVARIANT) == 20 and stage1.HC_INVARIANT.min() == 37 and stage1.HC_INVARIANT.max() == 56


def test_patient_folds_partition_stratified_and_seed_free():
    pats = np.repeat([f"p{i}" for i in range(30)], 4)
    devs = np.repeat([0] * 20 + [2] * 6 + [3] * 4, 4)
    a = stage1.patient_folds(pats, devs, 3)
    assert set(a) == set(pats) and set(a.values()) == {0, 1, 2}
    assert a == stage1.patient_folds(pats, devs, 3)  # independent of any run seed
    for d in (0, 2, 3):  # every device spread over the folds
        ids = {p for p, dd in zip(pats, devs) if dd == d}
        assert len({a[p] for p in ids}) == min(3, len(ids))
    seen = []
    for k in range(3):
        tr, va = stage1.cv_split(pats, devs, 3, k)
        assert not set(pats[tr]) & set(pats[va]) and len(tr) + len(va) == len(pats)
        seen.extend(va.tolist())
    assert sorted(seen) == list(range(len(pats)))  # folds cover every cycle exactly once


def test_pick_epochs_screening_and_fixed_modes_and_cv_epoch():
    rep = lambda s: {"all": {"score": s}, "Meditron": {"score": s}}
    screen = [{"epoch": e, "val": rep(v), "test": None} for e, v in [(1, 50), (2, 60), (3, 55)]]
    t = stage1.pick_epochs(screen)
    assert t["cv_selected"]["epoch"] == 2 and "test_best_optimistic" not in t
    fixed = [{"epoch": e, "val": None, "test": rep(v)} for e, v in [(1, 50), (2, 40)]]
    t = stage1.pick_epochs(fixed)
    assert t["test_best_optimistic"]["epoch"] == 1 and "cv_selected" not in t
    assert stage1.cv_epoch([[1, 3, 2], [1, 1, 5]]) == 3  # mean curve [1, 2, 3.5]
    assert stage1.cv_epoch([[1, 3, 3], [1, 3, 3, 9]]) == 2  # truncated to the shortest curve, earliest on ties


def _summary_module(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "baselines/patchmix_cl"
    monkeypatch.syspath_prepend(str(root))
    sp = importlib.util.spec_from_file_location("stage1_summary", root / "stage1_summary.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def test_summary_counts_match_metrics_and_expected_max(monkeypatch):
    m = _summary_module(monkeypatch)
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 300), rng.integers(0, 4, 300)
    pats = rng.integers(0, 20, 300).astype(str)
    c = m._score_counts(y, p, pats).sum(0)
    ref = stage1.metrics_from_preds(y, p)
    assert np.isclose(m._score_from_counts(c), ref["score"]) and np.isclose(m._score_from_counts(c, True), ref["two_cls_score"])
    assert abs(m.expected_max_normal(5) - 1.163) < 2e-3 and abs(m.expected_max_normal(10) - 1.539) < 2e-3


def test_summary_screen_and_final_end_to_end(tmp_path, monkeypatch, capsys):
    import json
    m = _summary_module(monkeypatch)
    save = tmp_path / "save"
    dev = lambda s: {"Meditron": s, "AKGC417L": s - 2}

    def write(tag, hist, extra=None):
        d = save / (m.PREFIX + tag)
        d.mkdir(parents=True)
        rep = {"history": hist, **(extra or {})}
        (d / "stage1_report.json").write_text(json.dumps(rep))
        return d

    curves = {"P0_baseline": [50, 58, 57], "P1_a1_input": [52, 60, 59], "P3_rand_bin_gain": [51, 55, 61]}
    for v, c in curves.items():
        for k in range(3):
            write(f"{v}_cv{k}", [{"epoch": e + 1, "val_score": s + k, "test_score": None, "val_worst": s - 5 + (e == 0) * 9}
                                 for e, s in enumerate(c)])
    m.screen(type("A", (), {"save": str(save), "folds": 3})())
    scr = json.loads((save / "stage1_screen.json").read_text())
    assert scr["epochs"] == {"P1_a1_input": 2, "P3_rand_bin_gain": 3, "P0_baseline": 2, "P8_worst_device": 1}
    assert scr["ranking"][0] == "P3_rand_bin_gain" and scr["finalists"][-1] == "P0_baseline"

    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 400)
    pats = np.repeat(np.arange(40), 10).astype(str)
    devices = np.where(np.arange(400) < 200, 0, 3)
    for v, ep, acc in [("P0_baseline", [2, 1], 0.5), ("P1_a1_input", [2], 0.8)]:
        for s in range(5):
            preds = np.stack([np.where(rng.random(400) < acc, y, rng.integers(0, 4, 400)) for _ in range(3)]).astype(np.int8)
            fixed = {str(e): {"epoch": e, "test": stage1.per_device_report(y, preds[e - 1], devices)} for e in ep}
            best = {"epoch": 3, "test": stage1.per_device_report(y, preds[2], devices)}
            d = write(f"{v}_fix_seed{s}", [{"epoch": e, "test_score": stage1.per_device_report(y, preds[e - 1], devices)["all"]["score"]}
                                          for e in (1, 2, 3)], {"fixed": fixed, "test_best_optimistic": best})
            probs = np.eye(4)[preds].astype(np.float16)
            np.savez(d / "test_preds.npz", epochs=np.array([1, 2, 3]), preds=preds, probs=probs, labels=y, device=devices, patient=pats)
    m.final(type("A", (), {"save": str(save), "boot": 200})())
    out = (save / "stage1_final.md").read_text()
    assert "P8_worst_device" in out and "G-base: Stage 2 model = **P1_a1_input**" in out
    assert "P1_a1_input (ensemble of 5 seeds)" in out
