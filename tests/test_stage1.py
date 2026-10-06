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
