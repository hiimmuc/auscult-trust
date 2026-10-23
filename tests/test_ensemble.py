import sys

import numpy as np


def test_fit_weights_prefers_the_informative_member():
    sys.path.insert(0, "scripts")
    from ensemble import fit_weights

    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    good = np.full((500, 4), 0.05)
    good[np.arange(500), y] = 0.85
    noise = rng.dirichlet(np.ones(4), 500)
    w = fit_weights(np.stack([noise, good]), y)
    assert abs(w.sum() - 1) < 1e-9 and w[1] > 0.9


def test_k_for_sp_reaches_the_target_specificity():
    sys.path.insert(0, "scripts")
    from ensemble_sesp import k_for_sp, predict

    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 2000)
    p = rng.dirichlet(np.ones(4), 2000)
    p[np.arange(2000), y] += 0.3
    p /= p.sum(1, keepdims=True)
    k = k_for_sp(p, y, 80)
    assert abs((predict(p, k)[y == 0] == 0).mean() - 0.80) < 0.01
    assert (predict(p) == p.argmax(1)).all()


def test_k_for_rate_calls_that_fraction_normal():
    sys.path.insert(0, "scripts")
    from ensemble_sesp import k_for_rate, predict

    p = np.random.default_rng(2).dirichlet(np.ones(4), 1000)
    assert abs((predict(p, k_for_rate(p, 0.6)) == 0).mean() - 0.6) < 0.01


def test_frozen_ensemble_rejects_shuffled_patients(tmp_path):
    import pytest
    sys.path.insert(0, "scripts")
    from final_ensembles import mean_exports

    p = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]])
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(a, probs=p, patient=np.array(["a", "b"]))
    np.savez(b, probs=p[::-1], patient=np.array(["b", "a"]))
    out = tmp_path / "ensemble.npz"
    with pytest.raises(AssertionError):
        mean_exports([a, b], out, {"a": "hash-a", "b": "hash-b"})
    assert not out.exists()
