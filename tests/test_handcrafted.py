import numpy as np

from src.features_handcrafted import N_DIM, handcrafted, handcrafted_table


def test_handcrafted_is_74_dim_finite_and_responds_to_gain():
    rng = np.random.default_rng(0)
    w = rng.normal(size=16000 * 2).astype(np.float32)
    v = handcrafted(w)
    assert v.shape == (N_DIM,) and np.isfinite(v).all()
    assert handcrafted_table([w, w * 3]).shape == (2, N_DIM)
    assert not np.allclose(v, handcrafted(w * 3))


def test_mfcc_sd_is_more_gain_invariant_than_mfcc_mean():
    from src.shift.correction import apply_spectrum_correction
    rng = np.random.default_rng(0)
    w = (rng.normal(size=16000 * 4) * np.linspace(0.2, 2, 16000 * 4)).astype(np.float32)
    coef = 10 ** ((6 * np.sin(np.linspace(0, 4, 513))) / 20)
    a, b = handcrafted(w), handcrafted(apply_spectrum_correction(w, coef).astype(np.float32))
    mean_shift = np.abs(a[1:20] - b[1:20]).mean()
    sd_shift = np.abs(a[37 + 1:57] - b[37 + 1:57]).mean()
    assert sd_shift < 0.3 * mean_shift
