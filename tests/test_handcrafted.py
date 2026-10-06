import numpy as np

from src.features_handcrafted import N_DIM, handcrafted, handcrafted_table


def test_handcrafted_is_74_dim_finite_and_responds_to_gain():
    rng = np.random.default_rng(0)
    w = rng.normal(size=16000 * 2).astype(np.float32)
    v = handcrafted(w)
    assert v.shape == (N_DIM,) and np.isfinite(v).all()
    assert handcrafted_table([w, w * 3]).shape == (2, N_DIM)
    assert not np.allclose(v, handcrafted(w * 3))
