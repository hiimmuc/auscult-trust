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
