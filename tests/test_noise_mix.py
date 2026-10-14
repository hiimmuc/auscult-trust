import numpy as np
import torch

from src.training.patchmix_cl.augment import noise_mix_image
from src.training.patchmix_cl.cycles import DB_PER_NAT, FBANK_STD, generate_fbank

DB_SCALE = 1 / (2 * FBANK_STD * DB_PER_NAT)


def _img(w):
    return generate_fbank(torch.from_numpy(w.astype(np.float32))[None], 16000)


def test_noise_mix_matches_waveform_mix():
    """Mixing in the mel power domain equals mixing the waveforms (independent signals), and lands on the requested SNR."""
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 8) / 16000
    sig = 0.1 * np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t))
    noise = rng.standard_normal(len(t)) * 0.05
    snr = 10.0
    # roll 0: rng.integers(n) with n frames, so pass a generator that always returns 0
    class Zero:
        def integers(self, n):
            return 0
    mixed = noise_mix_image(_img(sig), _img(noise), snr, DB_SCALE, Zero())
    k = DB_SCALE * DB_PER_NAT
    p = lambda im: np.mean(np.exp(im.astype(np.float64) / k))
    g = p(_img(sig)) / (p(_img(noise)) * 10 ** (snr / 10))
    ref = _img(sig + noise * np.sqrt(g))
    assert np.median(np.abs(mixed - ref)) < 0.05  # cross terms make single bins differ, the bulk agrees
    assert abs(10 * np.log10(p(_img(sig)) / (p(mixed) - p(_img(sig)))) - snr) < 0.2


def test_noise_mix_high_snr_is_identity():
    rng = np.random.default_rng(1)
    img = _img(rng.standard_normal(16000 * 8) * 0.1)
    out = noise_mix_image(img, _img(rng.standard_normal(16000 * 8)), 80.0, DB_SCALE, rng)
    assert np.abs(out - img).max() < 1e-3
