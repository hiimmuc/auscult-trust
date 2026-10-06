import numpy as np

from src.shift.correction import (apply_spectrum_correction, isa_match, logmel_stats, mean_spectrum, random_bin_gain,
                                  spectrum_coefficients)


def _device(waves, g):
    """Apply a fixed per-bin gain g to each wave via the STFT."""
    import librosa
    return [librosa.istft(librosa.stft(w, n_fft=1024, hop_length=512) * g[:, None], hop_length=512, length=len(w)) for w in waves]


def test_spectrum_correction_aligns_devices():
    rng = np.random.default_rng(0)
    waves = [rng.normal(size=16000 * 2) for _ in range(6)]
    g = 1 + 0.8 * np.sin(np.linspace(0, 3, 513))
    spec = {"A": mean_spectrum(waves), "B": mean_spectrum(_device(waves, g))}
    coef = spectrum_coefficients(spec, source_devices=["A"])
    fixed = [apply_spectrum_correction(w, coef["B"]) for w in _device(waves, g)]
    before = np.abs(np.log(spec["B"] / spec["A"])).mean()
    after = np.abs(np.log(mean_spectrum(fixed) / spec["A"])).mean()
    assert after < 0.1 * before


def test_isa_removes_per_bin_offset():
    rng = np.random.default_rng(0)
    src = [rng.normal(size=(50, 16)) * 3 + 10 for _ in range(8)]
    offset = rng.normal(0, 5, 16)
    tgt = [x + offset for x in src]
    out = [isa_match(x, logmel_stats(tgt), logmel_stats(src)) for x in tgt]
    assert np.allclose(np.concatenate(out), np.concatenate(src), atol=1e-6)


def test_random_bin_gain_is_smooth_per_clip_offset():
    x = np.zeros((10, 64))
    y = random_bin_gain(x, 6.0, rng=np.random.default_rng(1))
    assert np.allclose(y, y[0]) and np.abs(np.diff(y[0])).max() < 6.0


def test_random_stft_gain_changes_spectrum_keeps_rms():
    from src.shift.correction import random_stft_gain
    w = np.random.default_rng(0).normal(size=32000).astype(np.float32)
    y = random_stft_gain(w, 6.0, rng=np.random.default_rng(1))
    assert y.shape == w.shape and np.isclose(np.sqrt(np.mean(y ** 2)), np.sqrt(np.mean(w ** 2)), rtol=1e-3)
    assert np.abs(np.log(mean_spectrum([y]) / mean_spectrum([w]))).mean() > 0.2


def test_spectrum_reference_arithmetic_vs_geometric():
    spec = {"A": np.array([1.0, 4.0]), "B": np.array([4.0, 4.0])}
    ar = spectrum_coefficients(spec, reference="arithmetic")["A"]
    ge = spectrum_coefficients(spec, reference="geometric")["A"]
    assert np.allclose(ar, [2.5, 1.0]) and np.allclose(ge, [2.0, 1.0])
