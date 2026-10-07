import numpy as np

from src.processing.correction import (apply_spectrum_correction, isa_match, limit_coefficients, logmel_stats,
                                  mean_spectrum, random_bin_gain, spectrum_coefficients)


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


def test_spectrum_reference_arithmetic_vs_geometric():
    spec = {"A": np.array([1.0, 4.0]), "B": np.array([4.0, 4.0])}
    ar = spectrum_coefficients(spec, reference="arithmetic")["A"]
    ge = spectrum_coefficients(spec, reference="geometric")["A"]
    assert np.allclose(ar, [2.5, 1.0]) and np.allclose(ge, [2.0, 1.0])


def test_dynamic_sc_mode_clips_a_4khz_device_against_a_16khz_reference():
    # A 4 kHz-band device has ~0 energy above 2 kHz; the raw ratio there blows up (the +-65 dB fact in CLAUDE.md).
    n_fft, sr = 1024, 16000
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    s_ref = np.full_like(freqs, 1.0)
    s_dev = np.where(freqs <= 2000, 1.0, 1e-4)  # no energy above 2 kHz
    raw = s_ref / np.maximum(s_dev, 1e-8)
    clipped = limit_coefficients(raw, sc_mode="dynamic", limit_freq_diff=20.0)
    assert np.all(20 * np.log10(clipped) <= 20.0 + 1e-6)
    assert 20 * np.log10(raw[freqs > 2000]).max() > 20.0  # confirms the clip actually bound something


def test_static_sc_mode_is_a_no_op_outside_the_band():
    n_fft, sr = 1024, 16000
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    raw = np.full_like(freqs, 3.0)
    out = limit_coefficients(raw, sc_mode="static", limit_freq_low=50.0, limit_freq_high=2000.0, sr=sr, n_fft=n_fft)
    band = (freqs >= 50) & (freqs <= 2000)
    assert np.allclose(out[band], 3.0) and np.allclose(out[~band], 1.0)
