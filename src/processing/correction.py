"""Gain-based device corrections.

A1  spectrum correction (SC) on STFT magnitude, per device, needs the device id [Kosmider 2020; Nguyen & Pernkopf 2022]
A2  ISA: per-mel-bin moment matching of log-mel on unlabelled target clips
P3  random smooth per-mel-bin gain in dB, a train-time augmentation and the TTA-EQ view generator
"""
import numpy as np


def _stft(wave, n_fft, hop):
    import librosa
    return librosa.stft(wave, n_fft=n_fft, hop_length=hop)


def mean_spectrum(waves, n_fft=1024, hop=512):
    """Mean |STFT| per frequency bin over all clips and frames of one device.

    Args:
        waves: Iterable of 1-D float waveforms.
        n_fft: FFT size (OPERA: 1024 at 16 kHz).
        hop: Hop size.

    Returns:
        (n_fft // 2 + 1,) array.
    """
    return np.mean([np.abs(_stft(w, n_fft, hop)).mean(1) for w in waves], axis=0)


def limit_coefficients(coef, sc_mode="dynamic", limit_freq_low=50.0, limit_freq_high=2000.0,
                        limit_freq_diff=20.0, sr=16000, n_fft=1024):
    """Bound a raw A1 coefficient curve; two variants are compared by CV score.

    Args:
        coef: (bins,) raw coefficient, one STFT bin per entry (`np.fft.rfftfreq(n_fft, 1 / sr)` spacing).
        sc_mode: "dynamic" clips every bin's gain to +-`limit_freq_diff` dB (bounds the coefficient everywhere,
            including bins with near-zero reference or device energy). "static" instead zeroes the correction
            (coefficient = 1, no-op) outside [`limit_freq_low`, `limit_freq_high`] Hz and leaves bins inside the
            band unclipped.
        limit_freq_low, limit_freq_high: Band edges in Hz, used only by "static".
        limit_freq_diff: Clip in dB, used only by "dynamic".
        sr, n_fft: Needed to map bins to Hz for "static".

    Returns:
        (bins,) bounded coefficient.
    """
    assert sc_mode in ("dynamic", "static"), sc_mode
    if sc_mode == "dynamic":
        return 10 ** (np.clip(20 * np.log10(np.maximum(coef, 1e-8)), -limit_freq_diff, limit_freq_diff) / 20)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    band = (freqs >= limit_freq_low) & (freqs <= limit_freq_high)
    return np.where(band, coef, 1.0)


def reference_spectrum(spectra, reference="arithmetic"):
    """Reference spectrum s_ref: "arithmetic" mean of device spectra (Nguyen & Pernkopf) or "geometric" mean (Kosmider)."""
    assert reference in ("arithmetic", "geometric"), reference
    stack = np.array(list(spectra.values()))
    return stack.mean(0) if reference == "arithmetic" else np.exp(np.log(np.maximum(stack, 1e-8)).mean(0))


def spectrum_coefficients(device_spectra, source_devices=None, reference="arithmetic", sc_mode="dynamic",
                           limit_freq_low=50.0, limit_freq_high=2000.0, limit_freq_diff=20.0, sr=16000, n_fft=1024,
                           reference_spectra=None):
    """A1 coefficients c_k = s_ref / s_k per bin, with s_ref the mean of the source devices' spectra.

    Args:
        device_spectra: Dict device -> mean spectrum from `mean_spectrum`.
        source_devices: Devices that define the reference. Default: all.
        reference_spectra: Dict device -> spectrum defining the reference instead (target devices against the
            training reference). Overrides `source_devices`.
        reference: "arithmetic" mean of device spectra (Nguyen & Pernkopf) or "geometric" mean (Kosmider).
        sc_mode, limit_freq_low, limit_freq_high, limit_freq_diff, sr, n_fft: see `limit_coefficients`.

    Returns:
        Dict device -> (bins,) coefficients, bounded by `limit_coefficients`. Calibration clips use their own
        (source) coefficients and test clips the target's, so the domains are corrected separately.
    """
    ref = reference_spectrum(reference_spectra or {d: device_spectra[d] for d in (source_devices or device_spectra)}, reference)
    raw = {d: ref / np.maximum(s, 1e-8) for d, s in device_spectra.items()}
    return {d: limit_coefficients(c, sc_mode, limit_freq_low, limit_freq_high, limit_freq_diff, sr, n_fft)
            for d, c in raw.items()}


def apply_spectrum_correction(wave, coef, n_fft=1024, hop=512):
    """Scale the STFT magnitude by `coef` per bin (phase kept) and resynthesise the waveform.

    Returns:
        Waveform of the same length as `wave`.
    """
    import librosa
    return librosa.istft(_stft(wave, n_fft, hop) * coef[:, None], hop_length=hop, length=len(wave))


def logmel_stats(logmels):
    """Per-mel-bin mean and SD over all clips and frames.

    Args:
        logmels: Iterable of (frames, mels) log-mel arrays (dB). Pass only the first N clips for the sensitivity run.

    Returns:
        Tuple (mean, sd), each (mels,).
    """
    x = np.concatenate([np.asarray(m) for m in logmels], axis=0)
    return x.mean(0), x.std(0) + 1e-6


def isa_match(logmel, target_stats, source_stats):
    """A2: map target log-mel statistics onto the source statistics, per mel bin.

    Args:
        logmel: (frames, mels) target log-mel.
        target_stats: (mean, sd) from `logmel_stats` on unlabelled target clips.
        source_stats: (mean, sd) from the training clips.

    Returns:
        Matched (frames, mels) array. A pure per-bin gain (offset in dB) is removed exactly.
    """
    (mt, st), (ms, ss) = target_stats, source_stats
    return (logmel - mt) / st * ss + ms


def random_bin_gain(logmel, max_db=6.0, knots=6, rng=None):
    """Add a smooth random per-mel-bin gain (dB), one curve per clip, linearly interpolated between random knots.

    Args:
        logmel: (frames, mels) log-mel in dB.
        max_db: SD of the knot values in dB.
        knots: Number of random control points across the mel axis.
        rng: `np.random.Generator`.

    Returns:
        Augmented array, same shape.
    """
    rng = rng or np.random.default_rng()
    curve = np.interp(np.linspace(0, 1, logmel.shape[1]), np.linspace(0, 1, knots), rng.normal(0, max_db, knots))
    return logmel + curve[None, :]
