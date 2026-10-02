"""Audio loading and cycle cropping. Mono float32 at a common sample rate."""
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt

SR = 16000  # OPERA input rate
CYCLE_SEC = 8.0  # fixed cycle length fed to the encoder


def load_wav(path, sr=SR):
    """Load a wav file as mono float32 resampled to `sr`.

    Args:
        path: Path to a PCM wav file.
        sr: Target sample rate in Hz.

    Returns:
        1-D float32 array scaled to [-1, 1].
    """
    rate, x = wavfile.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    if rate != sr:
        g = gcd(rate, sr)
        x = resample_poly(x, sr // g, rate // g).astype(np.float32)
    return x


def bandpass(x, sr=SR, lo=50.0, hi=2000.0):
    """Zero-phase Butterworth band-pass.

    Args:
        x: 1-D waveform.
        sr: Sample rate in Hz.
        lo: Low cut-off in Hz.
        hi: High cut-off in Hz. Must be below sr / 2.

    Returns:
        Filtered float32 waveform.
    """
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def crop_pad(x, n):
    """Crop or zero-pad a waveform to exactly `n` samples.

    Args:
        x: 1-D waveform.
        n: Target length in samples.

    Returns:
        1-D array of length `n`.
    """
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def cycle_wave(x, start, end, sr=SR, sec=CYCLE_SEC):
    """Cut one breath cycle out of a recording and fix its length.

    Args:
        x: Full recording at rate `sr`.
        start: Cycle start in seconds.
        end: Cycle end in seconds.
        sr: Sample rate in Hz.
        sec: Output length in seconds.

    Returns:
        1-D array of `sec * sr` samples.
    """
    return crop_pad(x[int(start * sr):int(end * sr)], int(sec * sr))
