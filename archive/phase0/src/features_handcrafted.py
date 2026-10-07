"""74-dim handcrafted descriptors (H-diag, H-inv): mean and SD of 20 MFCC, ZCR, centroid, bandwidth, 12 chroma, RMS, rolloff.

Kept apart from `features.py`: these are per-cycle vectors, not frame caches.
"""
import numpy as np

SR = 16000
N_DIM = 74


def handcrafted(wave, sr=SR):
    """Mean and SD over frames of 37 frame-level descriptors.

    Args:
        wave: 1-D float waveform.
        sr: Sample rate in Hz.

    Returns:
        (74,) float32 vector: 37 means then 37 SDs.
    """
    import librosa
    wave = np.asarray(wave, dtype=np.float32)
    S = np.abs(librosa.stft(wave, n_fft=1024, hop_length=512))
    rows = [librosa.feature.mfcc(y=wave, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512),
            librosa.feature.zero_crossing_rate(wave, frame_length=1024, hop_length=512),
            librosa.feature.spectral_centroid(S=S, sr=sr),
            librosa.feature.spectral_bandwidth(S=S, sr=sr),
            librosa.feature.chroma_stft(S=S ** 2, sr=sr),
            librosa.feature.rms(S=S, frame_length=1024),
            librosa.feature.spectral_rolloff(S=S, sr=sr)]
    n = min(r.shape[1] for r in rows)
    F = np.vstack([r[:, :n] for r in rows])
    return np.r_[F.mean(1), F.std(1)].astype(np.float32)


def handcrafted_table(waves, sr=SR):
    """Stack `handcrafted` over an iterable of waveforms. Returns (n, 74)."""
    return np.stack([handcrafted(w, sr) for w in waves])
