"""Stationary environmental noise from ESC-50 (rain, wind, sea_waves, engine, vacuum_cleaner, washing_machine, airplane, train).

Impulsive classes (crackling_fire, clock_tick, door_wood_knock, footsteps, ...) are excluded because they can look like
crackles. Clips are resampled to 16 kHz and tiled to 8 s. ESC-50 folds keep the 5 recordings of one source clip together, so
noise for training (`TRAIN_FOLDS`) and for testing (`TEST_FOLDS`) never share a source.
"""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from src.paths import DATA

ESC50 = DATA / "raw" / "esc50" / "ESC-50-master"
NOISE_CLASSES = ("rain", "wind", "sea_waves", "engine", "vacuum_cleaner", "washing_machine", "airplane", "train")
TRAIN_FOLDS, TEST_FOLDS = (1, 2, 3), (4, 5)
SR, SAMPLES = 16000, 16000 * 8


def noise_bank(folds=None):
    """(n, SAMPLES) float32 array of the noise clips of the given ESC-50 folds (all folds when None), sorted by file name."""
    meta = pd.read_csv(ESC50 / "meta" / "esc50.csv")
    meta = meta[meta.category.isin(NOISE_CLASSES) & (meta.fold.isin(folds) if folds else True)]
    clips = []
    for f in sorted(meta.filename):
        x, sr = sf.read(ESC50 / "audio" / f, dtype="float32")
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
        if np.mean(x ** 2) < 1e-8:  # a few ESC-50 clips are near silent
            continue
        x = np.tile(x, int(np.ceil(SAMPLES / len(x))))[:SAMPLES]
        clips.append(x.astype(np.float32))
    return np.stack(clips)
