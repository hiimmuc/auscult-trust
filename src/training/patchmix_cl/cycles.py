"""ICBHI breathing cycles for Patch-Mix CL: annotation reading, cycle slicing, 8 s cut/pad and the AST filter bank."""

import math
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torchaudio import transforms as T

FBANK_MEAN, FBANK_STD = -4.2677393, 4.5689974  # AST normalisation of the Kaldi fbank
IMG_TIME, IMG_MEL = 798, 128  # AST fbank of an 8 s cycle
FADE_RATIO = 16  # fade length = sample_rate / FADE_RATIO


def read_annotations(data_folder):
    """Annotation table per recording.

    Args:
        data_folder: Directory with `<recording>.txt` files (`start end crackle wheeze` per line).

    Returns:
        Dict recording stem -> DataFrame with `Start`, `End`, `Crackles`, `Wheezes`.
    """
    return {
        f[:-4]: pd.read_csv(
            os.path.join(data_folder, f),
            names=["Start", "End", "Crackles", "Wheezes"],
            delimiter="\t",
        )
        for f in sorted(os.listdir(data_folder))
        if f.endswith(".txt")
    }


def lungsound_label(crackle, wheeze, n_cls):
    """4-class (normal, crackle, wheeze, both) or 2-class (normal, abnormal) label."""
    if n_cls == 4:
        return int(crackle) + 2 * int(wheeze)
    return int(crackle or wheeze)


def cut_pad(data, desired_length, sample_rate, pad_types):
    """Cut a cycle to `desired_length` seconds, or pad it (`zero`: centred zeros, `repeat`: tiled with a fade-out)."""
    target = desired_length * sample_rate
    if data.shape[-1] > target:
        return data[..., :target]
    if pad_types == "zero":
        out = torch.zeros(1, target, dtype=torch.float32)
        diff = target - data.shape[-1]
        out[..., diff // 2 : data.shape[-1] + diff // 2] = data
        return out
    if pad_types == "repeat":
        data = data.repeat(1, math.ceil(target / data.shape[-1]))[..., :target]
        return T.Fade(
            fade_in_len=0, fade_out_len=int(sample_rate / FADE_RATIO), fade_shape="linear"
        )(data)
    return data


def individual_cycles(annotations, data_folder, filename, args):
    """Cut one recording into its annotated cycles, each cut/padded to `args.desired_length` seconds.

    Returns:
        List of (waveform (1, samples) tensor, label).
    """
    fpath = os.path.join(data_folder, filename + ".wav")
    sr = librosa.get_samplerate(fpath)
    # soundfile instead of torchaudio.load: torchaudio >= 2.9 needs torchcodec; same float32 PCM
    data = torch.from_numpy(sf.read(fpath, dtype="float32", always_2d=True)[0].T.copy())
    if sr != args.sample_rate:
        data = T.Resample(sr, args.sample_rate)(data)
    fade = int(args.sample_rate / FADE_RATIO)
    data = T.Fade(fade_in_len=fade, fade_out_len=fade, fade_shape="linear")(data)

    out = []
    for _, row in annotations.iterrows():
        chunk = data[
            :,
            min(int(row["Start"] * args.sample_rate), data.shape[1]) : min(
                int(row["End"] * args.sample_rate), data.shape[1]
            ),
        ]
        label = lungsound_label(row["Crackles"], row["Wheezes"], args.n_cls)
        out.append((cut_pad(chunk, args.desired_length, args.sample_rate, args.pad_types), label))
    return out


def generate_fbank(audio, sample_rate, n_mels=128):
    """Kaldi fbank of AST, normalised; shape (time, n_mels, 1) numpy."""
    assert sample_rate == 16000, "input audio sampling rate must be 16kHz"
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # the same maths on the GPU, much faster than 4000 cycles on the CPU
    fbank = torchaudio.compliance.kaldi.fbank(
        audio.to(device),
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=n_mels,
        dither=0.0,
        frame_shift=10,
    )
    return ((fbank - FBANK_MEAN) / (FBANK_STD * 2)).unsqueeze(-1).cpu().numpy()
