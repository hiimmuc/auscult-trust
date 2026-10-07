"""HTS-AT family (Swin transformer on a 64-mel spectrogram): HTS-AT itself and OPERA-CT.

Both use the HTS-AT network of the OPERA repository (identical architecture, different weights):
HTS-AT: AudioSet weights of the HTS-AT authors, 32 kHz input; OPERA-CT: weights pretrained on respiratory audio, 16 kHz input.
The encoder input is the (B, 1, time, 64) log-mel spectrogram; batch norm, folding to a square image, patch embedding,
Swin stages and the mean over tokens are those of the reference network, with Patch-Mix applied to the patch tokens.
"""

import importlib
import sys
import types
from functools import lru_cache

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF
from src.paths import DATA, REPOS

from .patch_mix import mix_patches

OPERA_HTSAT = REPOS / "OPERA" / "src" / "model" / "htsat"
HTSAT_AUDIOSET = DATA / "models" / "htsat" / "AudioSet" / "HTSAT_AudioSet_Saved_1.ckpt"
OPERA_CT = DATA / "models" / "opera" / "encoder-operaCT.ckpt"
FEATURE_DIM = 768


@lru_cache(maxsize=1)
def _network_module():
    """The reference `htsat.py` loaded as a package so that its relative imports work."""
    package = types.ModuleType("opera_htsat")
    package.__path__ = [str(OPERA_HTSAT)]
    sys.modules["opera_htsat"] = package
    return importlib.import_module("opera_htsat.htsat"), importlib.import_module(
        "opera_htsat.config"
    )


class SwinEncoder(nn.Module):
    final_feat_dim = FEATURE_DIM

    def __init__(self, checkpoint, prefix, n_cls, mix_beta):
        super().__init__()
        module, config = _network_module()
        self.htsat = module.HTSAT_Swin_Transformer(config=config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        self.htsat.load_state_dict(state, strict=True)
        self.mlp_head = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, n_cls))
        self.mix_beta = mix_beta

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        h = self.htsat
        x = h.bn0(x.transpose(1, 3)).transpose(1, 3)
        x = h.patch_embed(h.reshape_wav2img(x))
        if h.ape:
            x = x + h.absolute_pos_embed
        x = h.pos_drop(x)
        if patch_mix:
            x, y_a, y_b, lam, index = mix_patches(x, y, self.mix_beta)
        for layer in h.layers:
            x, _ = layer(x)
        x = h.norm(x).mean(1)
        return (x, y_a, y_b, lam, index) if patch_mix else x


def build_htsat(args):
    return SwinEncoder(HTSAT_AUDIOSET, "sed_model.", args.n_cls, args.mix_beta)


def build_opera_ct(args):
    return SwinEncoder(OPERA_CT, "encoder.encoder.htsat.", args.n_cls, args.mix_beta)


@lru_cache(maxsize=1)
def _htsat_frontend():
    from torchlibrosa.stft import LogmelFilterBank, Spectrogram

    spec = Spectrogram(
        n_fft=1024,
        hop_length=320,
        win_length=1024,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    )
    mel = LogmelFilterBank(
        sr=32000,
        n_fft=1024,
        n_mels=64,
        fmin=50,
        fmax=14000,
        ref=1.0,
        amin=1e-10,
        top_db=None,
        freeze_parameters=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return spec.eval().to(device), mel.eval().to(device)


def preprocess_htsat(wave):
    """(1, N) 16 kHz waveform -> (time, 64, 1) log-mel as the HTS-AT authors compute it (32 kHz, hop 320)."""
    spec, mel = _htsat_frontend()
    with torch.no_grad():
        out = mel(
            spec(AF.resample(wave.to(next(spec.parameters()).device), 16000, 32000))
        )  # (1, 1, time, 64)
    return out[0, 0].cpu().numpy()[..., None].astype(np.float32)


def preprocess_opera_ct(wave):
    """(1, N) 16 kHz waveform -> (time, 64, 1) log-mel with OPERA's preprocessing (50-8000 Hz, hop 512, scaled to [0, 1] per clip)."""
    S = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=wave[0].numpy(), sr=16000, n_mels=64, fmin=50, fmax=8000, n_fft=1024, hop_length=512
        ),
        ref=np.max,
    )
    mel = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S
    return mel.T[..., None].astype(np.float32)
