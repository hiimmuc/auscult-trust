"""CLAP audio tower (laion/clap-htsat-unfused, an HTS-AT Swin transformer on 48 kHz audio) with Patch-Mix on its patch tokens."""

import contextlib
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF
from src.paths import DATA

from .patch_mix import mix_patches

MODEL = "laion/clap-htsat-unfused"
CACHE = DATA / "models" / "huggingface" / "hub"
FEATURE_DIM = 768


class ClapEncoder(nn.Module):
    final_feat_dim = FEATURE_DIM

    def __init__(self, n_cls, mix_beta):
        super().__init__()
        from transformers import ClapModel

        self.enc = ClapModel.from_pretrained(MODEL, cache_dir=CACHE).audio_model.audio_encoder
        self.mlp_head = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, n_cls))
        self.mix_beta = mix_beta

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        e = self.enc
        x = e.batch_norm(x.transpose(1, 3)).transpose(1, 3)
        x = e.patch_embed(e.reshape_mel2img(x), None)
        if patch_mix:
            x, y_a, y_b, lam, index = mix_patches(x, y, self.mix_beta)
        for i, layer in enumerate(e.layers):
            x = layer(x, e.input_resolutions[i], False, False)[0]
        x = e.norm(x).mean(1)
        return (x, y_a, y_b, lam, index) if patch_mix else x


def build_clap(args):
    return ClapEncoder(args.n_cls, args.mix_beta)


@lru_cache(maxsize=1)
def _extractor():
    from transformers import ClapFeatureExtractor

    return ClapFeatureExtractor.from_pretrained(MODEL, cache_dir=CACHE)


def preprocess_clap(wave):
    """(1, N) 16 kHz waveform -> (1001, 64, 1) log-mel of CLAP's feature extractor (48 kHz, repeat-padded to 10 s)."""
    features = _extractor()(
        AF.resample(wave, 16000, 48000)[0].numpy(), sampling_rate=48000, return_tensors="np"
    )["input_features"]
    return features[0, 0][..., None].astype(np.float32)
