"""HeAR (google/hear-pytorch, ViT-L masked autoencoder on 2 s health-acoustic clips) with Patch-Mix on its patch tokens.

An 8 s cycle is cut into four 2 s clips; each clip is a 192 x 128 PCEN mel image. The dataset image stacks the four clips
along time, (768, 128), and the encoder splits them again. Patch-Mix swaps the same clip and patch positions between samples.
"""

import importlib.util
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from src.paths import DATA, REPOS

from .patch_mix import mix_patches

MODEL = "google/hear-pytorch"
CACHE = DATA / "models" / "huggingface" / "hub"
AUDIO_UTILS = REPOS / "hear" / "python" / "data_processing" / "audio_utils.py"
CLIPS, CLIP_FRAMES, MELS, FEATURE_DIM = 4, 192, 128, 1024


class HearEncoder(nn.Module):
    final_feat_dim = FEATURE_DIM

    def __init__(self, n_cls, mix_beta):
        super().__init__()
        from transformers import AutoModel

        # gated model: the cached copy is used when the account has no access (request access at huggingface.co/google/hear-pytorch)
        self.vit = AutoModel.from_pretrained(
            MODEL,
            cache_dir=CACHE,
            local_files_only=(CACHE / "models--google--hear-pytorch").exists(),
        )
        self.mlp_head = nn.Sequential(nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, n_cls))
        self.mix_beta = mix_beta

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        batch = x.shape[0]
        h = self.vit.embeddings(
            x.reshape(batch * CLIPS, 1, CLIP_FRAMES, MELS)
        )  # (B * 4, 1 + patches, D)
        if patch_mix:
            patches = h[:, 1:].reshape(
                batch, -1, h.shape[-1]
            )  # clips and patches of one cycle in one sequence
            patches, y_a, y_b, lam, index = mix_patches(patches, y, self.mix_beta)
            h = torch.cat([h[:, :1], patches.reshape(batch * CLIPS, -1, h.shape[-1])], 1)
        for layer in self.vit.layers:
            h = layer(h)
        h = self.vit.layernorm(h)[:, 1:].mean(1).reshape(batch, CLIPS, -1).mean(1)
        return (h, y_a, y_b, lam, index) if patch_mix else h


def build_hear(args):
    return HearEncoder(args.n_cls, args.mix_beta)


@lru_cache(maxsize=1)
def _audio_utils():
    spec = importlib.util.spec_from_file_location("hear_audio_utils", AUDIO_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess_hear(wave):
    """(1, 128000) 16 kHz waveform -> (768, 128, 1): four 2 s mel-PCEN images stacked along time."""
    clips = _audio_utils().preprocess_audio(wave.reshape(CLIPS, 32000))  # (4, 1, 192, 128)
    return clips[:, 0].reshape(CLIPS * CLIP_FRAMES, MELS).numpy()[..., None].astype(np.float32)
