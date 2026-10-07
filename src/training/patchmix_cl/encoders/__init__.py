"""Encoders that Patch-Mix CL can train: each has its own spectrogram front end and the Patch-Mix hook on its patch tokens.

An encoder module takes a batch of its spectrogram images (B, 1, time, freq) and returns pooled features (B, D), or with
`patch_mix=True` the tuple (features, labels a, labels b, lambda, partner index), like the reference AST. It also has
`final_feat_dim` and `mlp_head` (a fresh classification head).

AST, DASS and HTS-AT are trained in both modes (full fine-tuning, frozen encoder); OPERA-CT, CLAP and HeAR only frozen.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    image_shape: Tuple[int, int]  # (time, freq) of the spectrogram image of one 8 s cycle
    db_scale: Optional[float]  # image units per dB of the log-mel; None where the image is not a log-mel
    frozen_only: bool
    preprocess: Callable  # (1, N) 16 kHz waveform tensor -> (time, freq, 1) float32 array
    build: Callable  # args -> encoder module


def _ast():
    from ..cycles import FBANK_STD, IMG_MEL, IMG_TIME, generate_fbank
    from ..upstream import build_ast
    return EncoderSpec('ast', (IMG_TIME, IMG_MEL), 1 / (2 * FBANK_STD), False, lambda w: generate_fbank(w, 16000, n_mels=IMG_MEL),
                       lambda a: build_ast(input_fdim=IMG_TIME, input_tdim=IMG_MEL, label_dim=a.n_cls, imagenet_pretrain=a.from_sl_official,
                                           audioset_pretrain=a.audioset_pretrained, mix_beta=a.mix_beta))


def _htsat():
    from .swin import build_htsat, preprocess_htsat
    return EncoderSpec('htsat', (801, 64), 1.0, False, preprocess_htsat, build_htsat)


def _opera_ct():
    from .swin import build_opera_ct, preprocess_opera_ct
    return EncoderSpec('opera_ct', (251, 64), 1 / 80, True, preprocess_opera_ct, build_opera_ct)


def _clap():
    from .clap import build_clap, preprocess_clap
    return EncoderSpec('clap', (1001, 64), 1.0, True, preprocess_clap, build_clap)


def _hear():
    from .hear import build_hear, preprocess_hear
    return EncoderSpec('hear', (768, 128), None, True, preprocess_hear, build_hear)


_FACTORIES = {'ast': _ast, 'htsat': _htsat, 'opera_ct': _opera_ct, 'clap': _clap, 'hear': _hear}
NAMES = list(_FACTORIES)


def get_encoder(name):
    """`EncoderSpec` of `name` (imports only that encoder's dependencies)."""
    return _FACTORIES[name]()
