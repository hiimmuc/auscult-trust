"""Encoders that Patch-Mix CL can train: each has its own spectrogram front end and the Patch-Mix hook on its patch tokens.

An encoder module takes a batch of its spectrogram images (B, 1, time, freq) and returns pooled features (B, D), or with
`patch_mix=True` the tuple (features, labels a, labels b, lambda, partner index), like the reference AST. It also has
`final_feat_dim` and `mlp_head` (a fresh classification head).

AST, BEATs, DASS, HTS-AT and OPERA-CT are trained in both modes (full fine-tuning, frozen encoder); CLAP and HeAR only frozen.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    image_shape: Tuple[Optional[int], int]  # (time, freq) of the image of one 8 s cycle; time None = Kaldi frames of the cycle length
    db_scale: Optional[float]  # image units per dB of the log-mel; None where the image is not a log-mel
    frozen_only: bool
    preprocess: Callable  # (1, N) 16 kHz waveform tensor -> (time, freq, 1) float32 array
    build: Callable  # args -> encoder module


def _ast():
    from ..cycles import DB_PER_NAT, FBANK_STD, IMG_MEL, generate_fbank
    from ..upstream import build_ast
    spec = EncoderSpec('ast', (None, IMG_MEL), 1 / (2 * FBANK_STD * DB_PER_NAT), False, lambda w: generate_fbank(w, 16000, n_mels=IMG_MEL),
                       lambda a: build_ast(input_fdim=image_shape(spec, a)[0], input_tdim=IMG_MEL, label_dim=a.n_cls,
                                           imagenet_pretrain=a.from_sl_official, audioset_pretrain=a.audioset_pretrained, mix_beta=a.mix_beta))
    return spec


def _htsat():
    from .swin import build_htsat, preprocess_htsat
    return EncoderSpec('htsat', (1001, 64), 1.0, False, preprocess_htsat, build_htsat)


def _opera_ct():
    from .swin import build_opera_ct, preprocess_opera_ct
    return EncoderSpec('opera_ct', (251, 64), 1 / 80, False, preprocess_opera_ct, build_opera_ct)


def _clap():
    from .clap import build_clap, preprocess_clap
    return EncoderSpec('clap', (1001, 64), 1.0, True, preprocess_clap, build_clap)


def _hear():
    from .hear import build_hear, preprocess_hear
    return EncoderSpec('hear', (768, 128), None, True, preprocess_hear, build_hear)


def _dass():
    from .dass import DB_SCALE, IMAGE_SHAPE, build_dass, preprocess_dass
    return EncoderSpec('dass', IMAGE_SHAPE, DB_SCALE, False, preprocess_dass, build_dass)


def _beats():
    from .beats import DB_SCALE, build_beats, preprocess_beats
    return EncoderSpec('beats', (None, 128), DB_SCALE, False, preprocess_beats, build_beats)


_FACTORIES = {'ast': _ast, 'beats': _beats, 'dass': _dass, 'htsat': _htsat, 'opera_ct': _opera_ct, 'clap': _clap, 'hear': _hear}
NAMES = list(_FACTORIES)


def image_shape(spec, args):
    """(time, freq) of one cycle image; a None time axis is the number of 25 ms / 10 ms Kaldi frames of the cycle."""
    if spec.image_shape[0] is not None:
        return spec.image_shape
    return 1 + (args.desired_length * args.sample_rate - 400) // 160, spec.image_shape[1]


def get_encoder(name):
    """`EncoderSpec` of `name` (imports only that encoder's dependencies)."""
    return _FACTORIES[name]()
