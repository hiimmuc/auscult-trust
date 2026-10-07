"""DASS (distilled audio state-space model, VMamba medium) from the Lung-SRAD repository, with its Patch-Mix.

Same input as AST (798 x 128 Kaldi fbank). The network is the Lung-SRAD one with the Gaussian blur blocks switched off, so
this is plain DASS; its own 2-D Patch-Mix (the AST-equivalent token swap) is used, like for the other encoders.
Needs the selective-scan CUDA kernel and runs in the separate `.venv-dass` (CUDA 12.8 build, see docs/SETUP.md).
Weights: DASS_medium_v2.pth (AudioSet) in `<data>/models/dass/pretrained_models/`.
"""
import contextlib
import importlib
import os
import sys
import types
from functools import lru_cache

import torch.nn as nn

from src.paths import DATA, REPOS
from ..cycles import FBANK_STD, IMG_MEL, IMG_TIME, generate_fbank

LUNG_SRAD = REPOS / 'Lung-SRAD' / 'models_DASS'
WEIGHTS_ROOT = DATA / 'models' / 'dass'  # holds pretrained_models/DASS_medium_v2.pth
FEATURE_DIM = 768


@contextlib.contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@lru_cache(maxsize=1)
def _dass_class():
    package = types.ModuleType('lung_srad_dass')
    package.__path__ = [str(LUNG_SRAD)]
    sys.modules['lung_srad_dass'] = package
    return importlib.import_module('lung_srad_dass.ast_models').DASS


class DassEncoder(nn.Module):
    final_feat_dim = FEATURE_DIM

    def __init__(self, n_cls, mix_beta):
        super().__init__()
        with _working_directory(WEIGHTS_ROOT):  # the reference reads ./pretrained_models/DASS_medium_v2.pth
            self.dass = _dass_class()(label_dim=n_cls, imagenet_pretrain=False, audioset_pretrain=True, enable_patch_mix=True,
                                      mix_beta=mix_beta, model_size='medium', blur_blocks=set(), gaussian_blur=True, verbose=False)
        self.mlp_head = nn.Linear(FEATURE_DIM, n_cls)  # fresh head; the reference's own head output is not used

    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        x = x.squeeze(1)  # (B, time, freq)
        if patch_mix:
            return self.dass(x, y=y, patch_mix=True, mix_type='2d')
        return self.dass(x)[0]


def build_dass(args):
    return DassEncoder(args.n_cls, args.mix_beta)


def preprocess_dass(wave):
    return generate_fbank(wave, 16000, n_mels=IMG_MEL)


IMAGE_SHAPE, DB_SCALE = (IMG_TIME, IMG_MEL), 1 / (2 * FBANK_STD)
