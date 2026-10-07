"""The Patch-Mix CL model and loss, taken unchanged from the read-only reference repository `repos/patch-mix_contrastive_learning`.

Only what the reference does not provide is added here: its attention is replaced by PyTorch's fused attention (same maths,
less memory, needed for batch 8 on a small GPU) and its hard-coded `./pretrained_models/` weight folder is resolved to
`<data>/models/ast/pretrained_models/` by changing the working directory while the model is built.
The reference modules are loaded by file path because their package names (`models`, `method`) are too generic to put on `sys.path`.
"""
import contextlib
import importlib.util
import os

import timm
import torch

from src.paths import DATA, REPOS

REFERENCE = REPOS / 'patch-mix_contrastive_learning'
WEIGHTS_ROOT = DATA / 'models' / 'ast'  # holds pretrained_models/audioset_10_10_0.4593.pth (scripts/download_data.sh)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REFERENCE / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _working_directory(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _fused_attention(self, x):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    x = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.scale,
                                                         dropout_p=self.attn_drop.p if self.training else 0.0)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))


_ast = _load('patchmix_reference_ast', 'models/ast.py')
Projector = _load('patchmix_reference_projector', 'models/projector.py').Projector
PatchMixLoss = _load('patchmix_reference_mix_loss', 'method/patchmix.py').PatchMixLoss
PatchMixConLoss = _load('patchmix_reference_loss', 'method/patchmix_cl.py').PatchMixConLoss


def build_ast(**kwargs):
    """`ASTModel` of the reference repository with fused attention; weights are read from `WEIGHTS_ROOT`."""
    timm.models.vision_transformer.Attention.forward = _fused_attention
    (WEIGHTS_ROOT / 'pretrained_models').mkdir(parents=True, exist_ok=True)
    with _working_directory(WEIGHTS_ROOT):  # the reference downloads or reads ./pretrained_models/audioset_10_10_0.4593.pth
        return _ast.ASTModel(**kwargs)
