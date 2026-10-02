"""Tent-style TTA restricted to normalisation parameters of the branch and head.

The encoder is frozen and never touched. Only LayerNorm affine parameters of the
trainable modules are updated.
"""
import copy

import torch
import torch.nn as nn


def norm_params(model):
    """Collect LayerNorm weight/bias of a LungModel.

    Args:
        model: A `LungModel`.

    Returns:
        List of parameters. All other parameters are frozen by `tent_adapt`.
    """
    return [p for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()]


def entropy(logits):
    """Mean softmax entropy.

    Args:
        logits: (B, K) logits.

    Returns:
        Scalar tensor.
    """
    lp = logits.log_softmax(-1)
    return -(lp.exp() * lp).sum(-1).mean()


def tent_adapt(model, batches, lr=1e-3, steps=1):
    """Adapt a copy of `model` by entropy minimisation on unlabelled target batches.

    Args:
        model: Trained `LungModel`. Left unchanged.
        batches: Iterable of kwargs dicts for `model(**batch)`, e.g. `{"emb":..., "wave":..., "phase":...}`.
        lr: SGD learning rate.
        steps: Gradient steps per batch.

    Returns:
        Adapted model copy in eval mode. Recompute calibration probabilities with it
        before any conformal calibration.
    """
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)
    params = norm_params(m)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.SGD(params, lr=lr)
    for b in batches:
        for _ in range(steps):
            opt.zero_grad()
            entropy(m(**b)).backward()
            opt.step()
    return m.eval()
