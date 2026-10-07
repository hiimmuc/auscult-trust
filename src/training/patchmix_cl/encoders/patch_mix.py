"""Patch-Mix for any token-based encoder, with the semantics of the reference AST implementation (time_domain=False)."""

import numpy as np
import torch


def mix_patches(tokens, target, mix_beta):
    """Replace a random subset of the patch tokens of every sample by the same positions of another sample of the batch.

    Args:
        tokens: (B, N, D) patch tokens right after the patch embedding.
        target: (B,) labels.
        mix_beta: Beta(mix_beta, mix_beta) concentration of the mixing ratio (0: no mixing).

    Returns:
        Tuple (mixed tokens, labels a, labels b, lambda = share of own tokens, partner index).
    """
    lam = np.random.beta(mix_beta, mix_beta) if mix_beta > 0 else 1
    batch, num_patch, _ = tokens.shape
    index = torch.randperm(batch).to(tokens.device)
    num_mask = int(num_patch * (1.0 - lam))
    mask = torch.randperm(num_patch)[:num_mask].to(tokens.device)
    tokens = tokens.clone()
    tokens[:, mask, :] = tokens[index][:, mask, :]
    return tokens, target, target[index], 1 - num_mask / num_patch, index
