"""Training recipes adopted from Patch-Mix CL (Bae et al., Interspeech 2023) for the cached-token rungs.

Patch-Mix CL fine-tunes AST end to end and mixes patch embeddings between samples at the transformer input. The
cached-token rungs start after block `token_layer`, so the mix happens there, on the token sequence. Flags are part of
the cell name (`ft-k4-attn+mix+ema`), hyperparameters sit under `ft:` in the config:

  mix   Patch-Mix: a random fraction 1 - lam of the patch tokens of each sample is replaced by the tokens of another sample
        (lam ~ Beta(alpha, alpha)); the mixed sample is pushed towards its clean self and its donor by the MixCL loss.
        `ft.mix: {alpha, con_weight, temp}`.
  ema   after every optimiser step the weights are averaged with the weights before the step. `ft.ema_beta`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Linear - BatchNorm - ReLU - Linear projection head of the contrastive branch."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, z):
        return self.net(z)


def patch_mix(tok, patch_index, alpha):
    """Replace a random fraction 1 - lam of the patch tokens of every sample by those of a random partner.

    Args:
        tok: (B, T, D) token sequences. Not modified.
        patch_index: (P,) positions of the patch tokens (class tokens stay).
        alpha: Beta(alpha, alpha) parameter of lam.

    Returns:
        Tuple (mixed tokens, partner index (B,), effective lam = share of original patches kept).
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    n_mask = int(len(patch_index) * (1.0 - lam))
    sel = patch_index[torch.randperm(len(patch_index), device=tok.device)[:n_mask]]
    perm = torch.randperm(len(tok), device=tok.device)
    mixed = tok.clone()
    mixed[:, sel] = tok[perm][:, sel]
    return mixed, perm, 1.0 - n_mask / len(patch_index)


def mixcl_loss(z_clean, z_mix, perm, lam, temp):
    """Patch-Mix contrastive loss: the mixed sample i has positives clean i (weight lam) and clean perm[i] (weight 1 - lam).

    Args:
        z_clean: (B, d) clean pooled features (detached by the caller).
        z_mix: (B, d) projected pooled features of the mixed samples.
        perm: (B,) partner index used for the mix.
        lam: Share of original patches kept.
        temp: Softmax temperature.

    Returns:
        Scalar loss. All other samples of the batch are negatives, as in the reference code (`negative_pair=all`).
    """
    B = len(z_mix)
    logits = F.normalize(z_mix) @ F.normalize(z_clean).T / temp
    logits = logits - logits.max(1, keepdim=True).values.detach()
    log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
    pos = lam * torch.eye(B, device=z_mix.device)
    pos[torch.arange(B), perm] += 1.0 - lam
    return -((pos * log_prob).sum(1) / pos.sum(1)).mean()


def mixcl_term(model, b, z, patch_index, cfg):
    """MixCL loss for one batch: mix tokens, embed the mixture, project, contrast with the detached clean features."""
    tok, perm, lam = patch_mix(b["tok"], patch_index, cfg["alpha"])
    zm = model.projector(model.embed({"tok": tok, "mask": b["mask"] | b["mask"][perm]}))
    return mixcl_loss(z.detach().float(), zm.float(), perm, lam, cfg["temp"])


@torch.no_grad()
def ema_step(params, before, beta):
    """Set each parameter to `beta * (value before the step) + (1 - beta) * (value after the step)`."""
    for p, old in zip(params, before):
        p.mul_(1 - beta).add_(old, alpha=beta)
