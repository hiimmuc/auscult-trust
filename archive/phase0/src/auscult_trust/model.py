"""AuscultTrust model: layer mixing, frame fusion, masked pooling over real frames, optional trainable ViT tail."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import encoders
from src.auscult_trust.augment import Projector


class Head(nn.Module):
    """Layer mixing, per-frame fusion, masked pooling over real frames, linear classifier.

    Args:
        dim: Feature size D.
        n_layers: Cached layers L in the input (B, L, 32, D).
        layer_mode: `last` (final layer only), `concat` (all layers), `scalar` (learned softmax mix).
        pool: `mean`, `max`, `meanmax` or `attn` over the real frames.
    """

    def __init__(self, dim, n_layers, layer_mode="last", pool="mean", hidden=128, n_cls=4):
        super().__init__()
        self.layer_mode, self.pool = layer_mode, pool
        self.w = nn.Parameter(torch.zeros(n_layers)) if layer_mode == "scalar" else None
        self.fuse = nn.Sequential(nn.Linear(dim * (n_layers if layer_mode == "concat" else 1), hidden),
                                  nn.LayerNorm(hidden), nn.GELU())
        self.att = nn.Linear(hidden, 1) if pool == "attn" else None
        h = hidden * (2 if pool == "meanmax" else 1)
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, n_cls))

    def embed(self, f, mask):
        """Pooled clip feature (B, hidden) of cached frames f (B, L, 32, D) over the real frames `mask` (B, 32)."""
        f = F.layer_norm(f, f.shape[-1:])
        if self.layer_mode == "scalar":
            f = (self.w.softmax(0)[None, :, None, None] * f).sum(1)
        elif self.layer_mode == "concat":
            f = f.permute(0, 2, 1, 3).flatten(2)
        else:
            f = f[:, -1]
        x, m = self.fuse(f), mask.unsqueeze(-1)
        mean = (x * m).sum(1) / m.sum(1)
        mx = x.masked_fill(~m, -1e4).max(1).values
        if self.pool == "mean":
            z = mean
        elif self.pool == "max":
            z = mx
        elif self.pool == "meanmax":
            z = torch.cat([mean, mx], -1)
        else:
            a = self.att(x).squeeze(-1).masked_fill(~mask, -1e4).softmax(1)
            z = (x * a.unsqueeze(-1)).sum(1)
        return z

    def forward(self, f, mask):
        return self.out(self.embed(f, mask))


class Tail(nn.Module):
    """ViT blocks after the cached layer: blocks up to `n - k` frozen, the last `k` trainable (k = 0: all frozen).

    With `lora_r > 0` all blocks after the cached layer are frozen and carry rank-`lora_r` LoRA adapters instead.
    """

    def __init__(self, encoder, token_layer, k, lora_r=0):
        super().__init__()
        vit = encoders.load(encoder).model
        n, t = len(vit.layers), token_layer
        self.enc, self.k = encoders.get(encoder), k
        if lora_r:  # LoRA on q_proj/v_proj of every block after the cached layer; base weights stay frozen
            from peft import LoraConfig, inject_adapter_in_model
            self.frozen = nn.ModuleList()
            blocks = copy.deepcopy(vit.layers[t:]).requires_grad_(False)
            self.train_blocks = inject_adapter_in_model(
                LoraConfig(r=lora_r, lora_alpha=2 * lora_r, target_modules=["q_proj", "v_proj"], lora_dropout=0.0), blocks)
            self.norm = copy.deepcopy(vit.layernorm).requires_grad_(False)
        else:
            self.frozen = copy.deepcopy(vit.layers[t:n - k]).requires_grad_(False)
            self.train_blocks = copy.deepcopy(vit.layers[n - k:]).requires_grad_(k > 0)
            self.norm = copy.deepcopy(vit.layernorm).requires_grad_(k > 0)
        del vit

    def forward(self, tok):
        B = tok.shape[0]
        tok = tok.reshape(B * self.enc.units, -1, tok.shape[-1])  # blocks see one unit (HeAR: one 2 s clip) at a time
        with torch.no_grad():
            for blk in self.frozen:
                tok = blk(tok)
        for blk in self.train_blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return self.enc.token_frames(tok.reshape(B, -1, tok.shape[-1]))


class AuscultTrust(nn.Module):
    """Head on cached frame features (`tail=None`) or on frames recomputed by a `Tail` from cached tokens.

    Args:
        head: `Head`.
        tail: `Tail` with the last k ViT blocks trainable, or None to use cached frame features.
        projector: Contrastive projection head of the `+mix` recipe, or None.
    """

    def __init__(self, head, tail=None, projector=None):
        super().__init__()
        self.head = head
        if tail is not None:
            self.tail = tail
        if projector is not None:
            self.projector = projector

    def embed(self, b):
        """Pooled clip feature (B, hidden) of a batch from `FeatureData.batch`."""
        if hasattr(self, "tail"):
            return self.head.embed(self.tail(b["tok"])[:, None], b["mask"])
        return self.head.embed(b["f"], b["mask"])

    def forward(self, b):
        return self.head.out(self.embed(b))


def mode_of(cell):
    """`tokens` for cells that recompute the ViT tail from cached tokens (`ft-*`, `lora-*`), else `frames`."""
    return "tokens" if cell.startswith(("ft-", "lora-")) else "frames"


def parse_cell(cell):
    """Split a cell name into (kind, arg, pool, flags).

    Names: `<layer_mode>-<pool>` (kind `head`), `ft-k<k>-<pool>`, `lora-r<r>-<pool>`; the token rungs may carry recipe
    flags, e.g. `ft-k4-attn+mix+ema` (see `src.auscult_trust.augment`).
    """
    base, *flags = cell.split("+")
    parts = base.split("-")
    kind, arg, pool = parts if parts[0] in ("ft", "lora") else ("head", *parts)
    assert not flags or kind != "head", f"recipe flags need a token rung: {cell}"
    assert set(flags) <= {"mix", "ema"}, f"unknown flag in {cell}"
    return kind, arg, pool, set(flags)


def make_model(cfg, cell, D):
    """Build the model of a cell name (see `parse_cell`).

    Returns:
        Tuple (model, mode) with mode `frames` (cached features) or `tokens` (tail recomputed).
    """
    kind, arg, pool, flags = parse_cell(cell)
    hidden = cfg["head"]["hidden"]
    if kind == "head":
        return AuscultTrust(Head(D.layers.shape[-1], D.layers.shape[1], arg, pool, hidden)), "frames"
    tail = Tail(D.encoder, D.token_layer, int(arg[1:])) if kind == "ft" else Tail(D.encoder, D.token_layer, 0, lora_r=int(arg[1:]))
    return AuscultTrust(Head(D.layers.shape[-1], 1, "last", pool, hidden), tail, Projector(hidden) if "mix" in flags else None), "tokens"
