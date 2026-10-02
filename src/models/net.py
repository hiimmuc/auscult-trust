"""Trainable parts on top of a frozen encoder: short-window branch, phase pooling, head.

Frame embeddings come from the frozen encoder (cached). Phase mask values per frame:
0 = none, 1 = inspiration, 2 = expiration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortBranch(nn.Module):
    """Short-window branch: ~5 ms STFT, small conv, pooled to the encoder frame rate.

    Args:
        hidden: Output channels.
        win: STFT window in samples (80 = 5 ms at 16 kHz).
        hop: STFT hop in samples.
    """

    def __init__(self, hidden=64, win=80, hop=40):
        super().__init__()
        self.win, self.hop = win, hop
        self.conv = nn.Sequential(nn.Conv1d(win // 2 + 1, hidden, 5, padding=2),
                                  nn.GELU(), nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
        self.norm = nn.LayerNorm(hidden)

    def forward(self, wave, n_frames):
        """Compute short-window features aligned to encoder frames.

        Args:
            wave: (B, T) waveform.
            n_frames: Number of encoder frames N to pool to.

        Returns:
            (B, N, hidden) features.
        """
        w = torch.hann_window(self.win, device=wave.device)
        spec = torch.stft(wave, self.win, self.hop, window=w, return_complex=True).abs()
        h = self.conv(torch.log1p(spec))  # (B, hidden, T')
        h = F.adaptive_max_pool1d(h, n_frames)  # max keeps short transients
        return self.norm(h.transpose(1, 2))


def phase_pool(x, phase):
    """Mean of frames within inspiration and within expiration, concatenated.

    Args:
        x: (B, N, H) frame features.
        phase: (B, N) integer mask, 1 = inspiration, 2 = expiration.

    Returns:
        (B, 2H). A phase with no frames gives zeros.
    """
    outs = []
    for p in (1, 2):
        m = (phase == p).unsqueeze(-1).to(x.dtype)
        outs.append((x * m).sum(1) / m.sum(1).clamp(min=1))
    return torch.cat(outs, -1)


class LungModel(nn.Module):
    """Frame fusion, optional short branch, optional phase pooling, linear head.

    Args:
        emb_dim: Encoder embedding size D.
        n_classes: Output classes (normal, crackle, wheeze, both).
        hidden: Fusion width.
        use_branch: Enable the short-window branch (2x2 ablation axis).
        use_phase: Enable phase-aware pooling (2x2 ablation axis).
        shuffle_phase: Mechanism control. Permute the phase mask over time.
    """

    def __init__(self, emb_dim, n_classes=4, hidden=128, use_branch=True, use_phase=True,
                 shuffle_phase=False):
        super().__init__()
        self.use_branch, self.use_phase, self.shuffle_phase = use_branch, use_phase, shuffle_phase
        self.branch = ShortBranch() if use_branch else None
        self.fuse = nn.Sequential(nn.Linear(emb_dim + (64 if use_branch else 0), hidden),
                                  nn.LayerNorm(hidden), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(hidden * (2 if use_phase else 1)),
                                  nn.Linear(hidden * (2 if use_phase else 1), n_classes))

    def forward(self, emb, wave=None, phase=None):
        """Classify a batch of cycles.

        Args:
            emb: (B, N, D) frozen-encoder frame embeddings.
            wave: (B, T) waveform. Required when `use_branch`.
            phase: (B, N) phase mask. Required when `use_phase`.

        Returns:
            (B, n_classes) logits.
        """
        x = emb
        if self.use_branch:
            x = torch.cat([x, self.branch(wave, emb.shape[1])], -1)
        x = self.fuse(x)
        if self.use_phase:
            if self.shuffle_phase:
                phase = phase[:, torch.randperm(phase.shape[1], device=phase.device)]
            return self.head(phase_pool(x, phase))
        return self.head(x.mean(1))


class PhaseDetector(nn.Module):
    """Frame-level phase classifier on frozen embeddings. Train on HF_Lung_V1 labels.

    A BiGRU gives each frame temporal context, which is needed to tell inspiration from expiration.

    Args:
        emb_dim: Encoder embedding size D.
        hidden: GRU hidden size per direction.
    """

    def __init__(self, emb_dim, hidden=64):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, emb):
        """Predict per-frame phase logits.

        Args:
            emb: (B, N, D) frame embeddings.

        Returns:
            (B, N, 3) logits over {none, inspiration, expiration}.
        """
        return self.out(self.gru(self.norm(emb))[0])
