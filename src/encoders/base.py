"""Interface shared by every frozen encoder.

An encoder maps one 8 s, 16 kHz cycle to `N_FRAMES` frame embeddings per cached layer, so every
downstream rung (probe, heads, fine-tuning) sees the same (32, D) layout whatever the backbone.
"""
import numpy as np
import torch
import torch.nn.functional as F

N_FRAMES = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_frames_batch(x, n=N_FRAMES):
    """Average-pool (B, time, D) sequences to `n` frames. Returns (B, n, D)."""
    return F.adaptive_avg_pool1d(x.transpose(1, 2), n).transpose(1, 2)


def to_frames(x, n=N_FRAMES):
    """Average-pool a (time, D) sequence to `n` frames. Returns (n, D)."""
    return to_frames_batch(x[None], n)[0]


class Encoder:
    """Frozen encoder.

    Class attributes:
        dim: Feature size D.
        n_layers: Layers that can be cached (1 for encoders that only expose their last layer).
        token_shape: (tokens, D) of the cached token sequence, or None when the encoder cannot be fine-tuned from tokens.
        units: Independent input units per cycle (HeAR: four 2 s clips); blocks see one unit at a time.
    """
    dim, n_layers, token_shape, units = 0, 1, None, 1

    @staticmethod
    def patch_index():
        """(P,) positions of the patch tokens inside the cached token sequence (class tokens excluded)."""
        raise NotImplementedError

    @staticmethod
    def token_frames(tok):
        """(B, T, D) cached token sequences after the last block -> (B, N_FRAMES, D) frames."""
        raise NotImplementedError

    def run(self, wave, keep, token_layer):
        """Embed one cycle.

        Args:
            wave: 1-D float32 waveform, 16 kHz, 8 s, not band-passed.
            keep: Layer numbers (1-based) whose frames are returned.
            token_layer: Layer after which the token sequence is returned, or None.

        Returns:
            Tuple (dict layer -> (N_FRAMES, D) tensor, token tensor or None).
        """
        raise NotImplementedError

    def __call__(self, wave):
        """Last-layer frames as a (N_FRAMES, D) float32 numpy array."""
        return self.run(wave, {self.n_layers}, None)[0][self.n_layers].float().cpu().numpy().astype(np.float32)
