"""OPERA-CT as a frozen frame-level encoder.

Uses the reference code in `../repos/OPERA` and the checkpoint
`../repos/OPERA/cks/model/encoder-operaCT.ckpt` (hf: evelyn0414/OPERA).
"""
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning  # noqa: F401  import before the `src` swap below, it inspects sys.modules
import torch

OPERA_ROOT = Path(__file__).resolve().parents[3] / "repos" / "OPERA"


def _import_cola(root):
    """Import OPERA's `Cola`. OPERA's package is also called `src`, so swap it in temporarily.

    Args:
        root: Path to the OPERA repo clone.

    Returns:
        The `Cola` class.
    """
    ours = {k: sys.modules.pop(k) for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]}
    ours_path = Path(__file__).resolve().parents[2]
    saved_path = sys.path[:]
    # our `src` is a regular package and would shadow OPERA's namespace package
    sys.path[:] = [str(root)] + [q for q in saved_path if q not in ("", ".") and Path(q).resolve() != ours_path]
    try:
        from src.model.models_cola import Cola
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(ours)
    return Cola


class OperaCT:
    """Frozen OPERA-CT (HTS-AT). Returns per-frame embeddings instead of the pooled vector.

    The HTS-AT latent output is an average over a (freq, time) token grid. A forward hook on
    its pooling layer grabs that grid, and frames are the mean over the frequency axis.

    Args:
        root: Path to the OPERA repo clone.
        device: Torch device.
    """

    def __init__(self, root=OPERA_ROOT, device="cpu"):
        root = Path(root)
        Cola = _import_cola(root)
        ckpt = torch.load(root / "cks/model/encoder-operaCT.ckpt", map_location="cpu")
        self.model = Cola(encoder="htsat")
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.eval().to(device)
        self.device = device
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._grid = None
        htsat = self.model.encoder.encoder.htsat
        self.c_freq_bin = htsat.spec_size // htsat.patch_stride[0] // 2 ** (len(htsat.depths) - 1) // htsat.freq_ratio
        htsat.avgpool.register_forward_hook(self._grab)

    def _grab(self, module, inp, out):
        if self._grid is None:  # first call is the latent-output pooling on (B, C, F*T)
            self._grid = inp[0]

    @torch.no_grad()
    def __call__(self, wave):
        """Embed one cycle.

        Args:
            wave: 1-D float32 waveform at 16 kHz, 8 s long (`src.data.audio.CYCLE_SEC`).

        Returns:
            (N, 768) float32 frame embeddings.
        """
        from librosa.feature import melspectrogram
        from librosa import power_to_db
        S = power_to_db(melspectrogram(y=wave, sr=16000, n_mels=64, fmin=50, fmax=2000,
                                       n_fft=1024, hop_length=512), ref=np.max)
        mel = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S  # as OPERA preprocessing
        x = torch.tensor(mel.T[None], dtype=torch.float, device=self.device)  # (1, T, 64)
        self._grid = None
        self.model.extract_feature(x, 768)
        g = self._grid  # (1, C, c_freq_bin * T')
        g = g.reshape(1, g.shape[1], self.c_freq_bin, -1).mean(2)  # (1, C, T')
        return g[0].T.cpu().numpy()
