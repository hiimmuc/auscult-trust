"""OPERA-CT (HTS-AT) and OPERA-CE (EfficientNet-B0) as frozen frame-level encoders.

Uses the reference code in `../repos/OPERA` (read-only). Checkpoints: CT `../repos/OPERA/cks/model/encoder-operaCT.ckpt`,
CE `../data/models/opera/encoder-operaCE.ckpt` (hf: evelyn0414/OPERA). Both were pretrained on ICBHI-train and
HF_Lung-train recordings, so they are the encoders under the leakage test (proposal A+B, H2 of proposal H).
Input for both: 64-mel, 50-8000 Hz, n_fft 1024, hop 512, power-to-dB, per-clip min-max, as in OPERA pretraining.
"""
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning  # noqa: F401  import before the `src` swap below, it inspects sys.modules
import torch
from librosa import power_to_db
from librosa.feature import melspectrogram

from src.encoders.base import DEVICE, Encoder, to_frames

OPERA_ROOT = Path(__file__).resolve().parents[3] / "repos" / "OPERA"
CKPT = {"htsat": OPERA_ROOT / "cks/model/encoder-operaCT.ckpt",
        "efficientnet": Path(__file__).resolve().parents[3] / "data" / "models" / "opera" / "encoder-operaCE.ckpt"}


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
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # ../repos is read-only reference
    # our `src` is a regular package and would shadow OPERA's namespace package
    sys.path[:] = [str(root)] + [q for q in saved_path if q not in ("", ".") and Path(q).resolve() != ours_path]
    try:
        from src.model.models_cola import Cola
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = dont_write
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            del sys.modules[k]
        sys.modules.update(ours)
    return Cola


def _load(kind):
    """Frozen OPERA `Cola` with the encoder weights of `CKPT[kind]`.

    The checkpoint also holds contrastive-head weights that the frozen encoder never uses; any *missing* encoder
    weight would leave random weights silently, so it is an error.
    """
    model = _import_cola(OPERA_ROOT)(encoder=kind)
    res = model.load_state_dict(torch.load(CKPT[kind], map_location="cpu")["state_dict"], strict=False)
    assert not [k for k in res.missing_keys if k.startswith("encoder")], f"OPERA {kind}: encoder weights missing {res.missing_keys[:3]}"
    return model.eval().to(DEVICE).requires_grad_(False)


def _logmel(wave):
    """(1, T, 64) float32 tensor on the device, OPERA preprocessing (`pre_process_audio_mel_t(f_max=8000)`)."""
    S = power_to_db(melspectrogram(y=wave, sr=16000, n_mels=64, fmin=50, fmax=8000, n_fft=1024, hop_length=512), ref=np.max)
    mel = (S - S.min()) / (S.max() - S.min()) if S.max() != S.min() else S
    return torch.tensor(mel.T[None], dtype=torch.float, device=DEVICE)


class OperaCT(Encoder):
    """OPERA-CT (HTS-AT). Frames = the latent token grid averaged over frequency, resampled to 32 frames.

    A forward hook on the HTS-AT pooling layer grabs the (freq, time) token grid that OPERA averages into its vector.
    """
    n_layers, dim = 1, 768

    def __init__(self):
        self.model = _load("htsat")
        self._grid = None
        htsat = self.model.encoder.encoder.htsat
        self.c_freq_bin = htsat.spec_size // htsat.patch_stride[0] // 2 ** (len(htsat.depths) - 1) // htsat.freq_ratio
        htsat.avgpool.register_forward_hook(self._grab)

    def _grab(self, module, inp, out):
        if self._grid is None:  # first call is the latent-output pooling on (B, C, F*T)
            self._grid = inp[0]

    @torch.no_grad()
    def run(self, wave, keep, token_layer):
        self._grid = None
        self.model.extract_feature(_logmel(wave), self.dim)
        g = self._grid.reshape(1, self.dim, self.c_freq_bin, -1).mean(2)  # (1, C, T')
        return {1: to_frames(g[0].T)}, None


class OperaCE(Encoder):
    """OPERA-CE (EfficientNet-B0). Frames = the feature map over time (mel axis collapsed), resampled to 32 frames.

    The model's own 1280-d vector is the global mean of the same map; `run(...)[0][1].mean(0)` matches it.
    """
    n_layers, dim = 1, 1280

    def __init__(self):
        self.model = _load("efficientnet")

    @torch.no_grad()
    def run(self, wave, keep, token_layer):
        enc = self.model.encoder
        fmap = enc.efficientnet.extract_features(enc.cnn1(_logmel(wave).unsqueeze(1)))  # (1, 1280, T' = 7, 1)
        return {1: to_frames(fmap.mean(3)[0].T)}, None  # OPERA feeds (time, mel) as (height, width): dim 2 is time
