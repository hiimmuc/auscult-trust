"""Frozen ViT encoders with every block exposed: AST and HeAR.

AST: MIT/ast-finetuned-audioset-10-10-0.4593 (AudioSet + ImageNet), the checkpoint Patch-Mix CL starts from.
HeAR: google/hear-pytorch (ViT-L MAE on 2 s health-acoustic clips; gated). Needs `../repos/hear` for its audio preprocessing.
Neither saw ICBHI or HF_Lung in pretraining.
"""
import importlib.util
import sys
from pathlib import Path

import torch

from src.encoders.base import DEVICE, Encoder, to_frames, to_frames_batch

HEAR_UTILS = Path(__file__).resolve().parents[3] / "repos" / "hear" / "python" / "data_processing" / "audio_utils.py"


class AstViT(Encoder):
    """AST: 12 blocks; 12 (freq) x 101 (time) patches for 10.24 s, the 8 s of audio fill the first 79 time columns."""
    n_layers, dim, token_shape = 12, 768, (2 + 12 * 79, 768)

    def __init__(self, name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        from transformers import ASTFeatureExtractor, ASTModel
        self.fe = ASTFeatureExtractor.from_pretrained(name)
        self.model = ASTModel.from_pretrained(name).eval().to(DEVICE).requires_grad_(False)

    @staticmethod
    def _grid(h):
        """(1, 2 + 12 * 101, D) hidden states -> (12, 79, D) patches of the real audio."""
        return h[0, 2:].reshape(12, 101, -1)[:, :79]

    @staticmethod
    def patch_index():
        return torch.arange(2, 2 + 12 * 79)

    @staticmethod
    def token_frames(tok):
        B, _, D = tok.shape
        return to_frames_batch(tok[:, 2:].reshape(B, 12, 79, D).mean(1))

    def logmel(self, wave):
        """(1024, 128) normalised fbank (affine in the log-mel; the 8 s of audio fill the first ~798 frames)."""
        return self.fe(wave, sampling_rate=16000, return_tensors="pt")["input_values"][0].numpy()

    @torch.no_grad()
    def run(self, wave, keep, token_layer):
        x = self.fe(wave, sampling_rate=16000, return_tensors="pt")["input_values"]
        if self.a2 is not None:
            x = torch.as_tensor(self.a2(x[0].numpy()), dtype=x.dtype)[None]
        h = self.model.embeddings(x.to(DEVICE))
        frames, tok = {}, None
        for i, layer in enumerate(self.model.layers, 1):
            h = layer(h)
            if i in keep:  # the last block goes through the encoder's final LayerNorm
                frames[i] = to_frames(self._grid(self.model.layernorm(h) if i == self.n_layers else h).mean(0))
            if i == token_layer:
                tok = torch.cat([h[0, :2], self._grid(h).reshape(-1, self.dim)])  # 2 cls + 12 x 79
        return frames, tok


class HeARViT(Encoder):
    """HeAR: ViT-L, 24 blocks; an 8 s cycle = four 2 s clips of 12 (time) x 8 (mel) patches + cls."""
    n_layers, dim, token_shape, units = 24, 1024, (4 * 97, 1024), 4

    @staticmethod
    def patch_index():
        return torch.arange(4 * 97).reshape(4, 97)[:, 1:].flatten()

    @staticmethod
    def token_frames(tok):
        B, _, D = tok.shape
        return to_frames_batch(tok.reshape(B, 4, 97, D)[:, :, 1:].reshape(B, 4, 12, 8, D).mean(3).reshape(B, 48, D))

    def __init__(self, name="google/hear-pytorch"):
        from transformers import AutoModel
        old, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # ../repos is read-only reference
        spec = importlib.util.spec_from_file_location("hear_audio_utils", HEAR_UTILS)
        self.utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.utils)
        sys.dont_write_bytecode = old
        self.model = AutoModel.from_pretrained(name).eval().to(DEVICE).requires_grad_(False)

    @torch.no_grad()
    def run(self, wave, keep, token_layer):
        h = self.model.embeddings(self.utils.preprocess_audio(torch.tensor(wave).reshape(4, 32000)).to(DEVICE))
        frames, tok = {}, None
        for i, layer in enumerate(self.model.layers, 1):
            h = layer(h)
            if i in keep:
                f = self.model.layernorm(h) if i == self.n_layers else h
                frames[i] = to_frames(f[:, 1:].reshape(4, 12, 8, -1).mean(2).reshape(48, -1))  # mel-averaged, clips joined in time
            if i == token_layer:
                tok = h.reshape(-1, h.shape[-1])  # 4 x 97
        return frames, tok
