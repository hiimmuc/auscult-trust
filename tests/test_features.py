import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import encoders, features  # noqa: E402
from src.encoders.base import Encoder, to_frames, to_frames_batch  # noqa: E402
from src.encoders.vit import AstViT, HeARViT  # noqa: E402


def test_batched_pooling_matches_single():
    x = torch.randn(3, 48, 5)
    assert torch.allclose(to_frames_batch(x), torch.stack([to_frames(v) for v in x]), atol=1e-6)
    assert to_frames(torch.randn(7, 5)).shape == (32, 5)


def test_token_frames_layout():
    assert AstViT.token_frames(torch.randn(2, 2 + 12 * 79, 6)).shape == (2, 32, 6)
    assert HeARViT.token_frames(torch.randn(2, 4 * 97, 6)).shape == (2, 32, 6) and HeARViT.units == 4


def test_registry_lists_all_encoders_and_resolves_classes_without_loading():
    assert set(encoders.ENCODERS) == {"opera_ct", "opera_ce", "ast_audioset", "hear", "clap_htsat"}
    assert all(issubclass(encoders.get(n), Encoder) for n in encoders.ENCODERS)


class _Stub(Encoder):
    n_layers, dim = 1, 4

    def run(self, wave, keep, token_layer):
        return {1: torch.full((32, 4), len(wave) / 1000)}, None


def _cycles(root):
    from scipy.io import wavfile
    wavfile.write(root / "101_1b1_Al_sc_Meditron.wav", 16000, (np.random.randn(16000 * 6) * 1000).astype(np.int16))
    (root / "101_1b1_Al_sc_Meditron.txt").write_text("0.0 2.0 0 0\n2.0 4.0 1 0\n4.0 5.5 0 1\n")


def test_extract_without_tokens_is_resumable_and_writes_meta(tmp_path, monkeypatch):
    _cycles(tmp_path)
    monkeypatch.setattr(encoders, "load", lambda name: _Stub())
    cfg = {"icbhi_root": str(tmp_path), "cache_root": str(tmp_path / "cache"), "name": "stub", "encoder": "stub"}
    out = tmp_path / "cache" / "stub"
    features.extract(cfg, limit=1)
    assert np.load(out / "done.npy").tolist() == [True, False, False]
    features.extract(cfg)
    layers = np.load(out / "layers.npy")
    assert layers.shape == (3, 1, 32, 4) and (layers[:, 0, 0, 0] == 128).all() and not (out / "tokens.npy").exists()
    assert json.loads((out / "meta.json").read_text()) == {"encoder": "stub", "keep_layers": [1], "token_layer": None}


def test_extract_with_correction_applies_spectrum_correction(tmp_path, monkeypatch):
    import librosa
    from scipy.io import wavfile

    _cycles(tmp_path)  # single Meditron recording, 3 cycles
    rng = np.random.default_rng(0)
    gain = 1 + 0.8 * np.sin(np.linspace(0, 3, 513))
    wave = rng.normal(size=16000 * 6).astype(np.float32)
    gained = librosa.istft(librosa.stft(wave, n_fft=1024, hop_length=512) * gain[:, None], hop_length=512, length=len(wave))
    wavfile.write(tmp_path / "102_1b1_Al_sc_AKGC417L.wav", 16000, (gained * 1000).astype(np.int16))
    (tmp_path / "102_1b1_Al_sc_AKGC417L.txt").write_text("0.0 2.0 0 0\n")

    seen_waves = []

    class _RecordingStub(_Stub):
        def run(self, wave, keep, token_layer):
            seen_waves.append(np.asarray(wave))
            return super().run(wave, keep, token_layer)

    monkeypatch.setattr(encoders, "load", lambda name: _RecordingStub())
    cfg = {
        "icbhi_root": str(tmp_path), "cache_root": str(tmp_path / "cache"), "name": "stub_corrected",
        "encoder": "stub", "correction": {"source_devices": ["Meditron"]},
    }
    features.extract(cfg)
    # Row order follows cycle_table (sorted by stem): "101_..." (Meditron) cycles first, then "102_..." (AKGC417L).
    from src.data.audio import cycle_wave, load_wav
    akg_wave = seen_waves[3]
    uncorrected_akg = cycle_wave(load_wav(str(tmp_path / "102_1b1_Al_sc_AKGC417L.wav")), 0.0, 2.0)
    assert not np.allclose(akg_wave, uncorrected_akg, atol=1e-4)
