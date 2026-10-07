"""Frozen CLAP audio tower (laion/clap-htsat-unfused, HTS-AT, 48 kHz; Apache-2.0). Not pretrained on ICBHI or HF_Lung."""
import torch
import torchaudio.functional as AF

from src.encoders.base import DEVICE, Encoder, to_frames


class ClapHtsat(Encoder):
    """The final HTS-AT map is 768 x 2 (freq) x 32 (time) for a 10 s input; the extractor repeat-pads 8 s to 10 s."""
    n_layers, dim = 1, 768

    def __init__(self, name="laion/clap-htsat-unfused"):
        from transformers import ClapFeatureExtractor, ClapModel
        self.fe = ClapFeatureExtractor.from_pretrained(name)
        self.model = ClapModel.from_pretrained(name).eval().to(DEVICE).requires_grad_(False)

    @torch.no_grad()
    def run(self, wave, keep, token_layer):
        j = self.fe(AF.resample(torch.tensor(wave), 16000, 48000).numpy(), sampling_rate=48000, return_tensors="pt")
        h = self.model.audio_model(input_features=j["input_features"].to(DEVICE),
                                   is_longer=j["is_longer"].to(DEVICE)).last_hidden_state[0]  # (768, 2, 32)
        return {1: to_frames(h.permute(1, 2, 0).mean(0))}, None
