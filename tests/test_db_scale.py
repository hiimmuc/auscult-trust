"""dB -> image-unit scale of the AST fbank image: +6 dB on the waveform must move the image by 6 * db_scale.

DASS is not tested here: it normalises each cycle by its own mean and std, so a constant gain cancels out.
"""

import numpy as np
import torch

from src.training.patchmix_cl.cycles import generate_fbank
from src.training.patchmix_cl.encoders import get_encoder


def test_ast_db_scale_matches_fbank_shift():
    torch.manual_seed(0)
    wave = torch.randn(1, 16000 * 8) * 0.1
    loud = wave * 10 ** (6 / 20)  # +6 dB power in every bin (dither 0, so the log-mel shifts exactly)
    shift = np.mean(generate_fbank(loud, 16000) - generate_fbank(wave, 16000))
    target = 6 * get_encoder('ast').db_scale
    assert abs(shift - target) / target < 0.01, (shift, target)
