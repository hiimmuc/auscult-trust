"""Consistency-loss pieces: the batched gain curve and the Jensen-Shannon divergence."""

import torch

from src.training.patchmix_cl.augment import js_divergence, random_bin_gain_batch


def test_gain_curve_is_constant_over_time_and_scaled():
    torch.manual_seed(0)
    x = torch.zeros(512, 1, 7, 128)
    y = random_bin_gain_batch(x, sd_db=6.0, db_scale=0.5)
    assert torch.allclose(y[:, :, 0], y[:, :, 3])  # same curve on every frame
    knots_sd = y[:, 0, 0, 0].std().item()  # first bin = first knot, SD = 6 dB * 0.5
    assert abs(knots_sd - 3.0) < 0.3


def test_js_divergence_bounds():
    a = torch.tensor([[5.0, 0, 0, 0]])
    assert js_divergence(a, a).item() < 1e-6
    b = torch.tensor([[0, 5.0, 0, 0]])
    v = js_divergence(a, b).item()
    assert 0 < v <= torch.log(torch.tensor(2.0)).item() + 1e-6
