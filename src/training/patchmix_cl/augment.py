"""Train-time augmentations: SpecAugment, Freq-MixStyle (P4) and random per-bin gain (P3).

Image layout used by the dataset: (time, mel, 1) numpy, or (B, 1, time, mel) tensors after `ToTensor`.
"""

import random

import numpy as np
import torch
from src.processing.correction import random_bin_gain


class SpecAugment(torch.nn.Module):
    """SpecAugment frequency and time masking with the `icbhi_ast_sup` policy of Patch-Mix CL (no time warp).

    Masks are filled with the image mean. Input (channel, time, freq), output the same shape.
    """

    F, M_F, T, M_T, P = (
        48,
        2,
        160,
        2,
        1.0,
    )  # max mask widths for a 128-mel x 798-frame image, mask counts, application probability

    @staticmethod
    def _mask(mel, axis, max_width, count):
        value = mel.mean()
        size = mel.shape[axis]
        for _ in range(count):
            width = int(np.random.uniform(0, max_width))
            start = random.randint(0, size - width)
            if axis == 1:
                mel[:, start : start + width, :] = value
            else:
                mel[:, :, start : start + width] = value
        return mel

    def forward(self, img):
        mel = img.transpose(2, 1)  # (channel, freq, time)
        if self.P >= torch.randn(1):
            # widths scale with the image so that every encoder's spectrogram is masked in the same proportion
            mel = self._mask(
                self._mask(mel, 1, self.F * mel.shape[1] / 128, self.M_F),
                2,
                self.T * mel.shape[2] / 798,
                self.M_T,
            )
        return mel.transpose(2, 1)


def random_bin_gain_image(image, max_db, db_scale, rng):
    """Smooth random per-mel-bin gain on a (time, mel, 1) dataset image.

    `max_db` is the gain SD in dB of the raw log-mel; `db_scale` converts dB to the units of the image (encoder specific).
    """
    if db_scale is None:
        raise ValueError("random gain needs an encoder whose image is a log-mel")
    out = random_bin_gain(image[..., 0], max_db * db_scale, rng=rng)
    return out[..., None].astype(image.dtype)


def random_bin_gain_batch(x, sd_db, db_scale, knots=6):
    """GPU version of the random per-bin gain: one smooth curve per sample, added to every frame of (B, 1, time, freq) images.

    Knot values are N(0, sd_db) dB, linearly interpolated over the frequency axis (as `random_bin_gain`), then converted to
    image units with `db_scale`.
    """
    k = torch.randn(x.shape[0], 1, knots, device=x.device, dtype=torch.float32) * (sd_db * db_scale)
    curve = torch.nn.functional.interpolate(k, size=x.shape[-1], mode="linear", align_corners=True)  # (B, 1, freq)
    return x + curve[:, :, None, :].to(x.dtype)


def js_divergence(logits_a, logits_b):
    """Jensen-Shannon divergence between two softmax distributions, per sample mean, computed in fp32."""
    p, q = torch.softmax(logits_a.float(), 1), torch.softmax(logits_b.float(), 1)
    m = (0.5 * (p + q)).clamp_min(1e-8).log()
    kl = lambda a: (a * (a.clamp_min(1e-8).log() - m)).sum(1)
    return (0.5 * kl(p) + 0.5 * kl(q)).mean()


def freq_mixstyle(x, p=0.5, alpha=0.1, eps=1e-6):
    """P4: Freq-MixStyle. Mix per-frequency-bin statistics (over time) between random samples of the batch.

    Args:
        x: (B, 1, time, mel) tensor.
        p: Probability of applying the mix to the batch.
        alpha: Beta(alpha, alpha) concentration.
        eps: Variance floor.

    Returns:
        Tensor of the same shape.
    """
    if torch.rand(1).item() > p or x.shape[0] < 2:
        return x
    mu, var = x.mean(2, keepdim=True), x.var(2, keepdim=True)
    sig = (var + eps).sqrt()
    xn = (x - mu) / sig
    lam = (
        torch.distributions.Beta(alpha, alpha).sample((x.shape[0], 1, 1, 1)).to(x.device, x.dtype)
    )
    perm = torch.randperm(x.shape[0], device=x.device)
    return xn * (lam * sig + (1 - lam) * sig[perm]) + (lam * mu + (1 - lam) * mu[perm])


def noise_mix_image(image, noise, snr_db, db_scale, rng):
    """Add a noise recording to a (time, mel, 1) log-mel image at `snr_db`, in the mel power domain.

    Both images are Kaldi fbanks (natural log of mel power) with the same affine normalisation, `db_scale` image units per dB,
    so power adds as k * logaddexp(x / k, n / k + log g) with k = image units per nat; the offset cancels. Cross terms between
    signal and noise are dropped (their expected value is zero). The noise is rolled by a random number of frames, and g sets
    the ratio of mean mel power (signal over noise) to `snr_db`.
    """
    k = db_scale * 10 / np.log(10)
    x, n = image.astype(np.float64) / k, np.roll(noise, rng.integers(noise.shape[0]), axis=0).astype(np.float64) / k
    log_g = np.log(np.mean(np.exp(x - x.max()))) + x.max() - np.log(np.mean(np.exp(n - n.max()))) - n.max() - snr_db * np.log(10) / 10
    return (k * np.logaddexp(x, n + log_g)).astype(image.dtype)
