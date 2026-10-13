"""The PAFA loss used in training is the reference implementation, loaded by path; check it on a fixed tensor."""

import torch

from src.training.patchmix_cl.upstream import PAFALoss


def test_pafa_loss_on_fixed_tensor():
    f = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [10.0, 2.0]])
    pid = torch.tensor([1, 1, 2, 2])
    # within = 2 + 2 = 4; centroids (1, 0), (10, 1): between = 81 + 1 = 82; global (5.5, 0.5): gpal = mean(20.5, 20.5) = 20.5
    got = PAFALoss()(f, pid, lambda_pcsl=5.0, lambda_gpal=0.005).item()
    assert abs(got - (5.0 * 4 / (82 + 1e-6) + 0.005 * 20.5)) < 1e-5
