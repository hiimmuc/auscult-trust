import torch

from src.training.patchmix_cl.pcmcl import COMBINATIONS, four_class, four_class_probs, multilabel


def test_combinations_are_the_or_of_their_parts_and_prediction_matches_the_threshold_rule():
    for parts, target in COMBINATIONS:
        assert list(target) == [max(multilabel(k)[j] for k in parts) for j in range(3)]
    logits = torch.randn(1000, 3) * 3
    p = four_class_probs(logits)
    assert torch.allclose(p.sum(1), torch.ones(1000))
    s = torch.sigmoid(logits)
    rule = (s[:, 1] > 0.5).long() + 2 * (s[:, 2] > 0.5).long()  # reference: threshold 0.5, both -> 3
    assert torch.equal(p.argmax(1), rule)
    assert four_class(torch.tensor([[1., 0, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]])).tolist() == [0, 1, 2, 3]
