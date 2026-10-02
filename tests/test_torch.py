import pytest

torch = pytest.importorskip("torch")

from src.models.net import LungModel, phase_pool  # noqa: E402
from src.tta.tent import norm_params, tent_adapt  # noqa: E402


def _batch(b=4, n=20, d=16, t=8000):
    return {"emb": torch.randn(b, n, d), "wave": torch.randn(b, t),
            "phase": torch.randint(0, 3, (b, n))}


def test_forward_all_ablation_cells():
    for br in (False, True):
        for ph in (False, True):
            m = LungModel(16, use_branch=br, use_phase=ph)
            assert m(**_batch()).shape == (4, 4)


def test_phase_pool_empty_phase_is_zero():
    x = torch.ones(1, 3, 2)
    out = phase_pool(x, torch.tensor([[1, 1, 0]]))
    assert out.tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_tent_changes_only_norm_params():
    m = LungModel(16)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    a = tent_adapt(m, [_batch()], lr=0.1)
    norm_ids = {id(p) for p in norm_params(a)}
    for (k, p), (_, q) in zip(a.named_parameters(), m.named_parameters()):
        if id(p) not in norm_ids:
            assert torch.equal(p, q), k
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())  # original untouched
