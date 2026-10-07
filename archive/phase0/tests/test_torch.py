import pytest

torch = pytest.importorskip("torch")

from src.legacy.net import LungModel, phase_pool  # noqa: E402
from src.legacy.tent import norm_params, tent_adapt  # noqa: E402


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


def test_long_window_branch_control_runs():
    assert LungModel(16, use_branch=True, use_phase=False, branch_win=1024)(**_batch()).shape == (4, 4)


def test_fit_eval_smoke():
    from src.legacy.train import device, fit_eval

    def part(n):
        b = _batch(b=n)
        return {k: v.to(device) for k, v in b.items()}, torch.arange(n, device=device) % 4
    cfg = {"use_branch": True, "use_phase": True, "lr": 1e-3, "wd": 0.0, "epochs": 2, "bs": 4, "alpha": 0.1}
    m, pt = fit_eval(cfg, 0, {"train": part(8), "val": part(8), "test": part(8)})
    assert pt.shape == (8, 4) and len(m["test_pred"]) == 8 and 0 <= m["icbhi_score"] <= 1


def test_encoder_registry_names():
    from src.encoders import ENCODERS
    assert set(ENCODERS) == {"opera_ct", "opera_ce", "ast_audioset", "hear", "clap_htsat"}


def test_head_ignores_padded_frames():
    from src.auscult_trust.model import Head
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[:, :10] = True
    f = torch.randn(2, 3, 32, 16)
    g = f.clone()
    g[:, :, 10:] = torch.randn(2, 3, 22, 16) * 5  # change only the padded frames
    for lm in ("last", "concat", "scalar"):
        for pool in ("mean", "max", "meanmax", "attn"):
            h = Head(16, 3, lm, pool, hidden=8).eval()
            assert torch.allclose(h(f, mask), h(g, mask), atol=1e-5), (lm, pool)


def test_auscult_trust_masked_pool_ignores_padding():
    """Frames outside the mask must not change the output (all four pools)."""
    import torch
    from src.auscult_trust.model import AuscultTrust, Head
    torch.manual_seed(0)
    f = torch.randn(2, 1, 32, 16)
    mask = torch.arange(32)[None].expand(2, 32) < 10
    g = f.clone()
    g[:, :, 10:] = torch.randn(2, 1, 22, 16)
    for pool in ("mean", "max", "meanmax", "attn"):
        m = AuscultTrust(Head(16, 1, "last", pool, 8)).eval()
        assert torch.allclose(m({"f": f, "mask": mask}), m({"f": g, "mask": mask}), atol=1e-5), pool
