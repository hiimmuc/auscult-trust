import pytest

torch = pytest.importorskip("torch")

from src.auscult_trust.augment import Projector, ema_step, mixcl_loss, mixcl_term, patch_mix  # noqa: E402
from src.auscult_trust.model import AuscultTrust, Head, parse_cell  # noqa: E402
from src.encoders.vit import AstViT, HeARViT  # noqa: E402


def test_parse_cell():
    assert parse_cell("last-mean") == ("head", "last", "mean", set())
    assert parse_cell("ft-k4-attn+mix+ema") == ("ft", "k4", "attn", {"mix", "ema"})
    assert parse_cell("lora-r8-attn") == ("lora", "r8", "attn", set())
    with pytest.raises(AssertionError):
        parse_cell("last-mean+mix")


def test_patch_index_excludes_class_tokens():
    assert AstViT.patch_index().tolist() == list(range(2, 2 + 12 * 79))
    idx = HeARViT.patch_index()
    assert len(idx) == 4 * 96 and not set(idx.tolist()) & {0, 97, 194, 291}


def test_patch_mix_replaces_only_patches_and_lam_matches():
    torch.manual_seed(0)
    tok = torch.randn(6, 2 + 10, 4)
    idx = torch.arange(2, 12)
    mixed, perm, lam = patch_mix(tok, idx, alpha=1.0)
    assert torch.equal(mixed[:, :2], tok[:, :2])  # class tokens untouched
    changed = (mixed != tok).any(-1)[:, 2:]  # (B, P) positions taken from the partner
    kept = 1 - changed.any(0).float().mean()  # positions replaced for every sample share one mask
    assert abs(float(kept) - lam) < 1e-6 or lam == 1.0
    assert torch.equal(mixed[:, 2:][:, changed[0]], tok[perm][:, 2:][:, changed[0]])


def test_mixcl_loss_prefers_matching_pairs():
    z = torch.randn(8, 16)
    perm = torch.randperm(8)
    aligned = mixcl_loss(z, z.clone(), perm, 1.0, 0.06)  # lam = 1: positive is the clean sample itself
    shuffled = mixcl_loss(z, z[torch.roll(torch.arange(8), 1)], perm, 1.0, 0.06)
    assert aligned < shuffled and torch.isfinite(aligned)


def test_mixcl_term_trains_projector_and_tail():
    class StubTail(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 16)

        def forward(self, tok):
            return torch.nn.functional.adaptive_avg_pool1d(self.lin(tok).transpose(1, 2), 32).transpose(1, 2)

    m = AuscultTrust(Head(16, 1, "last", "attn", 8), StubTail(), Projector(8))
    tok, mask = torch.randn(5, 12, 4), torch.ones(5, 32, dtype=torch.bool)
    b = {"tok": tok, "mask": mask}
    loss = mixcl_term(m, b, m.embed(b), torch.arange(2, 12), {"alpha": 1.0, "temp": 0.06})
    loss.backward()
    assert torch.isfinite(loss) and m.projector.net[0].weight.grad.abs().sum() > 0 and m.tail.lin.weight.grad.abs().sum() > 0


def test_ema_step_averages_before_and_after():
    p = torch.nn.Parameter(torch.tensor([2.0]))
    ema_step([p], [torch.tensor([0.0])], beta=0.25)
    assert p.item() == pytest.approx(1.5)  # 0.25 * before + 0.75 * after
