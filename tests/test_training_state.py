import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.artifacts import atomic_json
from src.training.state import capture_rng, restore_rng, completed, publish_completion


def test_rng_resume_reproduces_next_draws():
    original = capture_rng()
    try:
        saved = capture_rng()
        def draw():
            return random.random(), np.random.random(4), torch.rand(4), torch.rand(4, device="cuda") if torch.cuda.is_available() else None
        expected = draw()
        draw()
        restore_rng(saved)
        actual = draw()
        assert expected[0] == actual[0]
        assert np.array_equal(expected[1], actual[1])
        assert torch.equal(expected[2], actual[2])
        if expected[3] is not None:
            assert torch.equal(expected[3], actual[3])
        with pytest.raises(ValueError, match="no RNG"):
            restore_rng(None)
    finally:
        restore_rng(original)


def test_completion_rejects_partial_or_modified_results(tmp_path):
    out, checkpoints = tmp_path / "unit", tmp_path / "weights"
    out.mkdir()
    checkpoints.mkdir()
    assert not completed(out)
    atomic_json(out / "report.json", {"history": [{"epoch": 1}]})
    with pytest.raises(ValueError, match="Uncertified"):
        completed(out)
    (checkpoints / "last.pth").write_bytes(b"restart state")
    assert not completed(out, checkpoints)
    (checkpoints / "report_epoch_1.pth").write_bytes(b"weights")
    publish_completion(out, checkpoints)
    assert completed(out)
    (checkpoints / "report_epoch_1.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        completed(out)


def test_config_mismatch_is_checked_before_completed_skip(tmp_path, monkeypatch):
    from src.training.patchmix_cl import main
    args = SimpleNamespace(exp="train", cell="test", unit="seed0", seed=1)
    atomic_json(tmp_path / "train_args.json", {"seed": 0})
    atomic_json(tmp_path / "report.json", {"history": [1]})
    monkeypatch.setattr(main, "parse_args", lambda: args)
    monkeypatch.setattr(main.runs, "unit_dir", lambda *a: tmp_path)
    with pytest.raises(ValueError, match="different config"):
        main.main()
