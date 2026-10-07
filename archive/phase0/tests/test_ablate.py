import numpy as np
import pandas as pd

from src.legacy.ablate import paired_diff_ci, summarize
from src.eval.metrics import icbhi_score
from src.make_split import official_split
from src.legacy.phase import decode_cycle


def _run(ic, rc, cov=0.9):
    return {"icbhi_score": ic, "ece": 0.1, "coverage": cov, "set_size": 1.5,
            "recall": [0.5, rc, 0.5, 0.5], "f1": [0.5, rc, 0.5, 0.5]}


def test_summarize_flags_only_positive_perf_gain_above_seed_sd():
    base = [_run(0.50 + d, 0.40 + d) for d in (-0.01, 0.0, 0.01)]
    better = [_run(0.60 + d, 0.40 + d, cov=0.5) for d in (-0.01, 0.0, 0.01)]
    worse = [_run(0.40 + d, 0.40 + d) for d in (-0.01, 0.0, 0.01)]
    s = summarize({"fm": base, "fm+branch": better, "fm+phase": worse})
    assert s["fm+branch"]["real_gain"]["icbhi_score"] is True
    assert s["fm+branch"]["real_gain"]["recall_crackle"] is False
    assert "coverage" not in s["fm+branch"]["real_gain"]  # coverage change is not a gain
    assert s["fm+phase"]["real_gain"]["icbhi_score"] is False  # a loss is not a gain


def test_paired_diff_ci_sign():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    pat = np.repeat(np.arange(40), 10)
    a = np.stack([rng.integers(0, 4, 400) for _ in range(3)], 1)  # chance
    b = np.stack([y] * 3, 1)  # perfect
    est, lo, hi = paired_diff_ci(y, pat, a, b, icbhi_score, n_boot=100)
    assert lo > 0 and lo <= est <= hi


def test_decode_cycle_one_boundary_either_order():
    p = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.5, 0.5, 0.5])
    assert decode_cycle(p, 5).tolist() == [1, 1, 2, 2, 2, 1, 1, 2]  # 5 real frames, tiled
    assert decode_cycle(1 - p, 5).tolist() == [2, 2, 1, 1, 1, 2, 2, 1]
    assert set(decode_cycle(p, 8)) == {1, 2}


def test_official_split_keeps_test_and_drops_overlap():
    df = pd.DataFrame({"patient": ["a", "a", "b", "c", "d"], "stem": ["a1", "a2", "b1", "c1", "d1"]})
    s = official_split(df, {"a1": "train", "a2": "test", "b1": "test", "c1": "train", "d1": "train"})
    assert s["test"] == ["a", "b"] and s["drop"] == ["a1"]
    assert sorted(s["train"] + s["val"]) == ["c", "d"]
