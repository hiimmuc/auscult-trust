import numpy as np

from src.ablate import summarize


def _run(ic, rc):
    return {"icbhi_score": ic, "ece": 0.1, "coverage": 0.9, "set_size": 1.5,
            "recall": [0.5, rc, 0.5, 0.5], "f1": [0.5, rc, 0.5, 0.5]}


def test_summarize_flags_gain_only_above_seed_sd():
    base = [_run(0.50 + d, 0.40 + d) for d in (-0.01, 0.0, 0.01)]
    better = [_run(0.60 + d, 0.40 + d) for d in (-0.01, 0.0, 0.01)]
    s = summarize({"fm": base, "fm+branch": better})
    assert s["fm+branch"]["real_gain"]["icbhi_score"] is True
    assert s["fm+branch"]["real_gain"]["recall_crackle"] is False
