import pytest
from src.paths import DATA
from src.processing.kauh import ORDERED_PAIRS, file_table, partitions, sound_label
from src.processing.splits import assert_disjoint, split_frames

ROOT = str(DATA / "raw" / "kauh")


def test_sound_label():
    assert [
        sound_label(s) for s in ("N", "Crep", "C", "E W", "I C E W", "I C B", "Bronchial")
    ] == [0, 1, 1, 2, 3, 1, -1]


def test_ordered_pairs():
    assert (
        len(ORDERED_PAIRS) == 6
        and ("Bell", "Extended") in ORDERED_PAIRS
        and ("Bell", "Bell") not in ORDERED_PAIRS
    )


@pytest.mark.skipif(not __import__("pathlib").Path(ROOT).exists(), reason="KAUH data not present")
def test_real_table_and_partitions():
    df = file_table(ROOT)
    assert (
        len(df) == 336
        and df.patient.nunique() == 112
        and (df.groupby("patient").size() == 3).all()
    )
    assert (df.groupby("patient").label.nunique() == 1).all() and (
        df.groupby("patient")["filter"].nunique() == 3
    ).all()
    for sp in partitions(df):
        assert_disjoint(sp)
        parts = split_frames(df, sp)
        assert all((p.groupby("patient").size() == 3).all() for p in parts.values())


@pytest.mark.skipif(not __import__("pathlib").Path(ROOT).exists(), reason="KAUH data not present")
def test_windows_cover_recordings_and_keep_patient_groups():
    from src.processing.kauh import windows

    df = file_table(ROOT)
    w = windows(df)
    assert (
        (w.end - w.start).max() <= 8.0 + 1e-6
        and (w.label >= 0).all()
        and w.patient.nunique() == 111
    )
    assert (w.groupby("stem").window.min() == 0).all() and (w.groupby("stem").size() >= 1).all()
    assert w.stem.nunique() == 333


def test_rotation_partitions_each_patient_tested_once_and_disjoint():
    import pandas as pd
    from src.processing.kauh import rotation_partitions

    df = pd.DataFrame({"patient": [f"kauh_P{i}" for i in range(23) for _ in range(3)]})
    parts = rotation_partitions(df, 5)
    tested = [p for s in parts for p in s["test"]]
    assert sorted(tested) == sorted(set(df.patient))
    for s in parts:
        assert (
            not set(s["train"]) & set(s["cal"])
            and not set(s["train"]) & set(s["test"])
            and not set(s["cal"]) & set(s["test"])
        )


def test_b2_identical_conditions_have_zero_paired_interval(tmp_path, monkeypatch):
    import sys
    import json
    import numpy as np
    from src.evaluation import kauh

    rng = np.random.default_rng(5)
    patients = np.repeat(np.arange(12), 3)
    filters = np.tile(list(kauh.FILTERS.values()), 12)
    probs = rng.dirichlet(np.ones(4), len(patients))
    for condition in ("base", "same"):
        (tmp_path / condition).mkdir()
        for seed in (0, 1):
            np.savez(tmp_path / condition / f"seed{seed}.npz", probs=probs,
                     patient=patients, filter=filters, window=np.zeros(len(patients), int),
                     stem=np.array([f"{p}_{f}" for p, f in zip(patients, filters)]),
                     label=np.repeat(np.arange(12) % 2, 3))
    monkeypatch.setattr(kauh, "rotation_folds", lambda: [(set(range(6)), set(range(6, 12))),
                                                        (set(range(6, 12)), set(range(6)))])
    calls = []
    def null(*args, **kwargs):
        calls.append(kwargs["n_splits"])
        return -1.0  # Exercise B2 on every pair, regardless of the synthetic null gate.
    monkeypatch.setattr(kauh, "null_delta_p95", null)
    monkeypatch.setattr(sys, "argv", ["kauh", "--dir", str(tmp_path), "--base", "base",
                                     "--arms", "same", "--boot", "8"])
    kauh.main()
    result = json.loads((tmp_path / "kauh_summary.json").read_text())
    assert result["D_same"] == [0.0, 0.0, 0.0]
    assert set(calls) == {200}
