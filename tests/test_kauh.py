import pytest

from src.data.kauh import ORDERED_PAIRS, file_table, partitions, sound_label
from src.data.splits import assert_disjoint, split_frames

ROOT = "../data/raw/kauh"


def test_sound_label():
    assert [sound_label(s) for s in ("N", "Crep", "C", "E W", "I C E W", "I C B", "Bronchial")] == [0, 1, 1, 2, 3, 1, -1]


def test_ordered_pairs():
    assert len(ORDERED_PAIRS) == 6 and ("Bell", "Extended") in ORDERED_PAIRS and ("Bell", "Bell") not in ORDERED_PAIRS


@pytest.mark.skipif(not __import__("pathlib").Path(ROOT).exists(), reason="KAUH data not present")
def test_real_table_and_partitions():
    df = file_table(ROOT)
    assert len(df) == 336 and df.patient.nunique() == 112 and (df.groupby("patient").size() == 3).all()
    assert (df.groupby("patient").label.nunique() == 1).all() and (df.groupby("patient")["filter"].nunique() == 3).all()
    for sp in partitions(df):
        assert_disjoint(sp)
        parts = split_frames(df, sp)
        assert all((p.groupby("patient").size() == 3).all() for p in parts.values())


@pytest.mark.skipif(not __import__("pathlib").Path(ROOT).exists(), reason="KAUH data not present")
def test_windows_cover_recordings_and_keep_patient_groups():
    from src.data.kauh import windows
    df = file_table(ROOT)
    w = windows(df)
    assert (w.end - w.start).max() <= 8.0 + 1e-6 and (w.label >= 0).all() and w.patient.nunique() == 111
    assert (w.groupby("stem").window.min() == 0).all() and (w.groupby("stem").size() >= 1).all()
    assert w.stem.nunique() == 333


def test_rotation_partitions_each_patient_tested_once_and_disjoint():
    import pandas as pd
    from src.data.kauh import rotation_partitions
    df = pd.DataFrame({"patient": [f"kauh_P{i}" for i in range(23) for _ in range(3)]})
    parts = rotation_partitions(df, 5)
    tested = [p for s in parts for p in s["test"]]
    assert sorted(tested) == sorted(set(df.patient))
    for s in parts:
        assert not set(s["train"]) & set(s["cal"]) and not set(s["train"]) & set(s["test"]) and not set(s["cal"]) & set(s["test"])
