import numpy as np

from src.data.hf_lung import _sec, frame_phase


def test_sec():
    assert _sec("00:00:02.608") == 2.608


def test_frame_phase_windows():
    labels = [("I", 0.0, 2.0), ("E", 2.0, 4.0), ("D", 0.0, 8.0)]  # crackle label ignored
    p = frame_phase(labels, 0.0, sec=8.0, n_frames=8)  # 1 s frames
    assert p.tolist() == [1, 1, 2, 2, 0, 0, 0, 0]
    assert frame_phase(labels, 7.0, sec=8.0, n_frames=8).sum() == 0  # window past all phases
