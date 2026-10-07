from src.legacy.hf_lung import _sec, frame_phase


def test_sec():
    assert _sec("00:00:02.608") == 2.608


def test_frame_phase_windows():
    labels = [("I", 0.0, 2.0), ("E", 2.0, 4.0), ("D", 0.0, 8.0)]  # crackle label ignored
    p = frame_phase(labels, 0.0, sec=8.0, n_frames=8)  # 1 s frames
    assert p.tolist() == [1, 1, 2, 2, 0, 0, 0, 0]
    assert frame_phase(labels, 7.0, sec=8.0, n_frames=8).sum() == 0  # window past all phases


def test_window_label():
    from src.legacy.hf_ablate import window_label
    lab = [("I", 0.0, 2.0), ("D", 1.0, 1.5), ("Wheeze", 9.0, 10.0)]
    assert window_label(lab, 0.0) == 1  # crackle only in 0-8 s
    assert window_label(lab, 7.0) == 2  # wheeze only in 7-15 s
    assert window_label([("Rhonchi", 7.5, 8.5), ("D", 0.0, 0.1)], 0.0) == 3
    assert window_label([("I", 0.0, 2.0)], 0.0) == 0
