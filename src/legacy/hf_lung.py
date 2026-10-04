"""HF_Lung_V1 loader: 15 s recordings, label files `<stem>_label.txt` with lines `<type> <start> <end>`.

Types: I inspiration, E expiration, D crackle (discontinuous), Wheeze, Stridor, Rhonchi.
Files with the same date likely come from one subject, so groups are keyed by date.
"""
import re
from pathlib import Path

import numpy as np



def _sec(t):
    """Parse `HH:MM:SS.mmm` to seconds.

    Args:
        t: Time string.

    Returns:
        Seconds as float.
    """
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def read_labels(path):
    """Read one label file.

    Args:
        path: Path to `<stem>_label.txt`.

    Returns:
        List of (type, start_s, end_s).
    """
    return [(p[0], _sec(p[1]), _sec(p[2])) for p in (l.split() for l in Path(path).read_text().splitlines()) if len(p) == 3]


def file_table(root):
    """List recordings with their subject-group key.

    Args:
        root: HF_Lung_V1 directory containing `train/` and `test/`.

    Returns:
        List of dicts `{wav, label, group, official}`. `group` is the date in the file name.
    """
    rows = []
    for part in ("train", "test"):
        for wav in sorted((Path(root) / part).glob("*.wav")):
            date = re.search(r"(\d{4}-?\d{2}-?\d{2})", wav.stem).group(1).replace("-", "")
            rows.append({"wav": str(wav), "label": str(wav.with_name(wav.stem + "_label.txt")),
                         "group": date, "official": part})
    return rows


def frame_phase(labels, start, sec=8.0, n_frames=32):
    """Per-frame phase mask for one window: 0 none, 1 inspiration, 2 expiration.

    Args:
        labels: Output of `read_labels`.
        start: Window start in seconds.
        sec: Window length in seconds.
        n_frames: Encoder frames in the window.

    Returns:
        (n_frames,) int array. A frame takes the phase covering most of it.
    """
    edges = start + np.linspace(0, sec, n_frames + 1)
    cover = np.zeros((3, n_frames))
    for t, s, e in labels:
        if t in ("I", "E"):
            k = 1 if t == "I" else 2
            cover[k] += np.clip(np.minimum(edges[1:], e) - np.maximum(edges[:-1], s), 0, None)
    return np.where(cover.max(0) > 0.5 * (edges[1] - edges[0]), cover.argmax(0), 0)
