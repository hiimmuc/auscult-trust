"""KAUH loader (Fraiwan et al. 2021): 112 subjects, Littmann 3200, every recording exported with three software filters.

Filename: `<F>P<n>_<diagnosis>,<sound>,<site>,<age>,<sex>.wav` with F = B (Bell), D (Diaphragm), E (Extended). The
three renderings of subject n are one acoustic event, so they share a patient id and must share a split.

Unit: one whole recording (about 13.5 s, 4 kHz), one recording-level sound label. Sound codes: I/E inspiratory/
expiratory, W wheeze, C or Crep crackle, B bronchial, N normal.
"""
import re
from itertools import permutations
from pathlib import Path

import pandas as pd

from src.processing.splits import split_patients

FILTERS = {"B": "Bell", "D": "Diaphragm", "E": "Extended"}
ORDERED_PAIRS = list(permutations(FILTERS.values(), 2))  # 6 (source filter, target filter) pairs
_NAME = re.compile(r"^([BDE])P(\d+)_([^,]+),([^,]+),([^,]+),(\d+),([MF])$")


def sound_label(sound):
    """Map a KAUH sound code to an ICBHI class index (normal 0, crackle 1, wheeze 2, both 3).

    A recording with only a bronchial sound (`Bronchial`) has no crackle/wheeze class: returns -1 (excluded).
    """
    t = set(sound.split())
    if t == {"Bronchial"}:
        return -1
    crackle, wheeze = bool(t & {"C", "Crep"}), "W" in t
    return int(crackle) + 2 * int(wheeze)


def file_table(root):
    """One row per wav file.

    Args:
        root: KAUH directory holding `Audio Files/`.

    Returns:
        DataFrame with `stem, wav, patient, filter, diagnosis, sound, site, age, sex, label`; `label` is -1 for
        excluded recordings. `patient` is shared by the three filter renderings.
    """
    rows = []
    for f in sorted((Path(root) / "Audio Files").glob("*.wav")):
        m = _NAME.match(f.stem)
        assert m, f"unexpected KAUH file name: {f.name}"
        flt, n, dx, snd, site, age, sex = m.groups()
        rows.append({"stem": f.stem, "wav": str(f), "patient": f"kauh_P{n}", "filter": FILTERS[flt], "diagnosis": dx.strip().lower(),
                     "sound": snd.strip(), "site": site.strip(), "age": int(age), "sex": sex, "label": sound_label(snd.strip())})
    return pd.DataFrame(rows)


def partitions(df, n=5, fracs=(0.6, 0.2, 0.2)):
    """`n` random patient-disjoint train/calibration/test splits (E2 protocol of proposals A+B and H).

    Args:
        df: `file_table` output.
        n: Number of partitions; seed i for partition i.
        fracs: Patient fractions for train, cal, test.

    Returns:
        List of split dicts (name -> patient ids) usable with `src.processing.splits.split_frames`. All filter
        renderings of a patient fall in one part because the key is the patient id.
    """
    return [split_patients(df.patient, fracs, seed=i, names=("train", "cal", "test")) for i in range(n)]


def rotation_partitions(df, n_folds=5, seed=0):
    """Patient-level rotation: every patient is in the test part exactly once (halves the CI width of 60/20/20 x 5).

    Fold i tests on patient group i, calibrates on group (i + 1) % n, trains on the rest. All filter renderings of a
    patient share a group because the key is the patient id.

    Args:
        df: `file_table` output.
        n_folds: Number of groups and partitions.
        seed: RNG seed for the patient shuffle.

    Returns:
        List of `n_folds` split dicts (train/cal/test -> patient ids).
    """
    import numpy as np
    ids = sorted(set(df.patient))
    np.random.default_rng(seed).shuffle(ids)
    groups = [sorted(g) for g in np.array_split(np.array(ids, dtype=object), n_folds)]
    return [{"train": sorted(p for j, g in enumerate(groups) if j not in (i, (i + 1) % n_folds) for p in g),
             "cal": groups[(i + 1) % n_folds], "test": groups[i]} for i in range(n_folds)]


def windows(df, sec=8.0):
    """Cut every labelled recording into `max(1, floor(duration / sec))` windows of `sec` seconds, evenly spread.

    KAUH recordings last 8.7-24.9 s; the encoders take 8 s. Each window inherits the recording's label, so the
    unit of Tier 0 KAUH experiments is the window (a recording-level label on a window is an [Assumption]).
    Recordings without a crackle/wheeze class (`label` -1) are dropped.

    Args:
        df: `file_table` output.
        sec: Window length in seconds.

    Returns:
        DataFrame with the `df` columns plus `start`, `end` (seconds) and `window` (index in the recording).
    """
    import numpy as np
    import soundfile as sf
    rows = []
    for r in df[df.label >= 0].itertuples(index=False):
        i = sf.info(r.wav)
        dur = i.frames / i.samplerate
        n = max(1, int(dur // sec))
        for w, s in enumerate(np.linspace(0, max(dur - sec, 0), n)):
            rows.append({**r._asdict(), "start": round(float(s), 3), "end": round(float(min(s + sec, dur)), 3), "window": w})
    return pd.DataFrame(rows)
