"""ICBHI 2017 loader. Builds a per-cycle table from filenames and annotation files.

Filename: `<patient>_<rec>_<chest loc>_<mode>_<device>.wav`, e.g. `101_1b1_Al_sc_Meditron.wav`.
Annotation `.txt`: `start end crackle wheeze` per line.
"""
from pathlib import Path

import pandas as pd

CLASSES = ["normal", "crackle", "wheeze", "both"]
DEVICES = ["Meditron", "LittC2SE", "Litt3200", "AKGC417L"]  # device ids 0..3 in training batches and exports
OFFICIAL_SPLIT = "ICBHI_challenge_train_test.txt"


def parse_name(stem):
    """Split an ICBHI file stem into its metadata fields.

    Args:
        stem: File name without extension.

    Returns:
        Dict with `patient`, `device`, `site`.
    """
    p = stem.split("_")
    return {"patient": f"icbhi_{p[0]}", "site": p[2], "device": p[4]}


def cycle_table(root):
    """Build one row per annotated breath cycle.

    Args:
        root: Directory holding the wav and txt files. The official split file is read from
            `root` or its parent (optional).

    Returns:
        DataFrame with `stem, wav, patient, site, device, start, end, label, official`.
        `label` indexes `CLASSES`. `official` is "train"/"test" or None.
    """
    root = Path(root)
    official = {}
    f = next((d / OFFICIAL_SPLIT for d in (root, root.parent) if (d / OFFICIAL_SPLIT).exists()), None)
    if f:
        official = dict(line.split() for line in f.read_text().splitlines() if line.strip())
    rows = []
    for txt in sorted(root.glob("*.txt")):
        if not txt.with_suffix(".wav").exists():  # skip metadata txt files
            continue
        meta = parse_name(txt.stem)
        for line in txt.read_text().splitlines():
            s, e, c, w = line.split()[:4]
            rows.append({**meta, "stem": txt.stem, "wav": str(txt.with_suffix(".wav")),
                         "start": float(s), "end": float(e), "label": int(c) + 2 * int(w),
                         "official": official.get(txt.stem)})
    return pd.DataFrame(rows)

